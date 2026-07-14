"""SAM2Matting GUI - pick an input, tweak options, watch the run live.

Runs batch_matting.py as a subprocess (with --progress) and streams its
output into the log pane. Launch with the venv python:

    E:\\repos\\SAM2Matting\\venv\\Scripts\\pythonw.exe GUI.py
"""

import os
import queue
import re
import subprocess
import sys
import threading
from pathlib import Path
from tkinter import colorchooser, filedialog

import customtkinter as ctk

REPO_ROOT = Path(__file__).resolve().parent
PYTHON = REPO_ROOT / "venv" / "Scripts" / "python.exe"
PROGRESS_RE = re.compile(r"^PROGRESS (\d+)/(\d+)")
OUTPUT_ROOT_RE = re.compile(r"^OUTPUT_ROOT (.+)$")

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

VIDEO_FILETYPES = [
    ("Media files", "*.mp4 *.mov *.avi *.mkv *.webm *.png *.jpg *.jpeg *.bmp *.webp"),
    ("All files", "*.*"),
]


class MattingGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("SAM2Matting - batch runner")
        self.geometry("860x640")
        self.minsize(720, 520)

        self.proc = None
        self.out_queue = queue.Queue()
        self.output_root = None
        self.bg_color = (0, 0, 0)

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(6, weight=1)
        pad = {"padx": 8, "pady": 5}

        # --- input row ---
        ctk.CTkLabel(self, text="Input").grid(row=0, column=0, sticky="w", **pad)
        self.input_var = ctk.StringVar()
        ctk.CTkEntry(self, textvariable=self.input_var).grid(row=0, column=1, sticky="ew", **pad)
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.grid(row=0, column=2, sticky="e", **pad)
        ctk.CTkButton(row, text="Folder...", width=70, command=self.pick_folder).pack(side="left", padx=2)
        ctk.CTkButton(row, text="File...", width=70, command=self.pick_file).pack(side="left", padx=2)

        # --- output row ---
        ctk.CTkLabel(self, text="Output").grid(row=1, column=0, sticky="w", **pad)
        self.output_var = ctk.StringVar()
        e = ctk.CTkEntry(self, textvariable=self.output_var,
                         placeholder_text="auto: <input>_<timestamp>_matting")
        e.grid(row=1, column=1, sticky="ew", **pad)
        ctk.CTkButton(self, text="Browse...", width=148, command=self.pick_output).grid(
            row=1, column=2, sticky="e", **pad)

        # --- options row ---
        opts = ctk.CTkFrame(self, fg_color="transparent")
        opts.grid(row=2, column=0, columnspan=3, sticky="ew", **pad)
        ctk.CTkLabel(opts, text="Model").pack(side="left", padx=(0, 4))
        self.variant_var = ctk.StringVar(value="sam2.1base+")
        ctk.CTkOptionMenu(opts, variable=self.variant_var, width=130,
                          values=["sam2.1base+", "sam2.1tiny", "sam3"]).pack(side="left", padx=(0, 16))
        ctk.CTkLabel(opts, text="Composite bg").pack(side="left", padx=(0, 4))
        self.bg_btn = ctk.CTkButton(opts, text="0,0,0", width=90, fg_color="#000000",
                                    hover=False, command=self.pick_bg)
        self.bg_btn.pack(side="left", padx=(0, 16))
        ctk.CTkLabel(opts, text="First-frame mask (optional)").pack(side="left", padx=(0, 4))
        self.mask_var = ctk.StringVar()
        ctk.CTkEntry(opts, textvariable=self.mask_var, width=180,
                     placeholder_text="auto (rembg)").pack(side="left", padx=(0, 2))
        ctk.CTkButton(opts, text="...", width=30, command=self.pick_mask).pack(side="left")

        # --- run / cancel / open ---
        btns = ctk.CTkFrame(self, fg_color="transparent")
        btns.grid(row=3, column=0, columnspan=3, sticky="ew", **pad)
        self.run_btn = ctk.CTkButton(btns, text="Run matting", command=self.start_run)
        self.run_btn.pack(side="left", padx=2)
        self.cancel_btn = ctk.CTkButton(btns, text="Cancel", state="disabled",
                                        fg_color="#8a3333", hover_color="#a44", command=self.cancel_run)
        self.cancel_btn.pack(side="left", padx=2)
        self.open_btn = ctk.CTkButton(btns, text="Open output folder", state="disabled",
                                      command=self.open_output)
        self.open_btn.pack(side="left", padx=2)

        # --- progress ---
        self.progress = ctk.CTkProgressBar(self)
        self.progress.grid(row=4, column=0, columnspan=3, sticky="ew", **pad)
        self.progress.set(0)
        self.status_var = ctk.StringVar(value="Idle")
        ctk.CTkLabel(self, textvariable=self.status_var, anchor="w").grid(
            row=5, column=0, columnspan=3, sticky="ew", **pad)

        # --- log ---
        self.log_box = ctk.CTkTextbox(self, font=("Consolas", 12))
        self.log_box.grid(row=6, column=0, columnspan=3, sticky="nsew", **pad)
        self.log_box.configure(state="disabled")

        self.after(100, self.poll_queue)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # --- pickers ---
    def pick_folder(self):
        if p := filedialog.askdirectory(title="Frames directory"):
            self.input_var.set(p)

    def pick_file(self):
        if p := filedialog.askopenfilename(title="Video or image", filetypes=VIDEO_FILETYPES):
            self.input_var.set(p)

    def pick_output(self):
        if p := filedialog.askdirectory(title="Output directory"):
            self.output_var.set(p)

    def pick_mask(self):
        if p := filedialog.askopenfilename(title="First-frame mask PNG",
                                           filetypes=[("PNG", "*.png"), ("All files", "*.*")]):
            self.mask_var.set(p)

    def pick_bg(self):
        rgb, _hex = colorchooser.askcolor(color=f"#{self.bg_color[0]:02x}{self.bg_color[1]:02x}{self.bg_color[2]:02x}",
                                          title="Composite background color")
        if rgb:
            self.bg_color = tuple(int(v) for v in rgb)
            self.bg_btn.configure(text=",".join(map(str, self.bg_color)), fg_color=_hex)

    # --- run control ---
    def start_run(self):
        input_path = self.input_var.get().strip().strip('"')
        if not input_path:
            self.log_line("Pick an input folder, video, or image first.")
            return
        if not Path(input_path).exists():
            self.log_line(f"Input does not exist: {input_path}")
            return

        cmd = [str(PYTHON), "-u", str(REPO_ROOT / "batch_matting.py"),
               "--input", input_path,
               "--variant", self.variant_var.get(),
               "--bg", ",".join(map(str, self.bg_color)),
               "--progress"]
        if self.output_var.get().strip():
            cmd += ["--output", self.output_var.get().strip()]
        if self.mask_var.get().strip():
            cmd += ["--mask", self.mask_var.get().strip()]

        self.output_root = None
        self.progress.set(0)
        self.status_var.set("Starting...")
        self.run_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.open_btn.configure(state="disabled")
        self.log_line("$ " + " ".join(cmd[2:]))  # hide python path noise

        self.proc = subprocess.Popen(
            cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        threading.Thread(target=self.reader_thread, daemon=True).start()

    def reader_thread(self):
        for line in self.proc.stdout:
            self.out_queue.put(line.rstrip("\n"))
        code = self.proc.wait()
        self.out_queue.put(("__EXIT__", code))

    def cancel_run(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self.status_var.set("Cancelled")
            self.log_line("--- cancelled by user ---")

    def open_output(self):
        if self.output_root and Path(self.output_root).exists():
            os.startfile(self.output_root)

    def on_close(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
        self.destroy()

    # --- output handling ---
    def poll_queue(self):
        try:
            while True:
                item = self.out_queue.get_nowait()
                if isinstance(item, tuple) and item[0] == "__EXIT__":
                    code = item[1]
                    self.run_btn.configure(state="normal")
                    self.cancel_btn.configure(state="disabled")
                    if code == 0:
                        self.progress.set(1)
                        self.status_var.set("Done")
                        if self.output_root:
                            self.open_btn.configure(state="normal")
                    elif self.status_var.get() != "Cancelled":
                        self.status_var.set(f"Failed (exit {code}) - see log")
                    continue
                self.handle_line(item)
        except queue.Empty:
            pass
        self.after(100, self.poll_queue)

    def handle_line(self, line):
        if m := PROGRESS_RE.match(line):
            done, total = int(m.group(1)), int(m.group(2))
            self.progress.set(done / total)
            self.status_var.set(f"Matting frame {done}/{total}")
            if done in (1, total) or done % 25 == 0:
                self.log_line(line)
            return
        if m := OUTPUT_ROOT_RE.match(line):
            self.output_root = m.group(1)
            return
        self.log_line(line)

    def log_line(self, text):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", text + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")


if __name__ == "__main__":
    if not PYTHON.exists():
        sys.exit(f"venv python not found at {PYTHON} - run launcher.ps1 once to build it")
    MattingGUI().mainloop()
