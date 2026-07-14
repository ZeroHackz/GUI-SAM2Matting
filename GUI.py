"""SAM2Matting GUI - pick an input, tweak options, watch the run live.

Runs batch_matting.py as a subprocess (with --progress) and streams its
output into the log pane. Launch with the venv python:

    E:\\repos\\SAM2Matting\\venv\\Scripts\\pythonw.exe GUI.py
"""

import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import urllib.request
import zipfile
from pathlib import Path
from tkinter import colorchooser, filedialog

import customtkinter as ctk

FROZEN = getattr(sys, "frozen", False)
if FROZEN:
    # portable exe: work next to the .exe, python + deps live in python_embedded\
    REPO_ROOT = Path(sys.executable).resolve().parent
    BUNDLE_DIR = Path(sys._MEIPASS)
    PYTHON = REPO_ROOT / "python_embedded" / "python.exe"
else:
    REPO_ROOT = Path(__file__).resolve().parent
    BUNDLE_DIR = REPO_ROOT
    PYTHON = REPO_ROOT / "venv" / "Scripts" / "python.exe"

ICON_ICO = BUNDLE_DIR / "assets" / "icon.ico"
EMBED_PY_URL = "https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip"
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
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

        if ICON_ICO.exists():
            self.iconbitmap(str(ICON_ICO))

        self.after(100, self.poll_queue)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        if FROZEN:
            self.extract_app_source()
            if not PYTHON.exists():
                self.run_btn.configure(state="disabled")
                self.status_var.set("First-run setup...")
                threading.Thread(target=self.bootstrap_thread, daemon=True).start()

    # --- portable-mode bootstrap ---
    def extract_app_source(self):
        """Copy the bundled pipeline source next to the exe.

        Plain files are refreshed on every start so a newer exe updates them;
        the sam2/sam3 trees are only extracted when missing.
        """
        src = BUNDLE_DIR / "app_src"
        for name in ("batch_matting.py", "requirements.txt"):
            shutil.copy2(src / name, REPO_ROOT / name)
        for name in ("sam2", "sam3"):
            target = REPO_ROOT / name
            if not target.exists():
                shutil.copytree(src / name, target)

    def _download(self, url, dest: Path, label: str):
        self.out_queue.put(f"Downloading {label}...")
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
            total = int(r.headers.get("Content-Length", 0))
            done, next_pct = 0, 25
            while chunk := r.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total and done * 100 // total >= next_pct:
                    self.out_queue.put(f"      {done // (1 << 20)} / {total // (1 << 20)} MB")
                    next_pct += 25

    def _stream_cmd(self, cmd):
        proc = subprocess.Popen(
            cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        for line in proc.stdout:
            self.out_queue.put(line.rstrip("\n"))
        if proc.wait() != 0:
            raise RuntimeError(f"command failed: {' '.join(map(str, cmd))}")

    def bootstrap_thread(self):
        """First run of the portable exe: embedded Python + all dependencies (~3.5 GB)."""
        try:
            pydir = PYTHON.parent
            self.out_queue.put("=== First-run setup: this downloads Python and ~3.5 GB of dependencies ===")
            self.out_queue.put(f"Everything is installed locally under:\n  {pydir}")

            zip_path = REPO_ROOT / "python_embed.zip"
            self._download(EMBED_PY_URL, zip_path, "Python 3.10.11 (embeddable)")
            pydir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(pydir)
            zip_path.unlink()

            # enable site-packages so pip-installed packages are importable
            pth = pydir / "python310._pth"
            pth.write_text(pth.read_text().replace("#import site", "import site"),
                           encoding="utf-8")

            get_pip = pydir / "get-pip.py"
            self._download(GET_PIP_URL, get_pip, "pip")
            self._stream_cmd([str(PYTHON), str(get_pip), "--no-warn-script-location"])
            get_pip.unlink()

            self.out_queue.put("Installing PyTorch (CUDA 12.8) - the big one, ~3 GB...")
            self._stream_cmd([str(PYTHON), "-m", "pip", "install", "-q",
                              "torch==2.8.0", "torchvision==0.23.0",
                              "--index-url", TORCH_INDEX])

            self.out_queue.put("Installing remaining dependencies...")
            reqs = [ln.strip() for ln in (REPO_ROOT / "requirements.txt").read_text().splitlines()
                    if ln.strip() and "torch" not in ln]
            self._stream_cmd([str(PYTHON), "-m", "pip", "install", "-q",
                              *reqs, "rembg", "onnxruntime", "triton-windows"])

            self.out_queue.put("=== Setup complete. Model checkpoints download automatically on first matting run. ===")
            self.status_var.set("Idle")
        except Exception as e:
            self.out_queue.put(f"SETUP FAILED: {e}")
            self.out_queue.put("Close the window and start the exe again to retry.")
            self.status_var.set("Setup failed - see log")
        finally:
            self.run_btn.configure(state="normal")

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
        if not PYTHON.exists():
            self.log_line("Python environment not ready - setup must finish first.")
            return
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
    if not FROZEN and not PYTHON.exists():
        sys.exit(f"venv python not found at {PYTHON} - run launcher.ps1 once to build it")
    MattingGUI().mainloop()
