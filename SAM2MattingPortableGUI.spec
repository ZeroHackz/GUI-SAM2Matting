# -*- mode: python ; coding: utf-8 -*-
# Portable GUI build. The exe contains only the GUI + pipeline source code;
# Python, PyTorch, and the model checkpoints are downloaded on first run.

a = Analysis(
    ['GUI.py'],
    pathex=['venv/Lib/site-packages', '.'],
    binaries=[],
    datas=[
        ('venv/Lib/site-packages/customtkinter', 'customtkinter'),
        ('assets/icon.ico', 'assets'),
        # pipeline source, extracted next to the exe on first run
        ('batch_matting.py', 'app_src'),
        ('requirements.txt', 'app_src'),
        ('sam2', 'app_src/sam2'),
        ('sam3', 'app_src/sam3'),
    ],
    hiddenimports=['customtkinter'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch', 'torchvision', 'numpy', 'cv2', 'PIL.ImageQt'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='SAM2MattingPortableGUI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/icon.ico',
)
