@echo off
REM SAM2Matting batch runner - pass a frames directory, video file, or single image.
REM Usage: run_matting.bat <input> [extra args, e.g. --output D:\out --bg 0,255,0]
REM launcher.ps1 handles the end-of-run pause itself.
pwsh -NoProfile -ExecutionPolicy Bypass -File "E:\repos\SAM2Matting\launcher.ps1" %*
