@echo off
echo Building SAM2Matting portable GUI executable...

REM Check if virtual environment exists
if not exist venv (
    echo Virtual environment not found. Running the launcher once to build it...
    pwsh -NoProfile -ExecutionPolicy Bypass -File "%~dp0launcher.ps1" --setup-only
    if not exist venv (
        echo Setup failed. Run launcher.ps1 manually to see the error.
        pause
        exit /b 1
    )
)

echo Installing PyInstaller...
venv\Scripts\python.exe -m pip install pyinstaller -q
if %ERRORLEVEL% NEQ 0 (
    echo Failed to install PyInstaller.
    pause
    exit /b 1
)

echo Building executable...
venv\Scripts\python.exe -m PyInstaller SAM2MattingPortableGUI.spec --noconfirm

if %ERRORLEVEL% NEQ 0 (
    echo Build failed.
    pause
    exit /b 1
)

echo.
echo Build successful!
echo The executable can be found in the 'dist' folder: dist\SAM2MattingPortableGUI.exe
echo On first run it downloads Python plus ~3.5 GB of dependencies next to itself;
echo model checkpoints are downloaded automatically on the first matting run.
echo.
pause
