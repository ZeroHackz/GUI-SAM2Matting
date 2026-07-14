# SAM2Matting launcher (PowerShell 7) - self-heals venv, then runs the GUI or CLI.
# Usage: .\launcher.ps1              -> opens the GUI
#        .\launcher.ps1 <input> [..] -> CLI batch run
#   e.g. .\launcher.ps1 C:\frames_dir
#        .\launcher.ps1 C:\video.mp4 --bg 0,255,0
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$MattingArgs
)

$repo = 'E:\repos\SAM2Matting'
$python = "$repo\venv\Scripts\python.exe"
$checkpoint = "$repo\checkpoints\SAM2Matting-SAM2.1Base+.pt"

# Keep the window open when run interactively (double-click), but never
# block scripted/redirected runs.
function Exit-Launcher([int]$code) {
    if (-not [Console]::IsInputRedirected) {
        Write-Host ''
        Read-Host 'Finished - press Enter to close' | Out-Null
    }
    exit $code
}


function Install-Deps {
    & $python -m pip install --upgrade pip -q
    & $python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128 -q
    # requirements minus the torch pins (torch-tensorrt is only for --compiled mode)
    $req = Get-Content "$repo\requirements.txt" | Where-Object { $_ -notmatch 'torch' }
    $req | Set-Content "$repo\requirements-notorch.txt"
    & $python -m pip install -r "$repo\requirements-notorch.txt" rembg onnxruntime customtkinter triton-windows -q
}

# Rebuild the venv from scratch if it's missing
if (-not (Test-Path $python)) {
    Write-Host 'venv missing - rebuilding (torch download is ~3 GB)...'
    py -3.10 -m venv "$repo\venv"
    Install-Deps
    Write-Host 'venv rebuilt.'
}

# Repair a broken venv (torch failing to import = core deps damaged)
& $python -c "import torch" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host 'venv is broken (torch will not import) - reinstalling dependencies...'
    Install-Deps
    Write-Host 'dependencies reinstalled.'
}

# Re-download the checkpoint if it's missing
if (-not (Test-Path $checkpoint)) {
    Write-Host 'Checkpoint missing - downloading SAM2.1Base+ (383 MB)...'
    New-Item -ItemType Directory -Force "$repo\checkpoints" | Out-Null
    curl.exe -L -o $checkpoint "https://huggingface.co/FudanCVL/SAM2Matting/resolve/main/checkpoints/SAM2Matting-SAM2.1Base%2B.pt" --silent --show-error
    if ((Get-Item $checkpoint).Length -lt 100MB) { Write-Host 'Download looks incomplete - check your connection.'; Exit-Launcher 1 }
}

# --setup-only: build.bat uses this to ensure the venv exists, nothing else.
if ($MattingArgs -and $MattingArgs[0] -eq '--setup-only') {
    Write-Host 'Environment ready.'
    exit 0
}

# No args -> GUI (detached, no console). With args -> CLI batch run.
if (-not $MattingArgs) {
    Write-Host 'Opening SAM2Matting GUI...'
    Start-Process -FilePath "$repo\venv\Scripts\pythonw.exe" -ArgumentList "$repo\GUI.py" -WorkingDirectory $repo
    exit 0
}

& $python "$repo\batch_matting.py" --input @MattingArgs
Exit-Launcher $LASTEXITCODE
