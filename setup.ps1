<#
Create a compatible Python environment and install the project dependencies.

Run from PowerShell:
    powershell -ExecutionPolicy Bypass -File .\setup.ps1
#>

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$environmentPath = Join-Path $projectRoot ".venv"
$environmentPython = Join-Path $environmentPath "Scripts\python.exe"
$pythonInstallPath = Join-Path $projectRoot ".python"
$downloadCachePath = Join-Path $projectRoot ".uv-cache"

Set-Location -LiteralPath $projectRoot

# A newly installed uv may not appear in the current PowerShell PATH until the
# terminal is reopened. Check both PATH and uv's normal Windows install path.
$uvCommand = Get-Command "uv" -ErrorAction SilentlyContinue
if ($uvCommand) {
    $uvExecutable = $uvCommand.Source
} else {
    $uvExecutable = Join-Path $env:USERPROFILE ".local\bin\uv.exe"
    if (-not (Test-Path -LiteralPath $uvExecutable)) {
        Write-Host "The 'uv' command is required for automatic setup." -ForegroundColor Red
        Write-Host "Install it once with:"
        Write-Host 'powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"' -ForegroundColor Yellow
        exit 1
    }
}

# A partially created Python 3.14 environment cannot install Gensim on Windows.
# Remove only this project's .venv when it uses an incompatible Python version.
$createEnvironment = $true

if (Test-Path -LiteralPath $environmentPython) {
    $existingVersion = & $environmentPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    if ($existingVersion -notin @("3.11", "3.12")) {
        $resolvedProjectRoot = [System.IO.Path]::GetFullPath($projectRoot)
        $resolvedEnvironment = [System.IO.Path]::GetFullPath($environmentPath)
        if (-not $resolvedEnvironment.StartsWith($resolvedProjectRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Unsafe environment path: $resolvedEnvironment"
        }

        Write-Host "Removing incompatible Python $existingVersion environment..."
        Remove-Item -Recurse -Force -LiteralPath $resolvedEnvironment
    } else {
        Write-Host "Reusing compatible Python $existingVersion environment."
        $createEnvironment = $false
    }
} elseif (Test-Path -LiteralPath $environmentPath) {
    # A failed installation can leave a folder without a working interpreter.
    $resolvedProjectRoot = [System.IO.Path]::GetFullPath($projectRoot)
    $resolvedEnvironment = [System.IO.Path]::GetFullPath($environmentPath)
    if (-not $resolvedEnvironment.StartsWith($resolvedProjectRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Unsafe environment path: $resolvedEnvironment"
    }
    Remove-Item -Recurse -Force -LiteralPath $resolvedEnvironment
}

# Store uv's downloaded Python beside the project so setup does not depend on a
# separately installed system Python.
$env:UV_PYTHON_INSTALL_DIR = $pythonInstallPath
$env:UV_CACHE_DIR = $downloadCachePath

if ($createEnvironment) {
    Write-Host "Creating a Python 3.12 environment..." -ForegroundColor Cyan
    & $uvExecutable venv --python 3.12 $environmentPath
}

Write-Host "Installing project dependencies..." -ForegroundColor Cyan
& $uvExecutable pip install --python $environmentPython -r (Join-Path $projectRoot "requirements.txt")

# Register the project interpreter as a notebook kernel. Keeping Jupyter's
# writable folders inside the project avoids user-folder permission errors.
$env:JUPYTER_CONFIG_DIR = Join-Path $projectRoot "tmp\jupyter_config"
$env:JUPYTER_DATA_DIR = Join-Path $projectRoot "tmp\jupyter_data"
$env:JUPYTER_RUNTIME_DIR = Join-Path $projectRoot "tmp\jupyter_runtime"
$env:IPYTHONDIR = Join-Path $projectRoot "tmp\ipython"
& $environmentPython -m ipykernel install --sys-prefix `
    --name "bangla-code-mixed" `
    --display-name "Python 3.12 (Bangla Code-Mixed)"

$installedVersion = & $environmentPython --version
Write-Host ""
Write-Host "SETUP COMPLETE" -ForegroundColor Green
Write-Host "Environment: $environmentPath"
Write-Host "Python:      $installedVersion"
Write-Host "Notebook:    Python 3.12 (Bangla Code-Mixed)"
Write-Host ""
Write-Host "Start the web application with:"
Write-Host "powershell -ExecutionPolicy Bypass -File .\run.ps1" -ForegroundColor Yellow

# Dependency archives can be very large and are no longer needed after a
# successful installation. Removing this project-local cache keeps the folder
# clean; the installed environment remains available.
if (Test-Path -LiteralPath $downloadCachePath) {
    Remove-Item -Recurse -Force -LiteralPath $downloadCachePath
}
