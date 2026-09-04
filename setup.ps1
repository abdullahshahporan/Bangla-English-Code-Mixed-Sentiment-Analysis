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

if (-not (Get-Command "uv" -ErrorAction SilentlyContinue)) {
    Write-Host "The 'uv' command is required for automatic setup." -ForegroundColor Red
    Write-Host "Install it from https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
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
    uv venv --python 3.12 $environmentPath
}

Write-Host "Installing project dependencies..." -ForegroundColor Cyan
uv pip install --python $environmentPython -r (Join-Path $projectRoot "requirements.txt")

$installedVersion = & $environmentPython --version
Write-Host ""
Write-Host "SETUP COMPLETE" -ForegroundColor Green
Write-Host "Environment: $environmentPath"
Write-Host "Python:      $installedVersion"
Write-Host ""
Write-Host "Start the web application with:"
Write-Host ".\.venv\Scripts\python.exe -m streamlit run app.py" -ForegroundColor Yellow

# Dependency archives can be very large and are no longer needed after a
# successful installation. Removing this project-local cache keeps the folder
# clean; the installed environment remains available.
if (Test-Path -LiteralPath $downloadCachePath) {
    Remove-Item -Recurse -Force -LiteralPath $downloadCachePath
}
