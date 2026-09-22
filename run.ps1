<#
Run the Bangla-English sentiment web application with the correct environment.

From the project folder:
    powershell -ExecutionPolicy Bypass -File .\run.ps1
#>

param(
    [int]$Port = 8502
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$environmentPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$requirementsPath = Join-Path $projectRoot "requirements.txt"

Set-Location -LiteralPath $projectRoot
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:MPLBACKEND = "Agg"
$env:MPLCONFIGDIR = Join-Path $projectRoot ".matplotlib"
$env:JUPYTER_CONFIG_DIR = Join-Path $projectRoot "tmp\jupyter_config"
$env:JUPYTER_DATA_DIR = Join-Path $projectRoot "tmp\jupyter_data"
$env:JUPYTER_RUNTIME_DIR = Join-Path $projectRoot "tmp\jupyter_runtime"
$env:IPYTHONDIR = Join-Path $projectRoot "tmp\ipython"
$env:UV_PYTHON_INSTALL_DIR = Join-Path $projectRoot ".python"
$env:UV_CACHE_DIR = Join-Path $projectRoot ".uv-cache"

if (-not (Test-Path -LiteralPath $environmentPython)) {
    Write-Host "Project environment is missing. Running setup..." -ForegroundColor Yellow
    & powershell -ExecutionPolicy Bypass -File (Join-Path $projectRoot "setup.ps1")
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

# Checking module locations is quick and catches the common wrong/missing
# environment problem before Streamlit starts.
& $environmentPython (Join-Path $projectRoot "check_environment.py")
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing missing project dependencies..." -ForegroundColor Yellow
    $uvCommand = Get-Command "uv" -ErrorAction SilentlyContinue
    if ($uvCommand) {
        $uvExecutable = $uvCommand.Source
    } else {
        $uvExecutable = Join-Path $env:USERPROFILE ".local\bin\uv.exe"
    }
    if (-not (Test-Path -LiteralPath $uvExecutable)) {
        Write-Host "uv was not found. Run setup.ps1 after installing uv." -ForegroundColor Red
        exit 1
    }
    & $uvExecutable pip install --python $environmentPython -r $requirementsPath
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

$requiredModels = @(
    "tfidf_logistic.pkl",
    "word2vec_sentiment.pkl",
    "bilstm_best.pt",
    "transformer_best.pt"
)
$missingModels = @(
    $requiredModels | Where-Object {
        -not (Test-Path -LiteralPath (Join-Path $projectRoot "models\$_"))
    }
)
if ($missingModels.Count -gt 0) {
    Write-Host "Missing trained model files: $($missingModels -join ', ')" -ForegroundColor Red
    Write-Host "Run notebooks 01 through 05 once to create them."
    exit 1
}

$healthUrl = "http://127.0.0.1:$Port/_stcore/health"
try {
    $healthResponse = Invoke-WebRequest -Uri $healthUrl -TimeoutSec 2
    if ($healthResponse.StatusCode -eq 200) {
        Write-Host "The application is already running." -ForegroundColor Green
        Write-Host "Open: http://localhost:$Port" -ForegroundColor Cyan
        exit 0
    }
} catch {
    # No application is listening on this port, so start it below.
}

Write-Host "Environment: Python 3.12 project .venv" -ForegroundColor Green
Write-Host "All dependencies and model files are ready." -ForegroundColor Green
Write-Host "Open: http://localhost:$Port" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the application."

& $environmentPython -m streamlit run app.py `
    --server.port $Port `
    --server.address 127.0.0.1 `
    --browser.gatherUsageStats false
