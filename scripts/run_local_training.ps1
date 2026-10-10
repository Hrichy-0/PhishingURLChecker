[CmdletBinding()]
param(
    [ValidateRange(1, 2000)][int]$Trees = 800,
    [ValidateRange(1, 16)][int]$Threads = 4,
    [string]$ExperimentName = 'phreshphish_mixed_20261009'
)

$ErrorActionPreference = 'Stop'
$projectPath = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$pythonPath = Join-Path $projectPath '.venv/Scripts/python.exe'
$experimentPath = Join-Path $projectPath "artifacts/experiments/$ExperimentName"
$splitsPath = Join-Path $projectPath 'data/processed/phreshphish_mixed_v1'
$trainingLog = Join-Path $experimentPath 'training.log'

if ($ExperimentName -notmatch '^[a-zA-Z0-9_-]+$') {
    throw 'ExperimentName must contain only letters, digits, underscores or hyphens.'
}
if (Test-Path -LiteralPath (Join-Path $experimentPath 'model_metadata.json')) {
    throw 'This experiment already has a saved model. Choose a new ExperimentName.'
}
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'The project virtual environment is missing.'
}
New-Item -ItemType Directory -Path $experimentPath -Force | Out-Null

Push-Location $projectPath
try {
    Write-Host "Local training directory: $projectPath"
    Write-Host "Interpreter: $pythonPath"
    Write-Host "Device: local CPU; $Threads threads; up to $Trees trees per candidate"
    Write-Host "Results and log: $experimentPath"
    Write-Host 'Uses the native XGBoost API. No system security settings are changed.'

    $trainingArguments = @(
        '-u', 'scripts/train_native_model.py',
        '--input', 'data/processed/sources/phiusiil.csv',
        'data/processed/sources/balanced.csv',
        'data/processed/sources/phreshphish_train.csv',
        '--data-dir', $splitsPath,
        '--artifacts-dir', $experimentPath,
        '--external-test', 'data/external/huggingface/phreshphish/test.csv',
        '--trees', "$Trees", '--threads', "$Threads"
    )
    & $pythonPath @trainingArguments 2>&1 | Tee-Object -FilePath $trainingLog
    if ($LASTEXITCODE -ne 0) {
        throw "Training stopped with exit code $LASTEXITCODE. See $trainingLog"
    }
    & $pythonPath 'scripts/inspect_model_metadata.py' '--metadata' (Join-Path $experimentPath 'model_metadata.json') '--output' (Join-Path $experimentPath 'feature-importance.md')
    if ($LASTEXITCODE -ne 0) {
        throw 'The feature-importance report could not be generated.'
    }
}
finally {
    Pop-Location
}
