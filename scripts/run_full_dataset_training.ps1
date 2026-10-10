$ErrorActionPreference = 'Stop'
$projectPath = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$pythonPath = Join-Path $projectPath '.venv/Scripts/python.exe'
$fullDataset = Join-Path $projectPath 'data/raw/phreshphish'
$downloadLog = Join-Path $fullDataset 'full-download.log'
New-Item -ItemType Directory -Path $fullDataset -Force | Out-Null
Push-Location $projectPath
try {
    Write-Host "Downloading every original dataset file to $fullDataset"
    & $pythonPath '-u' 'scripts/download_phreshphish_full.py' 2>&1 | Tee-Object -FilePath $downloadLog -Append
    if ($LASTEXITCODE -ne 0) { throw 'Full download stopped. Rerun this script to resume.' }
    & $pythonPath 'scripts/prepare_data.py' '--input' 'data/external/huggingface/phreshphish_full/train.csv' '--url-column' 'url' '--label-column' 'label' '--legitimate-label' 'benign' '--phishing-label' 'phish' '--output' 'data/processed/sources/phreshphish_train.csv'
    if ($LASTEXITCODE -ne 0) { throw 'Dataset cleaning failed.' }
    & $pythonPath '-u' 'scripts/train_native_model.py' '--input' 'data/processed/sources/phiusiil.csv' 'data/processed/sources/balanced.csv' 'data/processed/sources/phreshphish_train.csv' '--data-dir' 'data/processed/phreshphish_mixed_v1' '--artifacts-dir' 'artifacts/experiments/phreshphish_full_download_20261009' '--external-test' 'data/external/huggingface/phreshphish_full/test.csv' '--trees' '800' '--threads' '4' 2>&1 | Tee-Object -FilePath 'data/raw/phreshphish/training-after-download.log'
    if ($LASTEXITCODE -ne 0) { throw 'Local training stopped; see training-after-download.log.' }
    & $pythonPath 'scripts/inspect_model_metadata.py' '--metadata' 'artifacts/experiments/phreshphish_full_download_20261009/model_metadata.json' '--output' 'artifacts/experiments/phreshphish_full_download_20261009/feature-importance.md'
    if ($LASTEXITCODE -ne 0) { throw 'Feature importance report failed.' }
}
finally {
    Pop-Location
}
