$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'set_model_cache.ps1')
$python = Join-Path $root 'services/ai-api/.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) {
  throw 'Create the canonical services/ai-api/.venv environment before verification.'
}

function Invoke-Checked([string]$label, [scriptblock]$command) {
  & $command
  if ($LASTEXITCODE -ne 0) { throw "$label failed with exit code $LASTEXITCODE." }
}

Push-Location (Join-Path $root 'apps/web')
try {
  Invoke-Checked 'Frontend lint' { & npm.cmd run lint }
  Invoke-Checked 'Frontend typecheck' { & npm.cmd run typecheck }
  Invoke-Checked 'Frontend tests' { & npm.cmd test -- --run }
  Invoke-Checked 'Frontend build' { & npm.cmd run build }
} finally { Pop-Location }

Push-Location (Join-Path $root 'services/ai-api')
try {
  Invoke-Checked 'Backend lint' { & $python -m ruff check . }
  Invoke-Checked 'Backend tests' { & $python -m pytest -q }
  Invoke-Checked 'Backend import' { & $python -c 'from app.main import app; print(app.title)' }
  Invoke-Checked 'Dependency integrity' { & $python -m pip check }
} finally { Pop-Location }
