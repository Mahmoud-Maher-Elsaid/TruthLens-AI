$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$cacheRoot = Join-Path $root '.cache\huggingface'
$hubCache = Join-Path $cacheRoot 'hub'
$sentenceTransformersCache = Join-Path $cacheRoot 'sentence-transformers'

foreach ($path in @($cacheRoot, $hubCache, $sentenceTransformersCache)) {
  New-Item -ItemType Directory -Path $path -Force | Out-Null
}

$env:TRUTHLENS_MODEL_CACHE = $cacheRoot
$env:HF_HOME = $cacheRoot
$env:HF_HUB_CACHE = $hubCache
$env:HUGGINGFACE_HUB_CACHE = $hubCache
Remove-Item Env:TRANSFORMERS_CACHE -ErrorAction SilentlyContinue
$env:SENTENCE_TRANSFORMERS_HOME = $sentenceTransformersCache

Write-Output "TruthLens model cache: $cacheRoot"
