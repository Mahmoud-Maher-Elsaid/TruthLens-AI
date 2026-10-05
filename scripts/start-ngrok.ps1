$ErrorActionPreference = 'Stop'
if (-not (Get-Command ngrok -ErrorAction SilentlyContinue)) { throw 'Install ngrok from https://ngrok.com/download first.' }
& ngrok http 8000
