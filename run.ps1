<#
.SYNOPSIS
  Start Beautify (API + web UI + AI inference — one process).

.DESCRIPTION
  Pins the Python interpreter and the working directory, then refuses to start if the
  interpreter cannot import torch. That check exists because the failure it prevents is silent:
  started with a system Python, the service falls back to a plain resize and still reports
  success — nothing errors, enhancement just quietly stops happening.

.EXAMPLE
  ./run.ps1
  ./run.ps1 -VenvPython 'C:\aienv\Scripts\python.exe' -Port 8080
#>
[CmdletBinding()]
param(
  [string]$VenvPython = 'C:\Users\Fresco-Arjun\aienv\Scripts\python.exe',
  [int]$Port = 8000,
  [string]$BindHost = '127.0.0.1',
  [switch]$Reload
)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot

if (-not (Test-Path $VenvPython)) {
  Write-Error @"
Python interpreter not found: $VenvPython

Create a venv at a SHORT path (deep paths blow past the Windows 260-char limit and break the
torch install), then install the requirements:

  python -m venv C:\aienv
  C:\aienv\Scripts\python.exe -m pip install --upgrade pip
  C:\aienv\Scripts\python.exe -m pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
  C:\aienv\Scripts\python.exe -m pip install -r "$Root\requirements.txt"

Then:  ./run.ps1 -VenvPython 'C:\aienv\Scripts\python.exe'
"@
  exit 1
}

Write-Host 'Checking the interpreter has torch...' -NoNewline
$torch = & $VenvPython -c "import torch; print(torch.__version__)" 2>&1
if ($LASTEXITCODE -ne 0) {
  Write-Host ' FAILED' -ForegroundColor Red
  Write-Error "This interpreter cannot import torch, so enhancement would silently degrade to a plain resize.`n  Interpreter : $VenvPython`n  Error       : $torch"
  exit 1
}
Write-Host " ok (torch $torch)" -ForegroundColor Green

# Free the port if a stale copy is squatting on it.
$existing = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
foreach ($conn in $existing) {
  Write-Warning "Port $Port is held by PID $($conn.OwningProcess) - stopping it."
  try { Stop-Process -Id $conn.OwningProcess -Force -ErrorAction Stop } catch { Write-Warning "  could not stop it: $_" }
}
if ($existing) { Start-Sleep -Seconds 2 }

# The working directory MUST be the project root: GFPGAN resolves its auxiliary face-detection
# and face-parsing weights relative to ./gfpgan/weights.
Push-Location $Root
try {
  $argv = @('-m', 'uvicorn', 'app.main:app', '--host', $BindHost, '--port', "$Port")
  if ($Reload) { $argv += '--reload' }
  Write-Host ''
  Write-Host "Starting Beautify on http://${BindHost}:${Port}" -ForegroundColor Cyan
  Write-Host 'First start loads ~420 MB of model weights; give it a moment.' -ForegroundColor DarkGray
  Write-Host ''
  & $VenvPython @argv
}
finally {
  Pop-Location
}
