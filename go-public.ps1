<#
.SYNOPSIS
  Start Beautify and expose it on a public HTTPS URL via a Cloudflare quick tunnel.

.DESCRIPTION
  No Cloudflare account, no card, no configuration. The tunnel runs on this machine and
  forwards to the app on localhost, so the app itself never has to listen on a public
  interface — only the tunnel can reach it.

  The URL is regenerated every run. For a URL that never changes you need a free Cloudflare
  account and a domain (`cloudflared tunnel create`), which is a different, longer setup.

  Close this window to take the site offline.

.EXAMPLE
  ./go-public.ps1
  ./go-public.ps1 -Port 8000
#>
[CmdletBinding()]
param(
  [string]$VenvPython = 'C:\Users\Fresco-Arjun\aienv\Scripts\python.exe',
  [int]$Port = 8000
)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$Cloudflared = Join-Path $Root 'tools\cloudflared.exe'

if (-not (Test-Path $Cloudflared)) {
  Write-Host 'Fetching cloudflared...' -ForegroundColor Cyan
  New-Item -ItemType Directory -Force -Path (Split-Path $Cloudflared) | Out-Null
  Invoke-WebRequest -UseBasicParsing -OutFile $Cloudflared `
    -Uri 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe'
}

# --- 1. app -------------------------------------------------------------------------------
$up = $false
try { $up = (Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 3).ready } catch {}

if (-not $up) {
  Write-Host "Starting Beautify on port $Port ..." -ForegroundColor Cyan
  Start-Process -FilePath $VenvPython `
    -ArgumentList "-m uvicorn app.main:app --host 127.0.0.1 --port $Port" `
    -WorkingDirectory $Root -WindowStyle Minimized
  for ($i = 0; $i -lt 45; $i++) {
    Start-Sleep -Seconds 2
    try { if ((Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 3).ready) { $up = $true; break } } catch {}
  }
}

if (-not $up) {
  Write-Error "The app did not become ready on port $Port. Start it with ./run.ps1 and read its output."
  exit 1
}
Write-Host 'Engine ready (models loaded).' -ForegroundColor Green

# --- 2. tunnel ----------------------------------------------------------------------------
$log = Join-Path $env:TEMP 'beautify-tunnel.log'
Remove-Item $log -ErrorAction SilentlyContinue
Write-Host 'Opening the public tunnel...' -ForegroundColor Cyan

$tunnel = Start-Process -FilePath $Cloudflared `
  -ArgumentList "tunnel --url http://localhost:$Port --no-autoupdate" `
  -WindowStyle Hidden -PassThru -RedirectStandardError $log -RedirectStandardOutput "$log.out"

$url = $null
for ($i = 0; $i -lt 40; $i++) {
  Start-Sleep -Seconds 2
  $text = Get-Content $log -Raw -ErrorAction SilentlyContinue
  if ($text -match 'https://[a-z0-9-]+\.trycloudflare\.com') { $url = $Matches[0]; break }
}

if (-not $url) {
  Write-Warning 'No tunnel URL appeared. Last lines:'
  Get-Content $log -Tail 20
  exit 1
}

Write-Host ''
Write-Host '  Your site is live at:' -ForegroundColor Green
Write-Host "  $url" -ForegroundColor White
Write-Host ''
Write-Host '  Anyone with this link can upload a photo and use your CPU.' -ForegroundColor DarkYellow
Write-Host '  The link dies when you close this window.' -ForegroundColor DarkGray
Write-Host ''
Write-Host 'Press Ctrl+C to take it offline.'

try { Wait-Process -Id $tunnel.Id } finally {
  Stop-Process -Id $tunnel.Id -Force -ErrorAction SilentlyContinue
  Write-Host 'Tunnel closed - the site is no longer reachable.' -ForegroundColor Cyan
}
