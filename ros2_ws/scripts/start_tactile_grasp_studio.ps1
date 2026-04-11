param(
    [string]$Distro = "Ubuntu-24.04",
    [switch]$NoBrowser,
    [switch]$Watchdog,
    [switch]$EnableWatchdog
)

$ErrorActionPreference = "Stop"

$runtimeDir = Join-Path $env:LOCALAPPDATA "ProgrammeWebUI"
$watchdogPidFile = Join-Path $runtimeDir "watchdog.win.pid"
$watchdogLog = Join-Path $runtimeDir "watchdog.log"
$debugHelperPidFile = Join-Path $runtimeDir "debug-helper.win.pid"
$linuxStartScript = "/home/whispers/programme/ros2_ws/scripts/start_tactile_grasp_studio.sh"
$stopScript = Join-Path $PSScriptRoot "stop_tactile_grasp_studio.ps1"
$bridgeStartScript = Join-Path $PSScriptRoot "start_stm32_tcp_bridge.ps1"
$debugHelperScript = Join-Path $PSScriptRoot "tactile_grasp_studio_helper.ps1"
$debugHelperLog = Join-Path $runtimeDir "debug-helper.log"
$frontendDevUrl = "http://127.0.0.1:5173/control"
$frontendGatewayUrl = "http://127.0.0.1:8765/control"
$gatewayHealthUrl = "http://127.0.0.1:8765/api/bootstrap"
$watchdogRestartThreshold = 3

function Test-HttpReady {
    param([Parameter(Mandatory = $true)][string]$Url)
    try {
        $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 3
        return ($response.StatusCode -ge 200 -and $response.StatusCode -lt 300)
    } catch {
        return $false
    }
}

function Wait-HttpReady {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][int]$TimeoutSec
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        if (Test-HttpReady -Url $Url) {
            Write-Host "[start] $Label ready: $Url"
            return
        }
        Start-Sleep -Seconds 1
    }

    throw "[start] $Label did not become ready: $Url"
}

function Get-ReadyHttpUrl {
    param([Parameter(Mandatory = $true)][string[]]$Urls)
    foreach ($url in $Urls) {
        if (Test-HttpReady -Url $url) {
            return $url
        }
    }
    return ""
}

function Wait-AnyHttpReady {
    param(
        [Parameter(Mandatory = $true)][string[]]$Urls,
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][int]$TimeoutSec
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        $readyUrl = Get-ReadyHttpUrl -Urls $Urls
        if ($readyUrl) {
            Write-Host "[start] $Label ready: $readyUrl"
            return $readyUrl
        }
        Start-Sleep -Seconds 1
    }

    throw "[start] $Label did not become ready: $($Urls -join ', ')"
}

function Ensure-TactileBridgeReady {
    param([switch]$Quiet)

    if (-not (Test-Path $bridgeStartScript)) {
        if (-not $Quiet) {
            Write-Warning "[start] tactile bridge helper not found: $bridgeStartScript"
        }
        return $false
    }

    try {
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $bridgeStartScript | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "[start] tactile bridge helper exited with code $LASTEXITCODE"
        }
        return $true
    } catch {
        if (-not $Quiet) {
            Write-Warning ("[start] tactile bridge helper failed: " + $_.Exception.Message)
        }
        return $false
    }
}

function Invoke-WslStart {
    $arguments = @("-d", $Distro, "--", $linuxStartScript, "--no-browser")
    & wsl.exe @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "[watchdog] start_tactile_grasp_studio.sh exited with code $LASTEXITCODE"
    }
}

if ($Watchdog) {
    New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null
    Add-Content -Path $watchdogLog -Value ("[watchdog] started at {0}" -f (Get-Date -Format o))
    $consecutiveFailures = 0
    while ($true) {
        $bridgeOk = Ensure-TactileBridgeReady -Quiet
        $frontendUrl = Get-ReadyHttpUrl -Urls @($frontendDevUrl, $frontendGatewayUrl)
        $frontendOk = -not [string]::IsNullOrWhiteSpace($frontendUrl)
        $gatewayOk = Test-HttpReady -Url $gatewayHealthUrl
        if ($frontendOk -and $gatewayOk) {
            $consecutiveFailures = 0
        } else {
            $consecutiveFailures += 1
            Add-Content -Path $watchdogLog -Value (
                "[watchdog] healthcheck failed at {0}: tactile_bridge={1} frontend={2} gateway={3} ready_url={4} consecutive={5}" -f
                (Get-Date -Format o), $bridgeOk, $frontendOk, $gatewayOk, $frontendUrl, $consecutiveFailures
            )
            if ($consecutiveFailures -ge $watchdogRestartThreshold) {
                Add-Content -Path $watchdogLog -Value ("[watchdog] restarting stack at {0}" -f (Get-Date -Format o))
                try {
                    Ensure-TactileBridgeReady -Quiet | Out-Null
                    Invoke-WslStart
                    Wait-HttpReady -Url $gatewayHealthUrl -Label "gateway" -TimeoutSec 60
                    Wait-AnyHttpReady -Urls @($frontendDevUrl, $frontendGatewayUrl) -Label "frontend" -TimeoutSec 30 | Out-Null
                    $consecutiveFailures = 0
                } catch {
                    Add-Content -Path $watchdogLog -Value (
                        "[watchdog] restart failed at {0}: {1}" -f (Get-Date -Format o), $_.Exception.Message
                    )
                }
            }
        }
        Start-Sleep -Seconds 5
    }
}

New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $stopScript -Distro $Distro | Out-Null

Ensure-TactileBridgeReady -Quiet | Out-Null
Invoke-WslStart

Wait-HttpReady -Url $gatewayHealthUrl -Label "gateway" -TimeoutSec 30
$controlUrl = Wait-AnyHttpReady -Urls @($frontendDevUrl, $frontendGatewayUrl) -Label "frontend" -TimeoutSec 20
$frontendMode = if ($controlUrl -eq $frontendDevUrl) { "vite-dev" } else { "gateway-static" }

$debugHelperProcess = Start-Process -FilePath "powershell.exe" `
    -ArgumentList @(
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        "& `"$debugHelperScript`" -Distro `"$Distro`""
    ) `
    -WindowStyle Hidden `
    -PassThru
Set-Content -Path $debugHelperPidFile -Value ([string]$debugHelperProcess.Id) -Encoding ascii

if ($EnableWatchdog) {
    $watchdogProcess = Start-Process -FilePath "powershell.exe" `
        -ArgumentList @(
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            $PSCommandPath,
            "-Distro",
            $Distro,
            "-Watchdog"
        ) `
        -WindowStyle Hidden `
        -PassThru
    Set-Content -Path $watchdogPidFile -Value ([string]$watchdogProcess.Id) -Encoding ascii
}

if (-not $NoBrowser) {
    Start-Process $controlUrl | Out-Null
}

Write-Host "[start] Tactile Grasp Studio is ready"
Write-Host "[start] control page: $controlUrl"
Write-Host "[start] frontend mode: $frontendMode"
Write-Host "[start] gateway root: http://127.0.0.1:8765/"
if ($EnableWatchdog) {
    Write-Host "[start] watchdog pid file: $watchdogPidFile"
    Write-Host "[start] watchdog log: $watchdogLog"
} else {
    Write-Host "[start] watchdog: disabled"
}
Write-Host "[start] debug helper pid file: $debugHelperPidFile"
Write-Host "[start] debug helper log: $debugHelperLog"
