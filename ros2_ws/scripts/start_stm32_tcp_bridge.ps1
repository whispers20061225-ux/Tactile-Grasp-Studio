param(
    [string]$PortName = "",
    [string]$ListenHost = "0.0.0.0",
    [int]$ListenPort = 19024,
    [string]$ReadyCheckHost = "127.0.0.1",
    [int]$StartupTimeoutSec = 8
)

$ErrorActionPreference = "Stop"

$runtimeDir = Join-Path $env:LOCALAPPDATA "ProgrammeWebUI"
$pidFile = Join-Path $runtimeDir "stm32-bridge.win.pid"
$logFile = Join-Path $runtimeDir "stm32-bridge.log"
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$bridgeScript = Join-Path $repoRoot "scripts\\stm32_com_tcp_bridge.ps1"

if (-not $PortName) {
    $PortName = if ($env:PROGRAMME_TACTILE_COM_PORT) {
        [string]$env:PROGRAMME_TACTILE_COM_PORT
    } else {
        "COM4"
    }
}

function Test-TcpListener {
    param(
        [Parameter(Mandatory = $true)][string]$TargetHost,
        [Parameter(Mandatory = $true)][int]$Port
    )

    $client = $null
    try {
        $client = [System.Net.Sockets.TcpClient]::new()
        $asyncResult = $client.BeginConnect($TargetHost, $Port, $null, $null)
        if (-not $asyncResult.AsyncWaitHandle.WaitOne(400)) {
            return $false
        }
        $client.EndConnect($asyncResult)
        return $true
    } catch {
        return $false
    } finally {
        if ($null -ne $client) {
            $client.Dispose()
        }
    }
}

function Stop-ManagedBridge {
    if (Test-Path $pidFile) {
        $rawPid = (Get-Content -Path $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1).Trim()
        $pidValue = 0
        if ([int]::TryParse($rawPid, [ref]$pidValue)) {
            Stop-Process -Id $pidValue -Force -ErrorAction SilentlyContinue
        }
        Remove-Item -Path $pidFile -Force -ErrorAction SilentlyContinue
    }

    Get-CimInstance Win32_Process |
        Where-Object {
            $_.Name -eq "powershell.exe" -and
            $_.CommandLine -match "stm32_com_tcp_bridge\.ps1"
        } |
        ForEach-Object {
            Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
        }
}

if (-not (Test-Path $bridgeScript)) {
    throw "[bridge] helper target not found: $bridgeScript"
}

New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null

if (Test-TcpListener -TargetHost $ReadyCheckHost -Port $ListenPort) {
    Write-Host "[bridge] STM32 TCP bridge already listening on $ListenHost`:$ListenPort"
    exit 0
}

Stop-ManagedBridge

$quotedBridgeScript = $bridgeScript.Replace("'", "''")
$quotedPortName = $PortName.Replace("'", "''")
$quotedListenHost = $ListenHost.Replace("'", "''")
$quotedLogFile = $logFile.Replace("'", "''")
$bridgeCommand = "& '$quotedBridgeScript' -PortName '$quotedPortName' -ListenHost '$quotedListenHost' -ListenPort $ListenPort *>> '$quotedLogFile'"

$bridgeProcess = Start-Process -FilePath "powershell.exe" `
    -ArgumentList @(
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        $bridgeCommand
    ) `
    -WindowStyle Hidden `
    -PassThru

Set-Content -Path $pidFile -Value ([string]$bridgeProcess.Id) -Encoding ascii

$deadline = (Get-Date).AddSeconds([Math]::Max(1, $StartupTimeoutSec))
while ((Get-Date) -lt $deadline) {
    if (Test-TcpListener -TargetHost $ReadyCheckHost -Port $ListenPort) {
        Write-Host "[bridge] STM32 TCP bridge ready on $ListenHost`:$ListenPort via $PortName"
        exit 0
    }
    if ($bridgeProcess.HasExited) {
        break
    }
    Start-Sleep -Milliseconds 400
}

$logTail = ""
if (Test-Path $logFile) {
    $logTail = (Get-Content -Path $logFile -Tail 20 -ErrorAction SilentlyContinue | Out-String).Trim()
}

throw "[bridge] STM32 TCP bridge failed to become ready on $ListenHost`:$ListenPort via $PortName. $logTail"
