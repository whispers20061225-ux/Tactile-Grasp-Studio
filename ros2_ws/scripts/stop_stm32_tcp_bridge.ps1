param()

$ErrorActionPreference = "SilentlyContinue"

$runtimeDir = Join-Path $env:LOCALAPPDATA "ProgrammeWebUI"
$pidFile = Join-Path $runtimeDir "stm32-bridge.win.pid"

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
