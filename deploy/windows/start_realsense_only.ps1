param(
    [string]$RosSetup = "C:\\opt\\ros\\jazzy\\x64\\local_setup.ps1",
    [string]$WorkspaceSetup = "",
    [int]$DomainId = 0,
    [string]$RealsenseSerial = "",
    [bool]$WarmupRosGraph = $true,
    [switch]$Foreground = $false,
    [switch]$Execute = $false
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

. (Join-Path $scriptDir "start_hw_nodes.ps1") `
    -RosSetup $RosSetup `
    -WorkspaceSetup $WorkspaceSetup `
    -DomainId $DomainId `
    -RealsenseSerial $RealsenseSerial `
    -WarmupRosGraph $WarmupRosGraph `
    -StartArm:$false `
    -StartRealsense:$true `
    -Foreground:$Foreground `
    -Execute:$Execute
