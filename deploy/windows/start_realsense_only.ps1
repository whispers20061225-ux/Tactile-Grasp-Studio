param(
    [string]$RosSetup = "C:\\opt\\ros\\jazzy\\x64\\local_setup.ps1",
    [string]$WorkspaceSetup = "",
    [int]$DomainId = 0,
    [string]$RealsenseSerial = "",
    [int]$ColorWidth = 640,
    [int]$ColorHeight = 480,
    [int]$ColorFps = 30,
    [int]$DepthWidth = 640,
    [int]$DepthHeight = 480,
    [int]$DepthFps = 30,
    [bool]$AlignDepth = $true,
    [bool]$WarmupRosGraph = $true,
    [switch]$Foreground = $false,
    [switch]$UseRealsenseWatchdog = $true,
    [switch]$Execute = $false
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

. (Join-Path $scriptDir "start_hw_nodes.ps1") `
    -RosSetup $RosSetup `
    -WorkspaceSetup $WorkspaceSetup `
    -DomainId $DomainId `
    -RealsenseSerial $RealsenseSerial `
    -ColorWidth $ColorWidth `
    -ColorHeight $ColorHeight `
    -ColorFps $ColorFps `
    -DepthWidth $DepthWidth `
    -DepthHeight $DepthHeight `
    -DepthFps $DepthFps `
    -AlignDepth $AlignDepth `
    -WarmupRosGraph $WarmupRosGraph `
    -StartArm:$false `
    -StartRealsense:$true `
    -Foreground:$Foreground `
    -UseRealsenseWatchdog:$UseRealsenseWatchdog `
    -Execute:$Execute
