param(
    [string]$WorkspaceSetup = "",
    [string]$ArmParamFile = "",
    [string]$RealsenseSerial = "_333422301846",
    [switch]$StartArm = $true,
    [switch]$StartRealsense = $true,
    [switch]$Execute = $false
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

. (Join-Path $scriptDir "env_ros2_windows.ps1") -WorkspaceSetup $WorkspaceSetup

$realsenseCmd = @(
    "ros2 run realsense2_camera realsense2_camera_node --ros-args",
    "-p serial_no:=$RealsenseSerial",
    "-p enable_color:=true",
    "-p enable_depth:=true",
    "-p align_depth.enable:=true",
    "-p rgb_camera.profile:=640x480x15",
    "-p depth_module.profile:=640x480x15"
) -join " "

$armCmd = ""
if ($ArmParamFile) {
    $armCmd = "ros2 run tactile_hardware arm_driver_node --ros-args --params-file `"$ArmParamFile`""
} else {
    $armCmd = "ros2 run tactile_hardware arm_driver_node  # add --ros-args --params-file <path>"
}

Write-Host ""
Write-Host "Windows hardware node commands:"
if ($StartRealsense) {
    Write-Host "  [RealSense] $realsenseCmd"
}
if ($StartArm) {
    Write-Host "  [Arm]       $armCmd"
}
Write-Host ""

if (-not $Execute) {
    Write-Host "Dry-run mode. Re-run with -Execute to spawn new PowerShell windows."
    exit 0
}

if ($StartRealsense) {
    Start-Process pwsh -ArgumentList "-NoExit", "-Command", "& { . `"$scriptDir\\env_ros2_windows.ps1`"; $realsenseCmd }"
}

if ($StartArm) {
    $armLaunch = "& { . `"$scriptDir\\env_ros2_windows.ps1`""
    if ($WorkspaceSetup) {
        $armLaunch += " -WorkspaceSetup `"$WorkspaceSetup`""
    }
    $armLaunch += "; $armCmd }"
    Start-Process pwsh -ArgumentList "-NoExit", "-Command", $armLaunch
}
