# Windows + VM One-Click Runbook

This runbook provides two one-click scripts for daily bringup:

1. Windows one-click hardware start + health check (RealSense)
2. VM one-click end-to-end debug startup (guard + launch + UI)

## 1) Windows: One-Click RealSense Ready

Script:

- `deploy/windows/start_realsense_ready.ps1`

What it does:

- loads ROS2 Windows environment
- starts RealSense node (`tactile_vision` fallback or `realsense2_camera` if present)
- waits for required topics
- samples color/depth fps and checks thresholds
- prints final `[READY]` status

Command (PowerShell):

```powershell
$PROJECT_ROOT = "C:\Users\whisp\Desktop\dayi\programme"
cd $PROJECT_ROOT
. .\deploy\windows\start_realsense_ready.ps1 `
  -RosSetup "C:\pixi_ws\ros2-windows\ros2-windows\local_setup.bat" `
  -WorkspaceSetup ".\ros2_ws\install\local_setup.ps1" `
  -DomainId 0 `
  -ColorWidth 640 `
  -ColorHeight 480 `
  -ColorFps 30 `
  -DepthWidth 640 `
  -DepthHeight 480 `
  -DepthFps 30 `
  -TopicTimeoutSec 20 `
  -HzSampleSec 10 `
  -MinColorHz 3.0 `
  -MinDepthHz 3.0
```

Success example:

- `[OK] topics discovered`
- `[OK] RealSense READY: color=...Hz depth=...Hz`
- `[READY] You can now start VM one-click debug script.`

## 2) VM: One-Click Debug (Guard -> Launch -> UI)

Script:

- `deploy/vm/start_ui_with_realsense_guard.sh`

What it does:

- validates Windows->VM RealSense link using `deploy/vm/test_realsense_stream.sh`
- starts `split_vm_app.launch.py` in background
- validates VM-side core nodes (`arm_control_node`, `demo_task_node`, `tactile_ui_subscriber`); relay/monitor nodes are optional by vision profile
- validates tactile simulation stream when enabled (`/tactile/raw`)
- validates arm chain when enabled (`/arm/state`, `/arm/*`, `/control/arm/*`)
- starts `main_ros2.py --control-mode ros2`
- when UI exits, it stops launch process by default

Command (VM terminal):

```bash
cd /home/zhuyiwei/programme/programme
bash deploy/vm/start_ui_with_realsense_guard.sh 0 20 12 3.0 3.0 true dayiprogramme312 20 3.0 true minimal
```

Parameters:

- `0`: ROS domain ID
- `20`: topic discovery timeout seconds
- `12`: hz sampling window seconds
- `3.0`: minimum color fps
- `3.0`: minimum depth fps
- `true`: `start_tactile_sensor` argument for VM launch
- `dayiprogramme312`: conda environment name for UI
- `20`: arm chain wait timeout seconds
- `3.0`: minimum tactile fps
- `true`: whether arm chain is required (`false` to skip arm guard)

Optional:

- keep launch alive after UI exits:

```bash
KEEP_LAUNCH=true bash deploy/vm/start_ui_with_realsense_guard.sh 0 20 12 3.0 3.0 true dayiprogramme312 20 3.0 true minimal
```

Skip arm guard (camera + tactile only):

```bash
bash deploy/vm/start_ui_with_realsense_guard.sh 0 20 12 3.0 3.0 true dayiprogramme312 20 3.0 false minimal
```

## Troubleshooting

1. VM reports CycloneDDS RMW missing:
   - `sudo apt update && sudo apt install -y ros-jazzy-rmw-cyclonedds-cpp`
2. Windows script reports missing `pyrealsense2`:
   - `python -m pip install pyrealsense2`
3. Windows build reports missing `pkg_resources`:
   - `python -m pip install "setuptools<81"`
4. VM one-click script exits after guard failure and launch log shows `does not match an available interface`:
   - the VM host-only IP changed and CycloneDDS picked a stale address
   - rerun with explicit overrides if auto-detect picked the wrong NIC:
     - `WINDOWS_HOST_ONLY_IP=<windows_ip> VM_HOST_ONLY_IP=<vm_ip> bash deploy/vm/start_ui_with_realsense_guard.sh ...`
   - the env script now writes a runtime DDS file and should no longer require editing `config/dds/cyclonedds_vm.xml`
