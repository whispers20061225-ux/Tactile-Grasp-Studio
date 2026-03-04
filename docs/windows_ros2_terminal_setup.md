# Windows ROS2 Terminal Setup (Pixi + Conda Safe Mode)

This guide defines a stable way to run ROS2 CLI and project nodes on Windows,
especially when `conda(base)` is active in PowerShell.

## Scope

- OS: Windows
- ROS2 install path (example): `C:\pixi_ws\ros2-windows\ros2-windows`
- Project path (example): `C:\Users\whisp\Desktop\大一年度项目\programme`

## Why this is needed

Typical failure symptoms on Windows:

- `ros2` not found
- `PackageNotFoundError: ros2cli`
- `ImportError: DLL load failed while importing _rclpy_pybind11`

Root causes:

- Shell PATH does not include ROS2 setup result
- `conda(base)` modifies Python/DLL lookup
- CycloneDDS chain needs OpenSSL DLLs from pixi env

## Project script to initialize ROS2

Use project script:

- `deploy/windows/env_ros2_windows.ps1`

This script now does:

1. Clean `conda` runtime contamination (`CONDA_*`, `PYTHONHOME`, `PYTHONPATH`, and conda PATH entries)
2. Load ROS2 `local_setup.bat` / `local_setup.ps1`
3. Auto add pixi OpenSSL DLL path (e.g. `.pixi/envs/default/Library/bin`)
4. Set DDS runtime variables:
   - `ROS_DOMAIN_ID`
   - `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`
   - `ROS_LOCALHOST_ONLY=0`
   - `CYCLONEDDS_URI=file:///.../config/dds/cyclonedds_windows.xml`

## One-command startup per terminal

```powershell
. "C:\Users\whisp\Desktop\大一年度项目\programme\deploy\windows\env_ros2_windows.ps1" `
  -RosSetup "C:\pixi_ws\ros2-windows\ros2-windows\local_setup.bat" `
  -DomainId 0
```

Verify:

```powershell
ros2 --help
```

## Recommended daily workflow

Use separate terminals:

- ROS2 terminal: run `env_ros2_windows.ps1`, then `ros2 ...`
- Conda/AI terminal: keep conda for model/tooling tasks

Do not mix both in one long-running terminal if avoidable.

## Optional: add helper command in PowerShell profile

Add function/alias in profile:

```powershell
function Enter-Ros2Env {
  param(
    [int]$DomainId = 0,
    [string]$RosSetup = "C:\pixi_ws\ros2-windows\ros2-windows\local_setup.bat",
    [string]$EnvScript = "C:\Users\whisp\Desktop\大一年度项目\programme\deploy\windows\env_ros2_windows.ps1"
  )
  . $EnvScript -RosSetup $RosSetup -DomainId $DomainId
}
Set-Alias use-ros2 Enter-Ros2Env
```

Then each new PowerShell terminal only needs:

```powershell
use-ros2
```

## About RTI warning

This line is expected and can be ignored in this project:

- `RTI Connext DDS environment script not found ...`

Reason:

- Project uses `rmw_cyclonedds_cpp`, not RTI Connext.

## Acceptance checklist

1. `use-ros2` (or script command) prints:
   - `ROS2 Windows environment ready.`
   - `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`
2. `ros2 --help` works
3. `ros2 topic list` works when graph is up
4. Cross-machine test (Windows <-> VM) works under configured `ROS_DOMAIN_ID` and CycloneDDS XML

