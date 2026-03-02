# Phase 4 UI Bridge Status

This document tracks phase 4 (GUI bridge migration) after integration to `main`.

## Scope of phase 4

- Keep PyQt GUI appearance and interaction flow unchanged.
- Switch GUI data source to ROS2 topics.
- Switch GUI control command output to ROS2 control services.
- Keep rollback path (`main.py` legacy and `main_ros2.py --control-mode stub`).

## Current status (2026-03-03)

- Branch status:
  - Phase 4 updates are merged into `main`.
- ROS2 GUI entry:
  - `main_ros2.py` is the phase-4 runtime entry.
- Data bridge:
  - GUI subscribes to `/tactile/raw` and `/system/health`.
- Control bridge:
  - GUI arm control path is bridged to:
    - `/control/arm/enable`
    - `/control/arm/home`
    - `/control/arm/move_joint`
    - `/control/arm/move_joints`
    - `/system/reset_emergency`
- Compatibility command bridge (newly completed):
  - `move_gripper`, `set_servo_position`, `set_servo_speed`, `set_servo_force`
  - `calibrate_hardware`, `calibrate_3d`
  - `auto_grasp` (phase-4 compatibility path)
- Stability fixes included:
  - Ctrl+C shutdown no longer logs ROS2 thread crashes (`ExternalShutdownException` handled).
  - Matplotlib Chinese font warnings reduced and CJK font probing added.

## Important behavior notes

- Single-arm architecture:
  - No dedicated gripper node is required in current hardware model.
  - Gripper-related UI commands are mapped to an arm joint service call in ROS2 bridge layer.
- Task orchestration:
  - `start_demo/pause_demo/resume_demo` are still compatibility behavior.
  - Full ROS2 Action task orchestration is phase 5 scope.

## Runtime commands (VM)

Terminal A:

```bash
cd /home/zhuyiwei/programme/programme
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dayiprogramme312
source /opt/ros/jazzy/setup.bash
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
ros2 launch tactile_bringup phase3_control.launch.py
```

Terminal B:

```bash
cd /home/zhuyiwei/programme/programme
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dayiprogramme312
source /opt/ros/jazzy/setup.bash
source /home/zhuyiwei/programme/programme/ros2_ws/install/setup.bash
python main_ros2.py --control-mode ros2 --log-level INFO
```

Terminal C (verification):

```bash
cd /home/zhuyiwei/programme/programme
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dayiprogramme312
source /opt/ros/jazzy/setup.bash
source /home/zhuyiwei/programme/programme/ros2_ws/install/setup.bash
ros2 node list
ros2 topic list
ros2 service list | grep /control/arm
```

## Exit criteria for phase 4

- GUI tactile data updates correctly from ROS2 topic stream.
- GUI arm commands are executed through `/control/arm/*`.
- Legacy servo/gripper UI controls are no longer ignored in ROS2 mode.
- `main.py` legacy path remains runnable.

## Remaining gap to phase 5

- Replace compatibility demo handling with ROS2 Action orchestration.
- Migrate `start_demo/pause_demo/resume_demo/stop_demo` to task action server.
