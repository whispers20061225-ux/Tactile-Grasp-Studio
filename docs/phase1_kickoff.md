# Phase 1 Kickoff (Develop Branch)

This file tracks the first implementation step of the ROS2 migration plan.

## Implemented in this kickoff

- Created `develop` branch as the integration line for staged migration.
- Added `ros2_ws` workspace skeleton.
- Added `tactile_interfaces` package with:
  - `msg/TactileRaw.msg`
  - `msg/SystemHealth.msg`
- Added `tactile_bringup` package with:
  - `fake_tactile_publisher` node
  - `launch/phase1_fake_chain.launch.py`
  - `config/phase1_fake_chain.yaml`
- Added `tactile_ui_bridge` package with:
  - `tactile_ui_subscriber` node (read-only cache bridge)

## Topics in the minimal chain

- `/tactile/raw` (fake tactile stream)
- `/system/health` (basic node health)

## Run guide (Ubuntu + ROS2 Jazzy)

```bash
cd ros2_ws
colcon build
source install/setup.bash
ros2 launch tactile_bringup phase1_fake_chain.launch.py
```

## Validation checklist

- [ ] Launch starts without missing package errors.
- [ ] `/tactile/raw` is visible from `ros2 topic list`.
- [ ] `ros2 topic echo /tactile/raw --once` returns one message.
- [ ] `tactile_ui_subscriber` prints periodic cache logs.
- [ ] Legacy entry (`python main.py`) remains unchanged.

## Rollback

- Safe rollback target: previous `develop` commit before phase 1 kickoff.
- Fast runtime fallback: keep using legacy mode (`main.py` current flow).
