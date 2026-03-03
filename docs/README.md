# Project Structure

This repository follows a common GitHub Python layout:

- `src/`: application source code packages
- `tests/`: automated test code
- `examples/`: demo scripts and visualization examples
- `config/`: runtime and environment configuration files
- `docs/`: project documentation
- `scripts/`: runnable helper scripts
- `models/`: model and mesh assets
- `data/`: sample/input data
- `robotic_arm/`: standalone robotic arm Python package
- `stm32_bridge/`: STM32 side firmware sources

Notes:

- The root `main.py` keeps backward compatibility and adds `src/` to `PYTHONPATH` at runtime.
- New modules should be added under `src/`.
- Python dependencies are managed via `requirements.txt` and `environment.yml`.

## Engineering Docs

- `ros2_refactor_plan.md`: phased ROS2 refactor plan with rollback and GitHub sync strategy.
- `github_code_architecture_style.md`: GitHub-side architecture style during refactor (branches, PR, release, repo layout).
- `phase1_kickoff.md`: concrete phase 1 implementation notes and run/verify steps.
- `phase2_hardware_kickoff.md`: phase 2 hardware-layer node migration notes and run/verify steps.
- `phase3_control_kickoff.md`: phase 3 control-layer migration notes (`/control/arm/*`, `MoveArmJoints` action, emergency reset).
- `phase4_ui_bridge_kickoff.md`: phase 4 UI bridge migration status and verification steps.
- `phase5_task_kickoff.md`: phase 5 ROS2 task orchestration and runtime profile switching.
- `../CHANGELOG.md`: repository-level change log (phase progress and merged updates).

## Current Refactor Snapshot (2026-03-03)

- `develop` contains phase 5 stabilization and hardware acceptance fixes.
- `main` is kept in sync with phase 5 merged baseline.
- GUI demo commands are routed to ROS2 task Action/Services in ROS2 mode.
- Bringup supports profile switching with:
  - `phase5_task.yaml` (simulation/default)
  - `phase5_task_hardware.yaml` (hardware profile template)
- Next target: phase 6 stabilization and policy backend migration (SmolVLA integration).
