#!/usr/bin/env bash
set -euo pipefail

DOMAIN_ID="${1:-0}"
PARAM_FILE="${2:-}"
START_TACTILE_SENSOR="${3:-true}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env_ros2_vm.sh" "${DOMAIN_ID}"

if [[ -n "${PARAM_FILE}" ]]; then
  exec ros2 launch tactile_bringup split_vm_app.launch.py \
    param_file:="${PARAM_FILE}" \
    start_tactile_sensor:="${START_TACTILE_SENSOR}"
else
  exec ros2 launch tactile_bringup split_vm_app.launch.py \
    start_tactile_sensor:="${START_TACTILE_SENSOR}"
fi

