#!/usr/bin/env bash
set -euo pipefail

DOMAIN_ID="${1:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ROS_SETUP="/opt/ros/jazzy/setup.bash"
WS_SETUP="${PROJECT_ROOT}/ros2_ws/install/setup.bash"

source_setup_compat() {
  local setup_path="$1"
  local had_nounset=0
  case "$-" in
    *u*) had_nounset=1 ;;
  esac

  # Some ROS setup scripts read vars that may be unset under `set -u`.
  export AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES:-}"
  set +u
  # shellcheck disable=SC1090
  source "${setup_path}"
  if [[ "${had_nounset}" -eq 1 ]]; then
    set -u
  fi
}

if [[ -f "${ROS_SETUP}" ]]; then
  source_setup_compat "${ROS_SETUP}"
else
  echo "ROS2 setup not found: ${ROS_SETUP}" >&2
  exit 1
fi

if [[ -f "${WS_SETUP}" ]]; then
  source_setup_compat "${WS_SETUP}"
fi

export ROS_DOMAIN_ID="${DOMAIN_ID}"
export RMW_IMPLEMENTATION="rmw_cyclonedds_cpp"
export ROS_LOCALHOST_ONLY="0"
export CYCLONEDDS_URI="file://${PROJECT_ROOT}/config/dds/cyclonedds_vm.xml"

echo "ROS2 VM environment ready."
echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
echo "RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION}"
echo "CYCLONEDDS_URI=${CYCLONEDDS_URI}"
