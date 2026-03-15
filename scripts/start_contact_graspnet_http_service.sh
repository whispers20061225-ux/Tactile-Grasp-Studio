#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONTACT_GRASPNET_ROOT="${CONTACT_GRASPNET_ROOT:-${HOME}/contact_graspnet}"
CONTACT_GRASPNET_CKPT_DIR="${CONTACT_GRASPNET_CKPT_DIR:-${CONTACT_GRASPNET_ROOT}/checkpoints/scene_test_2048_bs3_hor_sigma_001}"
CONTACT_GRASPNET_PYTHON="${CONTACT_GRASPNET_PYTHON:-python}"
CONTACT_GRASPNET_HOST="${CONTACT_GRASPNET_HOST:-127.0.0.1}"
CONTACT_GRASPNET_PORT="${CONTACT_GRASPNET_PORT:-5001}"
CONTACT_GRASPNET_DEFAULT_VISUALIZE="${CONTACT_GRASPNET_DEFAULT_VISUALIZE:-0}"

extra_args=()
if [[ "${CONTACT_GRASPNET_DEFAULT_VISUALIZE}" != "0" ]]; then
  extra_args+=(--default-visualize)
fi

exec "${CONTACT_GRASPNET_PYTHON}" "${REPO_ROOT}/scripts/contact_graspnet_http_service.py" \
  --host "${CONTACT_GRASPNET_HOST}" \
  --port "${CONTACT_GRASPNET_PORT}" \
  --contact-graspnet-root "${CONTACT_GRASPNET_ROOT}" \
  --ckpt-dir "${CONTACT_GRASPNET_CKPT_DIR}" \
  "${extra_args[@]}"
