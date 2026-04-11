#!/usr/bin/env bash
set -euo pipefail

REALSENSE_VIDPID="${REALSENSE_VIDPID:-8086:0b5c}"
USBIPD_EXE_WIN='C:\Program Files\usbipd-win\usbipd.exe'
WSL_DISTRO_NAME_VALUE="${WSL_DISTRO_NAME:-Ubuntu-24.04}"
ATTACH_TIMEOUT_SEC="${ATTACH_TIMEOUT_SEC:-20}"

log() {
  printf '[ensure-realsense] %s\n' "$*"
}

if lsusb 2>/dev/null | grep -qi "$REALSENSE_VIDPID"; then
  log "RealSense already attached to WSL ($REALSENSE_VIDPID)"
  exit 0
fi

if ! command -v powershell.exe >/dev/null 2>&1; then
  log "powershell.exe is unavailable; skipping usbip attach"
  exit 0
fi

attach_cmd="& '$USBIPD_EXE_WIN' attach --wsl $WSL_DISTRO_NAME_VALUE --hardware-id $REALSENSE_VIDPID --auto-attach"
log "attempting usbip attach for $REALSENSE_VIDPID"
powershell.exe -NoProfile -Command "Start-Process -WindowStyle Hidden powershell.exe -ArgumentList '-NoProfile','-Command',\"$attach_cmd\"" >/dev/null 2>&1 || true

for ((i=1; i<=ATTACH_TIMEOUT_SEC; i++)); do
  if lsusb 2>/dev/null | grep -qi "$REALSENSE_VIDPID"; then
    log "RealSense attached after ${i}s"
    exit 0
  fi
  sleep 1
done

log "RealSense did not appear in WSL after attach attempt"
exit 0
