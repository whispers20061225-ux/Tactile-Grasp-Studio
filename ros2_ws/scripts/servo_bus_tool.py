#!/usr/bin/env python3
"""Minimal tool for the STM32 servo line bridge.

Examples:
  python3 ros2_ws/scripts/servo_bus_tool.py scan --ids 1-10
  python3 ros2_ws/scripts/servo_bus_tool.py read --id 6 --addr 0x05 --length 1
  python3 ros2_ws/scripts/servo_bus_tool.py write8 --id 6 --addr 0x28 --value 0
  python3 ros2_ws/scripts/servo_bus_tool.py write16 --id 6 --addr 0x10 --value 200
  python3 ros2_ws/scripts/servo_bus_tool.py set-id --current-id 1 --new-id 6
  python3 ros2_ws/scripts/servo_bus_tool.py line --line-command "GSTAT"
  python3 ros2_ws/scripts/servo_bus_tool.py line --line-command "GFORCE 20"
"""

from __future__ import annotations

import argparse
import socket
import sys
from typing import Iterable
from urllib.parse import urlparse


def parse_endpoint(endpoint: str) -> tuple[str, int]:
    parsed = urlparse(endpoint)
    if parsed.scheme.lower() not in {"tcp", "socket"}:
        raise ValueError(f"unsupported endpoint: {endpoint}")
    host = str(parsed.hostname or "").strip() or "127.0.0.1"
    port = int(parsed.port or 0)
    if port <= 0:
        raise ValueError(f"invalid endpoint port: {endpoint}")
    return host, port


def send_line_command(endpoint: str, command: str, timeout: float) -> str:
    host, port = parse_endpoint(endpoint)
    payload = command.strip()
    if not payload:
        raise ValueError("command is empty")

    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        sock.sendall((payload + "\n").encode("utf-8"))
        data = bytearray()
        while not data.endswith(b"\n"):
            chunk = sock.recv(4096)
            if not chunk:
                break
            data.extend(chunk)
    return data.decode("utf-8", errors="replace").strip()


def expand_ids(expr: str) -> list[int]:
    ids: list[int] = []
    for chunk in str(expr or "").split(","):
        token = chunk.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            step = 1 if end >= start else -1
            ids.extend(list(range(start, end + step, step)))
        else:
            ids.append(int(token))
    deduped: list[int] = []
    seen = set()
    for servo_id in ids:
        if servo_id in seen:
            continue
        seen.add(servo_id)
        deduped.append(servo_id)
    return deduped


def run_scan(endpoint: str, ids: Iterable[int], timeout: float) -> int:
    found = []
    for servo_id in ids:
        response = send_line_command(endpoint, f"SPING {servo_id}", timeout)
        print(f"{servo_id}: {response}")
        if response == "OK":
            found.append(servo_id)
    print(f"FOUND: {found}")
    return 0


def run_read(endpoint: str, servo_id: int, addr: int, length: int, timeout: float) -> int:
    response = send_line_command(endpoint, f"SREADREG {servo_id} {addr} {length}", timeout)
    print(response)
    return 0


def run_set_id(endpoint: str, current_id: int, new_id: int, timeout: float) -> int:
    if current_id <= 0 or current_id > 250:
        raise ValueError("current servo id must be in 1..250")
    if new_id <= 0 or new_id > 250 or new_id == 254:
        raise ValueError("new servo id must be in 1..250 and cannot be 254")
    response = send_line_command(endpoint, f"SSETID {current_id} {new_id}", timeout)
    print(response)
    return 0


def run_write8(endpoint: str, servo_id: int, addr: int, value: int, timeout: float) -> int:
    response = send_line_command(endpoint, f"SWRITE8 {servo_id} {addr} {value}", timeout)
    print(response)
    return 0


def run_write16(endpoint: str, servo_id: int, addr: int, value: int, timeout: float) -> int:
    response = send_line_command(endpoint, f"SWRITE16 {servo_id} {addr} {value}", timeout)
    print(response)
    return 0


def run_line(endpoint: str, command: str, timeout: float) -> int:
    response = send_line_command(endpoint, command, timeout)
    print(response)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="STM32 servo bus helper over the line bridge")
    parser.add_argument(
        "--endpoint",
        default="tcp://172.17.192.1:19024",
        help="bridge endpoint, default: tcp://172.17.192.1:19024",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=2.0,
        help="socket timeout in seconds",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser("scan", help="scan servo ids with SPING")
    scan_parser.add_argument("--ids", default="1-10", help="id list/range, e.g. 1-6 or 1,3,6")

    read_parser = subparsers.add_parser("read", help="read a raw servo register")
    read_parser.add_argument("--id", dest="servo_id", required=True, type=int)
    read_parser.add_argument("--addr", required=True, type=lambda value: int(value, 0))
    read_parser.add_argument("--length", required=True, type=lambda value: int(value, 0))

    write8_parser = subparsers.add_parser("write8", help="write an 8-bit servo register")
    write8_parser.add_argument("--id", dest="servo_id", required=True, type=int)
    write8_parser.add_argument("--addr", required=True, type=lambda value: int(value, 0))
    write8_parser.add_argument("--value", required=True, type=lambda value: int(value, 0))

    write16_parser = subparsers.add_parser("write16", help="write a 16-bit servo register")
    write16_parser.add_argument("--id", dest="servo_id", required=True, type=int)
    write16_parser.add_argument("--addr", required=True, type=lambda value: int(value, 0))
    write16_parser.add_argument("--value", required=True, type=lambda value: int(value, 0))

    set_id_parser = subparsers.add_parser("set-id", help="change servo id")
    set_id_parser.add_argument("--current-id", required=True, type=int)
    set_id_parser.add_argument("--new-id", required=True, type=int)

    line_parser = subparsers.add_parser("line", help="send a raw line command to the STM32 bridge")
    line_parser.add_argument("--line-command", required=True, type=str)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "scan":
            return run_scan(args.endpoint, expand_ids(args.ids), args.timeout)
        if args.command == "read":
            return run_read(args.endpoint, args.servo_id, args.addr, args.length, args.timeout)
        if args.command == "write8":
            return run_write8(args.endpoint, args.servo_id, args.addr, args.value, args.timeout)
        if args.command == "write16":
            return run_write16(args.endpoint, args.servo_id, args.addr, args.value, args.timeout)
        if args.command == "set-id":
            return run_set_id(args.endpoint, args.current_id, args.new_id, args.timeout)
        if args.command == "line":
            return run_line(args.endpoint, args.line_command, args.timeout)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
