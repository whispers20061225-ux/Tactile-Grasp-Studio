#!/usr/bin/env python3
"""德晟总线舵机协议测试脚本（PING/读位置/写位置）"""
import argparse
import time

try:
    import serial
except ImportError:
    serial = None

HEADER_TX = bytes([0xFF, 0xFF])
HEADER_RX = bytes([0xFF, 0xF5])

CMD_PING = 0x01
CMD_READ = 0x02
CMD_WRITE = 0x03

ADDR_GOAL_POS = 0x2A
ADDR_MOVE_TIME = 0x2C
ADDR_PRESENT_POS = 0x38


def checksum(servo_id, length, cmd, params):
    # 校验和：按协议取反并截断 8 位
    total = (servo_id + length + cmd + sum(params)) & 0xFF
    return (~total) & 0xFF


def build_packet(servo_id, cmd, params):
    # 组包：FF FF + ID + LEN + CMD + PARAMS + CHECKSUM
    length = len(params) + 2
    ck = checksum(servo_id, length, cmd, params)
    body = bytes([servo_id & 0xFF, length & 0xFF, cmd & 0xFF] + [p & 0xFF for p in params] + [ck])
    return HEADER_TX + body


def read_response(ser, timeout=0.3):
    # 读取响应帧：FF F5 + ID + LEN + STATUS + PARAMS + CHECKSUM
    deadline = time.time() + timeout
    buf = bytearray()
    while time.time() < deadline:
        b = ser.read(1)
        if not b:
            continue
        buf += b
        if len(buf) >= 2 and buf[-2:] == HEADER_RX:
            id_b = ser.read(1)
            len_b = ser.read(1)
            status_b = ser.read(1)
            if not id_b or not len_b or not status_b:
                return None
            servo_id = id_b[0]
            length = len_b[0]
            status = status_b[0]
            params_len = max(0, length - 2)
            params = list(ser.read(params_len)) if params_len else []
            ck_b = ser.read(1)
            if not ck_b:
                return None
            return {
                "id": servo_id,
                "length": length,
                "status": status,
                "params": params,
                "checksum": ck_b[0],
            }
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--id", type=int, default=1)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--angle", type=float, default=90.0)
    parser.add_argument("--time", type=int, default=1000)
    args = parser.parse_args()

    if serial is None:
        print("pyserial is required: pip install pyserial")
        return

    ser = serial.Serial(args.port, args.baud, timeout=0.2)

    # PING
    pkt = build_packet(args.id, CMD_PING, [])
    ser.write(pkt)
    resp = read_response(ser)
    print("PING resp:", resp)

    # 读取当前位置
    pkt = build_packet(args.id, CMD_READ, [ADDR_PRESENT_POS, 0x02])
    ser.write(pkt)
    resp = read_response(ser)
    print("READ pos resp:", resp)

    # 写入目标位置 + 运行时间
    pos_raw = int(900 + (args.angle / 180.0) * (3100 - 900))
    params = [
        ADDR_GOAL_POS,
        (pos_raw >> 8) & 0xFF, pos_raw & 0xFF,
        (args.time >> 8) & 0xFF, args.time & 0xFF,
    ]
    pkt = build_packet(args.id, CMD_WRITE, params)
    ser.write(pkt)
    print("WRITE sent: pos_raw=", pos_raw)

    ser.close()


if __name__ == "__main__":
    main()
