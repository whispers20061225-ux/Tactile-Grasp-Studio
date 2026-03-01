import serial
import time

PORT = "COM4"     # 改成你的
BAUD = 115200
TIMEOUT = 0.5

def send_cmd(ser, cmd: str, wait=0.05):
    if not cmd.endswith("\n"):
        cmd += "\n"
    ser.write(cmd.encode("utf-8"))
    time.sleep(wait)
    resp = ser.readline().decode(errors="ignore").strip()
    return resp

def main():
    ser = serial.Serial(PORT, BAUD, timeout=TIMEOUT)
    print("串口已打开:", ser.name)
    time.sleep(1.0)  # 让 STM32 复位完成

    print(">> PING")
    print("<<", send_cmd(ser, "PING"))

    print(">> VER")
    print("<<", send_cmd(ser, "VER"))

    # 1) 扫描 1~6
    for sid in range(1, 7):
        cmd = f"SPING {sid}"
        resp = send_cmd(ser, cmd)
        print(f">> {cmd}")
        print("<<", resp)

    # 2) 读位置示例
    sid = 1                                                
    cmd = f"SREADPOS {sid}"
    resp = send_cmd(ser, cmd)
    print(f">> {cmd}")
    print("<<", resp)

    # 3) 运动示例（注意：pos 是“位置值”，不是角度）
    # 这里给两个常见的测试值，你需要按你的舵机协议实际范围调整：
    # 例如某些舵机 pos: 0~1000；也可能是 0~4095；或 500~2500(脉宽)
    tests = [200, 500, 800, 500]
    ms = 800
    for pos in tests:
        cmd = f"SMOVE {sid} {pos} {ms}"
        resp = send_cmd(ser, cmd, wait=0.1)
        print(f">> {cmd}")
        print("<<", resp)
        time.sleep(ms / 1000.0 + 0.2)

        

    ser.close()
    print("串口已关闭")

if __name__ == "__main__":
    main()
