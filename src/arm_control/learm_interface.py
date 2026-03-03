"""
LeArm 接口：基于 STM32 USB-CDC 的行命令方式。

期望的命令/响应（按行）：
  PING -> PONG
  VER -> VER <name> <version> <date> <time>
  SPING <id> -> OK/FAIL
  SREADPOS <id> -> POS <id> <pos> | FAIL <id>
  SMOVE <id> <pos> <ms> -> OK/FAIL
"""

import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

# 运行成脚本时，Python 默认只把当前文件所在目录加入 sys.path，
# 顶层的 config 包不在搜索路径里，会导致 ModuleNotFoundError。
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import serial
except ImportError:
    # 允许没有 pyserial 的环境继续加载模块
    serial = None

from config.demo_config import DemoConfig

logger = logging.getLogger(__name__)


class ArmConnectionType(Enum):
    """连接方式枚举。"""

    SERIAL = "SERIAL"


@dataclass
class ArmStatus:
    """机械臂状态快照（用于 UI/上层展示）。"""

    connected: bool = False
    moving: bool = False
    error: bool = False
    error_msg: str = ""
    battery_voltage: float = 0.0
    joint_positions: List[int] = None
    joint_angles: List[float] = None
    temperature: Optional[float] = None


class LearmInterface:
    """LeArm 接口：使用 USB-CDC 行命令控制机械臂。"""

    def __init__(self, config: DemoConfig):
        # 保存配置并解析机械臂连接参数
        self.config = config
        self.arm_profile, self.arm_config = self._resolve_configs(config)

        # 初始化状态缓存（关节原始位置/角度）
        self.status = ArmStatus()
        self.status.joint_positions = [2048] * 6
        self.status.joint_angles = [0.0] * 6

        # 串口句柄与线程锁
        self._serial = None
        self._lock = threading.RLock()
        self._owns_serial = True  # 是否拥有串口句柄（共享时为 False）

        # 机械臂基础参数
        self.num_joints = 6
        self.default_duration = 1000
        self.default_wait = False
        self._startup_delay = float(self.arm_config.get("startup_delay", 1.0))

        # 关节零点偏移（用于“当前姿态设为0度”）
        offsets = self.arm_config.get("joint_zero_offsets_deg", None)
        if not offsets:
            # 若未在连接配置中指定，则尝试从默认配置读取固定零点
            try:
                from config.learm_arm_config import LearmArmConfig
                offsets = LearmArmConfig.CONNECTION.get("joint_zero_offsets_deg")
            except Exception:
                offsets = None
        if isinstance(offsets, (list, tuple)) and offsets:
            self.joint_zero_offsets_deg = [float(v) for v in offsets]
        else:
            self.joint_zero_offsets_deg = [0.0] * self.num_joints

        # 对指定关节禁用角度软限制，允许负角度/超范围映射
        disabled_ids = self.arm_config.get("joint_limit_disabled_ids", None)
        if not disabled_ids:
            # 若连接配置未显式给出，则回退到默认 LeArm 配置
            try:
                from config.learm_arm_config import LearmArmConfig

                disabled_ids = LearmArmConfig.CONNECTION.get("joint_limit_disabled_ids")
            except Exception:
                disabled_ids = None
        try:
            self._limit_disabled_joints = {int(v) for v in (disabled_ids or [])}
        except Exception:
            self._limit_disabled_joints = set()

        # 角度->位置映射范围（来自 LeArm 参考值）
        self._range_default = (900, 3100)
        self._range_j5 = (380, 3700)
        # 安全位置上下限（防止发送负位置）
        self._pos_min_safe = 0
        self._pos_max_safe = 4095

        logger.info("LeArm line-command interface initialized")

    def _resolve_configs(self, config: DemoConfig):
        # 从 DemoConfig 中提取连接参数：优先 learm_arm.CONNECTION
        arm_profile = None
        connection_cfg = {}
        if hasattr(config, "learm_arm") and config.learm_arm is not None:
            arm_profile = config.learm_arm
            if isinstance(arm_profile, dict):
                connection_cfg = arm_profile.get("CONNECTION", arm_profile)
            else:
                connection_cfg = getattr(arm_profile, "CONNECTION", {}) or {}
        elif hasattr(config, "CONNECTION"):
            arm_profile = config
            connection_cfg = getattr(config, "CONNECTION", {}) or {}
        return arm_profile, connection_cfg

    def connect(
        self,
        connection_type: Optional[Union[str, ArmConnectionType]] = None,
        port: Optional[str] = None,
        shared_serial=None,
        shared_lock=None,
    ) -> bool:
        # 建立串口连接并通过 PING 做连通性测试
        if serial is None:
            logger.error("pyserial is not installed")
            return False

        with self._lock:
            try:
                # 使用共享串口（STM32 同一 CDC 口），不需要额外的端口配置
                if shared_serial is not None:
                    if shared_lock is not None:
                        self._lock = shared_lock
                    self._serial = shared_serial
                    self._owns_serial = False
                    if not (self._serial and self._serial.is_open):
                        raise ValueError("shared serial is not open")

                    # 使用 PING 验证链路可用性
                    if not self._ping_link():
                        raise RuntimeError("PING failed")

                    self.status.connected = True
                    self.status.error = False
                    self.status.error_msg = ""
                    logger.info("Connected to LeArm via shared STM32 serial")
                    return True

                if connection_type is None:
                    connection_type = ArmConnectionType.SERIAL
                if isinstance(connection_type, str):
                    connection_type = ArmConnectionType(connection_type.upper())

                if connection_type != ArmConnectionType.SERIAL:
                    raise ValueError(f"Unsupported connection type: {connection_type}")

                # 读取端口配置（可由调用方覆盖）
                if not port:
                    port = self.arm_config.get("serial_port")
                if not port:
                    raise ValueError("serial_port is not configured")

                # 串口参数（波特率/超时）
                baudrate = int(self.arm_config.get("baud_rate", 115200))
                timeout = float(self.arm_config.get("timeout", 1.0))

                self._serial = serial.Serial(
                    port=port,
                    baudrate=baudrate,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    timeout=timeout,
                )
                self._owns_serial = True

                # 给 MCU 一点时间重启并准备 CDC
                time.sleep(self._startup_delay)
                try:
                    self._serial.reset_input_buffer()
                    self._serial.reset_output_buffer()
                except Exception:
                    pass

                # 通过 PING 验证链路
                if not self._ping_link():
                    raise RuntimeError("PING failed")

                self.status.connected = True
                self.status.error = False
                self.status.error_msg = ""

                logger.info("Connected to LeArm over serial: %s", port)
                return True

            except Exception as exc:
                self.status.connected = False
                self.status.error = True
                self.status.error_msg = str(exc)
                logger.error("Failed to connect arm: %s", exc)
                return False

    def disconnect(self):
        # 断开串口连接并清理状态
        with self._lock:
            if self._serial:
                if self._owns_serial:
                    try:
                        self._serial.close()
                    except Exception:
                        pass
            self._serial = None
            self.status.connected = False
            self._owns_serial = True

    def is_connected(self) -> bool:
        # 判断串口是否处于打开状态
        return bool(self._serial and self._serial.is_open)

    def ping(self, servo_id: int) -> bool:
        # 单个舵机连通性测试（SPING <id>）
        if not self.is_connected():
            return False
        resp = self._send_cmd(f"SPING {int(servo_id)}")
        return self._is_ok_response(resp)

    def move_joint(
        self,
        servo_id: int,
        position: Union[int, float],
        duration: Optional[int] = None,
        wait: Optional[bool] = None,
    ) -> bool:
        # 移动单个关节（角度/位置 -> 原始位置值）
        if not self.is_connected():
            logger.error("Arm is not connected")
            return False

        if duration is None:
            duration = self.default_duration
        if wait is None:
            wait = self.default_wait

        # 角度转位置、构造 SMOVE 命令
        pos_raw = self._angle_to_position(servo_id, position)
        time_ms = max(0, int(duration))

        resp = self._send_cmd(f"SMOVE {int(servo_id)} {int(pos_raw)} {time_ms}")
        ok = self._is_ok_response(resp)

        if wait and time_ms > 0:
            time.sleep(time_ms / 1000.0)
        return ok

    def move_joint_raw(
        self,
        servo_id: int,
        position: Union[int, float],
        duration: Optional[int] = None,
        wait: Optional[bool] = None,
    ) -> bool:
        """
        直接发送原始位置值（不做角度映射）。

        Args:
            servo_id: 舵机 ID
            position: 原始位置值
            duration: 运动时间（ms）
            wait: 是否等待动作完成
        """
        if not self.is_connected():
            logger.error("Arm is not connected")
            return False

        if duration is None:
            duration = self.default_duration
        if wait is None:
            wait = self.default_wait

        pos_raw = int(position)
        time_ms = max(0, int(duration))

        resp = self._send_cmd(f"SMOVE {int(servo_id)} {int(pos_raw)} {time_ms}")
        ok = self._is_ok_response(resp)

        if wait and time_ms > 0:
            time.sleep(time_ms / 1000.0)
        return ok

    def move_joints(
        self,
        positions: Dict[int, Union[int, float]],
        duration: Optional[int] = None,
        wait: Optional[bool] = None,
    ) -> bool:
        # 依次下发每个舵机的 SMOVE（当前协议不支持批量同步）
        if not self.is_connected():
            logger.error("Arm is not connected")
            return False

        if duration is None:
            duration = self.default_duration
        if wait is None:
            wait = self.default_wait

        time_ms = max(0, int(duration))
        ok = True
        for servo_id in sorted(positions.keys()):
            # 按关节编号排序，保证命令发送顺序稳定
            pos_raw = self._angle_to_position(servo_id, positions[servo_id])
            resp = self._send_cmd(f"SMOVE {int(servo_id)} {int(pos_raw)} {time_ms}")
            ok = ok and self._is_ok_response(resp)

        if wait and time_ms > 0:
            time.sleep(time_ms / 1000.0)
        return ok

    def move_joints_raw(
        self,
        positions: Dict[int, Union[int, float]],
        duration: Optional[int] = None,
        wait: Optional[bool] = None,
    ) -> bool:
        """
        直接发送原始位置值（批量逐个发送）。

        Args:
            positions: {id: 原始位置}
            duration: 运动时间（ms）
            wait: 是否等待动作完成
        """
        if not self.is_connected():
            logger.error("Arm is not connected")
            return False

        if duration is None:
            duration = self.default_duration
        if wait is None:
            wait = self.default_wait

        time_ms = max(0, int(duration))
        ok = True
        for servo_id in sorted(positions.keys()):
            pos_raw = int(positions[servo_id])
            resp = self._send_cmd(f"SMOVE {int(servo_id)} {int(pos_raw)} {time_ms}")
            ok = ok and self._is_ok_response(resp)

        if wait and time_ms > 0:
            time.sleep(time_ms / 1000.0)
        return ok

    def get_joint_position(self, servo_id: int, degrees: bool = True) -> Optional[float]:
        # 读取单个关节的当前位置（SREADPOS <id>）
        if not self.is_connected():
            return None

        resp = self._send_cmd(f"SREADPOS {int(servo_id)}")
        pos = self._parse_pos_response(resp)
        if pos is None:
            return None

        if degrees:
            return self._position_to_angle(servo_id, pos)
        return float(pos)

    def get_all_joint_positions(self, degrees: bool = True) -> Dict[int, float]:
        # 依次读取所有关节的位置
        positions: Dict[int, float] = {}
        for servo_id in range(1, self.num_joints + 1):
            pos = self.get_joint_position(servo_id, degrees=degrees)
            if pos is not None:
                positions[servo_id] = pos
        return positions

    def servo_on(self, servo_id: Optional[int] = None):
        # 该协议未定义扭矩开关命令，保留接口以兼容上层调用
        logger.warning("servo_on is not supported by the line protocol")

    def servo_off(self, servo_id: Optional[int] = None):
        # 该协议未定义扭矩开关命令，保留接口以兼容上层调用
        logger.warning("servo_off is not supported by the line protocol")

    def update_status(self) -> ArmStatus:
        # 从硬件读取状态并更新缓存
        if not self.is_connected():
            return self.status

        # 使用上一帧作为兜底，避免单次读失败导致角度回零
        positions = list(self.status.joint_positions or [])
        angles = list(self.status.joint_angles or [])
        if len(positions) < self.num_joints:
            positions += [2048] * (self.num_joints - len(positions))
        if len(angles) < self.num_joints:
            angles += [0.0] * (self.num_joints - len(angles))

        for servo_id in range(1, self.num_joints + 1):
            pos = self.get_joint_position(servo_id, degrees=False)
            if pos is None:
                # 读取失败时保留上一帧，避免 UI 突然跳回 0
                continue
            positions[servo_id - 1] = int(pos)
            angles[servo_id - 1] = self._position_to_angle(servo_id, int(pos))

        self.status.joint_positions = positions
        self.status.joint_angles = angles
        self.status.connected = True
        self.status.error = False
        self.status.error_msg = ""
        return self.status

    def set_zero_offsets(self, offsets_deg: List[float]) -> List[float]:
        """
        设置关节零点偏移（度）。

        Args:
            offsets_deg: 每个关节的零点偏移角度

        Returns:
            实际生效的偏移列表
        """
        offsets = [float(v) for v in offsets_deg]
        if len(offsets) < self.num_joints:
            offsets = offsets + [0.0] * (self.num_joints - len(offsets))
        self.joint_zero_offsets_deg = offsets[: self.num_joints]
        # 同步到配置（仅内存中，便于后续读取）
        try:
            self.arm_config["joint_zero_offsets_deg"] = list(self.joint_zero_offsets_deg)
        except Exception:
            pass
        return list(self.joint_zero_offsets_deg)

    def set_zero_offsets_from_current(self) -> Optional[List[float]]:
        """
        读取当前关节角度，并将其记录为“零点偏移”。

        Returns:
            设置后的零点偏移角度列表；失败返回 None
        """
        if not self.is_connected():
            return None

        offsets: List[float] = []
        for servo_id in range(1, self.num_joints + 1):
            pos = self.get_joint_position(servo_id, degrees=False)
            if pos is None:
                return None
            offsets.append(self._position_to_angle_raw(servo_id, int(pos)))

        return self.set_zero_offsets(offsets)

    def emergency_stop(self):
        # 紧急停止：当前实现为关闭扭矩（若协议支持）
        self.servo_off()

    def set_speed(self, speed: float):
        # 速度映射为默认运动时间（越快 -> 时间越短）
        speed = max(0.0, min(1.0, speed))
        self.default_duration = int(1000 * (1.0 - speed * 0.8))

    def _ping_link(self) -> bool:
        # 基础连通性测试（PING/PONG）
        resp = self._send_cmd("PING")
        return bool(resp and resp.strip().upper() == "PONG")

    def _send_cmd(self, cmd: str) -> Optional[str]:
        # 发送一行命令并读取单行响应
        if not self._serial:
            return None

        line = cmd.strip()
        if not line:
            return None

        with self._lock:
            try:
                # 清空残留输入，避免混入旧数据
                if self._serial.in_waiting:
                    self._serial.reset_input_buffer()

                # 发送命令行
                payload = (line + "\n").encode("utf-8")
                self._serial.write(payload)
                self._serial.flush()

                # 读取一行响应（以 \n 结束）
                resp = self._serial.readline()
                if not resp:
                    return None

                return resp.decode("utf-8", errors="ignore").strip()
            except Exception as exc:
                logger.error("Command failed (%s): %s", line, exc)
                return None

    def _is_ok_response(self, resp: Optional[str]) -> bool:
        # 判断是否为 OK 响应
        return bool(resp and resp.strip().upper() == "OK")

    def _parse_pos_response(self, resp: Optional[str]) -> Optional[int]:
        # 解析 POS <id> <pos> 响应
        if not resp:
            return None
        parts = resp.strip().split()
        if not parts:
            return None
        if parts[0].upper() == "POS" and len(parts) >= 3:
            try:
                return int(parts[2])
            except ValueError:
                return None
        if parts[0].upper() == "FAIL":
            return None
        return None

    def _angle_to_position(self, servo_id: int, angle: Union[int, float]) -> int:
        # 角度 -> 原始位置值（不同关节范围不同）
        angle_f = float(angle)
        # 叠加零点偏移，使“0度”映射到记录的姿态
        if 1 <= servo_id <= self.num_joints:
            angle_f += self.joint_zero_offsets_deg[servo_id - 1]

        limit_disabled = servo_id in self._limit_disabled_joints

        if servo_id == 5:
            pos_min, pos_max = self._range_j5
            if not limit_disabled:
                angle_f = max(0.0, min(270.0, angle_f))
            pos = pos_min + (angle_f / 270.0) * (pos_max - pos_min)
        else:
            pos_min, pos_max = self._range_default
            if not limit_disabled:
                angle_f = max(0.0, min(180.0, angle_f))
            pos = pos_min + (angle_f / 180.0) * (pos_max - pos_min)
            if servo_id in (2, 3, 4):
                pos = pos_max - (pos - pos_min)

        # 禁用限制时仅做安全边界裁剪，避免发送负位置
        if limit_disabled:
            pos = max(self._pos_min_safe, min(self._pos_max_safe, pos))

        return int(round(pos))

    def _position_to_angle_raw(self, servo_id: int, pos: int) -> float:
        # 原始位置值 -> 角度（未应用零点偏移）
        limit_disabled = servo_id in self._limit_disabled_joints
        if servo_id == 5:
            pos_min, pos_max = self._range_j5
            if limit_disabled:
                pos = max(self._pos_min_safe, min(self._pos_max_safe, pos))
            else:
                pos = max(pos_min, min(pos_max, pos))
            angle = (pos - pos_min) / float(pos_max - pos_min) * 270.0
        else:
            pos_min, pos_max = self._range_default
            if limit_disabled:
                pos = max(self._pos_min_safe, min(self._pos_max_safe, pos))
            else:
                pos = max(pos_min, min(pos_max, pos))
            if servo_id in (2, 3, 4):
                pos = pos_max - (pos - pos_min)
            angle = (pos - pos_min) / float(pos_max - pos_min) * 180.0
        return float(angle)

    def _position_to_angle(self, servo_id: int, pos: int) -> float:
        # 原始位置值 -> 角度（应用零点偏移）
        angle = self._position_to_angle_raw(servo_id, pos)
        if 1 <= servo_id <= self.num_joints:
            angle -= self.joint_zero_offsets_deg[servo_id - 1]
        return float(angle)


def create_learrm_interface(config: DemoConfig) -> LearmInterface:
    # 工厂方法，保持外部调用一致
    return LearmInterface(config)


if __name__ == "__main__":
    # 简单连通性测试：打开串口 -> PING -> SPING 1~6 -> 读取位置
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="LeArm line-command connectivity test")
    parser.add_argument("--port", type=str, default="COM4", help="serial port")
    parser.add_argument("--baud", type=int, default=115200, help="baud rate")
    parser.add_argument("--timeout", type=float, default=1.0, help="serial timeout")
    parser.add_argument("--startup-delay", type=float, default=1.0, help="MCU startup delay")
    parser.add_argument(
        "--ids",
        type=str,
        default="1,2,3,4,5,6",
        help="servo ids, e.g. 1,2,3",
    )
    parser.add_argument(
        "--read-pos",
        action="store_true",
        help="also read joint positions",
    )
    args = parser.parse_args()

    # 构造最小配置并注入连接参数
    demo_config = DemoConfig()
    demo_config.learm_arm = {
        "CONNECTION": {
            "serial_port": args.port,
            "baud_rate": args.baud,
            "timeout": args.timeout,
            "startup_delay": args.startup_delay,
        }
    }

    arm = LearmInterface(demo_config)
    ok = arm.connect()
    if not ok:
        logger.error("Connect failed")
        raise SystemExit(1)

    logger.info("PING link: OK")

    # 解析舵机 ID 列表
    ids: List[int] = []
    for part in args.ids.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            ids.append(int(part))
        except ValueError:
            logger.warning("Invalid servo id: %s", part)

    if not ids:
        ids = list(range(1, arm.num_joints + 1))

    # 逐个 SPING 测试
    for sid in ids:
        result = arm.ping(sid)
        logger.info("SPING %s -> %s", sid, "OK" if result else "FAIL")

    # 可选读取关节位置
    if args.read_pos:
        for sid in ids:
            pos = arm.get_joint_position(sid, degrees=False)
            logger.info("SREADPOS %s -> %s", sid, pos)

    arm.disconnect()
