"""
PyBullet GUI 进程客户端。

负责：
- 启动独立 PyBullet GUI 进程
- 通过本地 socket 发送控制指令
- 获取仿真状态供 UI 显示
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from typing import Any, Dict, Optional


class PyBulletProcessClient:
    """UI 侧使用的进程客户端"""

    def __init__(self, config: Any):
        self._config = config
        self._port = self._pick_free_port()
        self._process: Optional[subprocess.Popen] = None
        self.running = False
        # 记录项目根目录，保证子进程能找到 simulation 包
        self._project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        # 子进程日志路径
        self._log_path = os.path.join(self._project_root, "logs", "pybullet_gui.log")
        self._log_fp = None

    def start(self) -> None:
        """启动子进程并初始化配置"""
        if self._process is None or self._process.poll() is not None:
            os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
            self._log_fp = open(self._log_path, "a", encoding="utf-8")
            cmd = [
                sys.executable,
                "-m",
                "simulation.pybullet_gui_process",
                "--port",
                str(self._port),
            ]
            env = os.environ.copy()
            # macOS 下 fork 初始化 Cocoa 可能导致崩溃，启用安全开关
            env.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
            # 确保子进程能 import 项目包
            env["PYTHONPATH"] = self._project_root + os.pathsep + env.get("PYTHONPATH", "")
            self._process = subprocess.Popen(
                cmd,
                env=env,
                cwd=self._project_root,
                stdout=self._log_fp,
                stderr=self._log_fp,
            )

        # 等待服务就绪
        self._wait_for_server()
        config_dict = self._normalize_config(self._config)
        if config_dict:
            self._send({"cmd": "update_config", "config": config_dict}, timeout=8.0)
        self._send({"cmd": "start"}, timeout=8.0)
        self.running = True

    def stop(self) -> None:
        """停止仿真但不退出进程"""
        try:
            self._send({"cmd": "stop"})
        except Exception:
            pass
        self.running = False

    def shutdown(self) -> None:
        """关闭子进程"""
        try:
            self._send({"cmd": "shutdown"})
        except Exception:
            pass
        if self._process and self._process.poll() is None:
            self._process.terminate()
        self._process = None
        if self._log_fp:
            try:
                self._log_fp.close()
            except Exception:
                pass
            self._log_fp = None
        self.running = False

    def pause(self) -> None:
        self._send({"cmd": "pause"})

    def resume(self) -> None:
        self._send({"cmd": "resume"})

    def reset(self) -> None:
        self._send({"cmd": "reset"})

    def step(self) -> None:
        self._send({"cmd": "step"})

    def get_state(self) -> Optional[Dict[str, Any]]:
        """拉取当前仿真状态"""
        resp = self._send({"cmd": "get_state"})
        if isinstance(resp, dict) and resp.get("ok") is False:
            return None
        return resp

    def set_joint_targets(self, targets) -> None:
        self._send({"cmd": "set_joint_targets", "targets": list(targets)})

    def _send(self, payload: Dict[str, Any], timeout: float = 2.0) -> Dict[str, Any]:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n"
        with socket.create_connection(("127.0.0.1", self._port), timeout=timeout) as sock:
            sock.settimeout(timeout)
            sock.sendall(data)
            with sock.makefile("r") as fp:
                line = fp.readline()
                if not line:
                    return {"ok": False, "error": "empty response"}
                return json.loads(line)

    def _wait_for_server(self, timeout: float = 5.0) -> None:
        """等待子进程监听端口"""
        start = time.time()
        while time.time() - start < timeout:
            try:
                with socket.create_connection(("127.0.0.1", self._port), timeout=0.5):
                    return
            except Exception:
                time.sleep(0.1)
        raise RuntimeError(f"PyBullet GUI process not responding, see {self._log_path}")

    @staticmethod
    def _pick_free_port() -> int:
        """挑一个本地空闲端口"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]

    @staticmethod
    def _normalize_config(config: Any) -> Dict[str, Any]:
        """把配置对象转成 dict 并强制 GUI 模式"""
        if config is None:
            return {"ENGINE": {"mode": "gui"}}
        if isinstance(config, dict):
            cfg = dict(config)
        else:
            cfg = {}
            for key in (
                "ENGINE",
                "SCENE",
                "ARM_SIMULATION",
                "GRIPPER_SIMULATION",
                "OBJECT_SIMULATION",
                "PHYSICS_ADVANCED",
                "VISUALIZATION",
            ):
                if hasattr(config, key):
                    value = getattr(config, key)
                    if isinstance(value, dict):
                        cfg[key] = dict(value)
        engine = cfg.setdefault("ENGINE", {})
        engine["mode"] = "gui"
        return cfg
