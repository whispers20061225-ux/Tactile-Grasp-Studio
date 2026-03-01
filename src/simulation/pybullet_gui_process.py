"""
独立进程里的 PyBullet GUI 服务。

用途：
- PyQt 主程序不再直接加载 PyBullet GUI，避免 macOS 下 GUI 事件循环冲突。
- 该进程启动 PyBullet 自带窗口，并通过本地 socket 接收控制指令。
"""

from __future__ import annotations

import argparse
import copy
import json
import queue
import socketserver
import threading
import time
from typing import Any, Dict, Optional

from simulation.simulator import Simulator


class _GuiController:
    """封装 PyBullet 仿真控制（GUI 相关调用必须在主线程）"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._lock = threading.Lock()
        self._running = True
        self._paused = False
        self._started = False
        self._config = config or self._load_default_config()
        self._force_gui_mode(self._config)
        self._simulator = Simulator(self._config)

    def start(self) -> Dict[str, Any]:
        with self._lock:
            if not self._started:
                self._simulator.start()
                self._started = True
        return {"ok": True}

    def stop(self) -> Dict[str, Any]:
        with self._lock:
            if self._started:
                self._simulator.stop()
                self._started = False
        return {"ok": True}

    def shutdown(self) -> Dict[str, Any]:
        self._running = False
        # 尝试停止仿真，关闭窗口
        self.stop()
        return {"ok": True}

    def pause(self) -> Dict[str, Any]:
        self._paused = True
        return {"ok": True}

    def resume(self) -> Dict[str, Any]:
        self._paused = False
        return {"ok": True}

    def reset(self) -> Dict[str, Any]:
        with self._lock:
            self._simulator.reset()
        return {"ok": True}

    def step(self) -> Dict[str, Any]:
        with self._lock:
            # 如果连接已断开，停止步进，避免重复异常
            if not self._simulator.is_connected():
                self._started = False
                return {"ok": False, "error": "not_connected"}
            try:
                self._simulator.step()
            except Exception as exc:
                # 捕获 PyBullet 断连等异常，避免进程直接退出
                self._simulator.stop()
                self._started = False
                return {"ok": False, "error": str(exc)}
        return {"ok": True}

    def get_state(self) -> Dict[str, Any]:
        with self._lock:
            return self._simulator.get_state()

    def update_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(config, dict):
            return {"ok": False, "error": "config must be dict"}
        self._force_gui_mode(config)
        with self._lock:
            was_started = self._started
            if was_started:
                self._simulator.stop()
                self._started = False
            self._config = config
            self._simulator.update_config(config)
            if was_started:
                self._simulator.start()
                self._started = True
        return {"ok": True}

    def set_joint_targets(self, targets) -> Dict[str, Any]:
        with self._lock:
            self._simulator.set_joint_targets(list(targets))
        return {"ok": True}

    @staticmethod
    def _force_gui_mode(config: Dict[str, Any]) -> None:
        engine = config.setdefault("ENGINE", {})
        engine["mode"] = "gui"

    @staticmethod
    def _load_default_config() -> Dict[str, Any]:
        """加载默认仿真配置为 dict"""
        try:
            from config.simulation_config import SimulationConfig
            cfg = SimulationConfig()
        except Exception:
            return {"ENGINE": {"mode": "gui"}}

        return {
            "ENGINE": copy.deepcopy(cfg.ENGINE),
            "SCENE": copy.deepcopy(cfg.SCENE),
            "ARM_SIMULATION": copy.deepcopy(cfg.ARM_SIMULATION),
            "GRIPPER_SIMULATION": copy.deepcopy(cfg.GRIPPER_SIMULATION),
            "OBJECT_SIMULATION": copy.deepcopy(cfg.OBJECT_SIMULATION),
            "PHYSICS_ADVANCED": copy.deepcopy(cfg.PHYSICS_ADVANCED),
            "VISUALIZATION": copy.deepcopy(cfg.VISUALIZATION),
        }


class _CommandHandler(socketserver.StreamRequestHandler):
    """处理一条条 JSON 指令"""

    def handle(self):
        server = self.server
        for raw in self.rfile:
            try:
                payload = json.loads(raw.decode("utf-8").strip())
            except Exception as exc:
                resp = {"ok": False, "error": str(exc)}
                self._send(resp)
                continue

            request = _QueuedRequest(payload)
            server.command_queue.put(request)
            try:
                resp = request.response.get(timeout=server.response_timeout)
            except queue.Empty:
                resp = {"ok": False, "error": "timeout"}
            self._send(resp)
            if server.stop_requested:
                return

    def _send(self, payload: Dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n"
        self.wfile.write(data)


class _SingleThreadServer(socketserver.TCPServer):
    allow_reuse_address = True


class _QueuedRequest:
    __slots__ = ("payload", "response")

    def __init__(self, payload: Dict[str, Any]) -> None:
        self.payload = payload
        self.response: queue.Queue = queue.Queue(maxsize=1)


def _dispatch_command(
    controller: _GuiController,
    payload: Dict[str, Any],
    server: _SingleThreadServer,
) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {"ok": False, "error": "payload must be dict"}
    cmd = payload.get("cmd", "")
    try:
        if cmd == "start":
            return controller.start()
        if cmd == "stop":
            return controller.stop()
        if cmd == "pause":
            return controller.pause()
        if cmd == "resume":
            return controller.resume()
        if cmd == "reset":
            return controller.reset()
        if cmd == "step":
            return controller.step()
        if cmd == "get_state":
            return controller.get_state()
        if cmd == "update_config":
            return controller.update_config(payload.get("config", {}))
        if cmd == "set_joint_targets":
            return controller.set_joint_targets(payload.get("targets", []))
        if cmd == "shutdown":
            resp = controller.shutdown()
            server.stop_requested = True
            return resp
        return {"ok": False, "error": f"unknown cmd: {cmd}"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser(description="PyBullet GUI process")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--config-json", type=str, default="")
    args = parser.parse_args()

    config = None
    if args.config_json:
        try:
            config = json.loads(args.config_json)
        except Exception:
            config = None

    controller = _GuiController(config)

    # macOS 要求 NSWindow 必须在主线程创建。
    # 通过命令队列把 PyBullet 调用集中到主线程执行。
    server = _SingleThreadServer(("127.0.0.1", args.port), _CommandHandler)
    server.controller = controller
    server.stop_requested = False
    server.command_queue = queue.Queue()
    server.response_timeout = 8.0

    server_thread = threading.Thread(
        target=server.serve_forever,
        kwargs={"poll_interval": 0.01},
        daemon=True,
    )
    server_thread.start()

    try:
        while not server.stop_requested:
            while True:
                try:
                    request = server.command_queue.get_nowait()
                except queue.Empty:
                    break
                resp = _dispatch_command(controller, request.payload, server)
                try:
                    request.response.put_nowait(resp)
                except queue.Full:
                    pass
            if controller._started and not controller._paused:
                controller.step()
            # 适当让出 CPU，避免满负载
            time.sleep(0.001)
    finally:
        controller.shutdown()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
