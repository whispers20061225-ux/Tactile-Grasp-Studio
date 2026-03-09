from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from typing import Dict, Optional

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image


@dataclass
class StreamState:
    received: bool = False
    total_count: int = 0
    first_stamp: Optional[float] = None
    last_stamp: Optional[float] = None
    sample_count: int = 0
    sample_first_stamp: Optional[float] = None
    sample_last_stamp: Optional[float] = None

    def touch(self, now: float, sample_active: bool) -> None:
        if not self.received:
            self.received = True
            self.first_stamp = now
        self.total_count += 1
        self.last_stamp = now
        if sample_active:
            if self.sample_count == 0:
                self.sample_first_stamp = now
            self.sample_count += 1
            self.sample_last_stamp = now

    def reset_sample(self) -> None:
        self.sample_count = 0
        self.sample_first_stamp = None
        self.sample_last_stamp = None

    def sample_hz(self, sample_sec: float) -> Optional[float]:
        if sample_sec <= 0:
            return None
        if self.sample_count <= 0:
            return 0.0
        if self.sample_count >= 2 and self.sample_first_stamp is not None and self.sample_last_stamp is not None:
            duration = self.sample_last_stamp - self.sample_first_stamp
            if duration > 0:
                return float(self.sample_count - 1) / duration
        return float(self.sample_count) / sample_sec


class RealsenseStreamProbe(Node):
    def __init__(self, color_topic: str, depth_topic: str, info_topic: str) -> None:
        super().__init__("programme_realsense_probe")
        self.color_topic = color_topic
        self.depth_topic = depth_topic
        self.info_topic = info_topic
        self.sample_active = False
        self.streams: Dict[str, StreamState] = {
            "color": StreamState(),
            "depth": StreamState(),
            "camera_info": StreamState(),
        }
        qos_sensor = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.create_subscription(Image, color_topic, lambda _: self._on_stream("color"), qos_sensor)
        self.create_subscription(Image, depth_topic, lambda _: self._on_stream("depth"), qos_sensor)
        self.create_subscription(CameraInfo, info_topic, lambda _: self._on_stream("camera_info"), qos_sensor)

    def _on_stream(self, stream_name: str) -> None:
        self.streams[stream_name].touch(time.monotonic(), self.sample_active)

    def all_required_received(self) -> bool:
        return all(self.streams[name].received for name in ("color", "depth", "camera_info"))

    def begin_sample(self) -> None:
        for state in self.streams.values():
            state.reset_sample()
        self.sample_active = True

    def end_sample(self) -> None:
        self.sample_active = False

    def diagnostics(self) -> Dict[str, object]:
        topics = []
        try:
            for name, types in self.get_topic_names_and_types(no_demangle=True):
                if "camera/camera" in name:
                    topics.append({"topic": name, "types": list(types)})
        except Exception:
            topics = []

        publishers: Dict[str, object] = {}
        for label, topic in (
            ("color", self.color_topic),
            ("depth", self.depth_topic),
            ("camera_info", self.info_topic),
        ):
            entries = []
            try:
                infos = self.get_publishers_info_by_topic(topic)
            except Exception:
                infos = []
            for info in infos:
                try:
                    reliability = info.qos_profile.reliability.name
                except Exception:
                    reliability = str(getattr(info.qos_profile, "reliability", "unknown"))
                entries.append(
                    {
                        "node_name": getattr(info, "node_name", ""),
                        "node_namespace": getattr(info, "node_namespace", ""),
                        "topic_type": getattr(info, "topic_type", ""),
                        "reliability": reliability,
                    }
                )
            publishers[label] = entries
        return {"topics": topics, "publishers": publishers}

    def to_result(self, *, success: bool, reason: str, sample_sec: float, elapsed_sec: float) -> Dict[str, object]:
        return {
            "success": success,
            "reason": reason,
            "elapsed_sec": round(elapsed_sec, 3),
            "sample_sec": sample_sec,
            "streams": {
                name: {
                    "received": state.received,
                    "total_count": state.total_count,
                    "sample_count": state.sample_count,
                    "sample_hz": state.sample_hz(sample_sec),
                    "last_age_ms": None if state.last_stamp is None else round((time.monotonic() - state.last_stamp) * 1000.0, 1),
                }
                for name, state in self.streams.items()
            },
            "diagnostics": self.diagnostics(),
        }


def spin_until(executor: SingleThreadedExecutor, predicate, timeout_sec: float) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if predicate():
            return True
        executor.spin_once(timeout_sec=0.1)
    return predicate()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe RealSense ROS2 streams without ros2 CLI")
    parser.add_argument("--color-topic", default="/camera/camera/color/image_raw")
    parser.add_argument("--depth-topic", default="/camera/camera/aligned_depth_to_color/image_raw")
    parser.add_argument("--info-topic", default="/camera/camera/color/camera_info")
    parser.add_argument("--first-timeout-sec", type=float, default=20.0)
    parser.add_argument("--sample-sec", type=float, default=0.0)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    started_at = time.monotonic()
    rclpy.init(args=None)
    node = RealsenseStreamProbe(args.color_topic, args.depth_topic, args.info_topic)
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    exit_code = 1
    try:
        ready = spin_until(executor, node.all_required_received, max(0.1, float(args.first_timeout_sec)))
        if not ready:
            print(json.dumps(node.to_result(success=False, reason="timeout_waiting_first_messages", sample_sec=float(args.sample_sec), elapsed_sec=time.monotonic() - started_at), ensure_ascii=True))
            return 2

        sample_sec = max(0.0, float(args.sample_sec))
        if sample_sec > 0:
            node.begin_sample()
            spin_until(executor, lambda: False, sample_sec)
            node.end_sample()

        print(json.dumps(node.to_result(success=True, reason="ok", sample_sec=sample_sec, elapsed_sec=time.monotonic() - started_at), ensure_ascii=True))
        exit_code = 0
        return exit_code
    finally:
        executor.remove_node(node)
        node.destroy_node()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
