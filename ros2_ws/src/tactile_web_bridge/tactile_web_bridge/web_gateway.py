import asyncio
import copy
import hashlib
import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import cv2
import numpy as np
import requests
import rclpy
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
try:
    from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
except Exception:  # noqa: BLE001
    PackageNotFoundError = RuntimeError
    get_package_share_directory = None
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger
from tactile_interfaces.msg import (
    ArmState,
    DetectionResult,
    GraspProposalArray,
    SemanticTask,
    SystemHealth,
    TactileRaw,
)


def _safe_json_loads(raw_text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_text)
    except Exception:
        return {"raw": raw_text}
    if isinstance(parsed, dict):
        return parsed
    return {"value": parsed}


def _compact_json(payload: Any) -> str:
    return json.dumps(_json_compatible(payload), ensure_ascii=False, separators=(",", ":"))


def _extract_model_message_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise ValueError("response missing choices")

    first_choice = choices[0] or {}
    message = first_choice.get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                text_parts.append(text.strip())
        if text_parts:
            return "\n".join(text_parts)

    fallback_text = first_choice.get("text")
    if isinstance(fallback_text, str):
        return fallback_text.strip()
    raise ValueError("response missing textual content")


def _extract_first_json_object(text: str) -> dict[str, Any]:
    candidate = str(text or "").strip()
    if not candidate:
        raise ValueError("empty model response")
    decoder = json.JSONDecoder()
    for index, char in enumerate(candidate):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(candidate[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("response does not contain a JSON object")


def _json_compatible(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, deque)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_compatible(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _now_sec() -> float:
    return time.time()


def _msg_stamp_sec(msg: Any) -> float:
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return 0.0
    return float(getattr(stamp, "sec", 0)) + float(getattr(stamp, "nanosec", 0)) / 1e9


def _point_to_dict(point: Any) -> dict[str, float]:
    return {
        "x": float(getattr(point, "x", 0.0)),
        "y": float(getattr(point, "y", 0.0)),
        "z": float(getattr(point, "z", 0.0)),
    }


def _vector_to_dict(vector: Any) -> dict[str, float]:
    return {
        "x": float(getattr(vector, "x", 0.0)),
        "y": float(getattr(vector, "y", 0.0)),
        "z": float(getattr(vector, "z", 0.0)),
    }


def _pose_to_dict(pose: Any) -> dict[str, Any]:
    return {
        "position": _point_to_dict(getattr(pose, "position", None)),
        "orientation": {
            "x": float(getattr(getattr(pose, "orientation", None), "x", 0.0)),
            "y": float(getattr(getattr(pose, "orientation", None), "y", 0.0)),
            "z": float(getattr(getattr(pose, "orientation", None), "z", 0.0)),
            "w": float(getattr(getattr(pose, "orientation", None), "w", 1.0)),
        },
    }


def _semantic_task_to_dict(msg: SemanticTask) -> dict[str, Any]:
    constraints = [str(item) for item in msg.constraints]
    return {
        "task": str(msg.task or "pick"),
        "target_label": str(msg.target_label or ""),
        "target_hint": str(msg.target_hint or ""),
        "constraints": constraints,
        "gripper": constraints[0] if constraints else "",
        "excluded_labels": [str(item) for item in msg.excluded_labels],
        "confidence": float(msg.confidence),
        "need_human_confirm": bool(msg.need_human_confirm),
        "reason": str(msg.reason or ""),
        "prompt_text": str(msg.prompt_text or ""),
        "raw_json": str(msg.raw_json or ""),
        "updated_at": _now_sec(),
        "stamp_sec": _msg_stamp_sec(msg),
    }


def _semantic_payload_defaults(semantic: dict[str, Any]) -> dict[str, Any]:
    constraints = [
        str(item).strip()
        for item in list(semantic.get("constraints", []) or [])
        if str(item).strip()
    ]
    return {
        "task": str(semantic.get("task", "pick") or "pick"),
        "target_label": str(semantic.get("target_label", "") or ""),
        "target_hint": str(semantic.get("target_hint", "") or ""),
        "constraints": constraints,
        "excluded_labels": [
            str(item).strip()
            for item in list(semantic.get("excluded_labels", []) or [])
            if str(item).strip()
        ],
        "confidence": float(semantic.get("confidence", 1.0) or 1.0),
        "need_human_confirm": bool(semantic.get("need_human_confirm", False)),
        "reason": str(semantic.get("reason", "") or ""),
    }


def _detection_result_to_dict(msg: DetectionResult) -> dict[str, Any]:
    bbox = None
    if msg.has_bbox:
        bbox = {
            "x_offset": int(msg.bbox.x_offset),
            "y_offset": int(msg.bbox.y_offset),
            "width": int(msg.bbox.width),
            "height": int(msg.bbox.height),
            "xyxy": [
                int(msg.bbox.x_offset),
                int(msg.bbox.y_offset),
                int(msg.bbox.x_offset + msg.bbox.width),
                int(msg.bbox.y_offset + msg.bbox.height),
            ],
        }
    return {
        "backend": str(msg.backend or ""),
        "accepted": bool(msg.accepted),
        "candidate_visible": bool(msg.candidate_visible),
        "candidate_complete": bool(msg.candidate_complete),
        "task": str(msg.task or ""),
        "target_label": str(msg.target_label or ""),
        "confidence": float(msg.confidence),
        "need_human_confirm": bool(msg.need_human_confirm),
        "reason": str(msg.reason or ""),
        "image_width": int(msg.image_width),
        "image_height": int(msg.image_height),
        "bbox": bbox,
        "point_px": list(msg.point_px) if msg.has_point else None,
        "has_mask": bool(msg.has_mask),
        "updated_at": _now_sec(),
        "stamp_sec": _msg_stamp_sec(msg),
    }


def _proposal_to_dict(proposal: Any) -> dict[str, Any]:
    return {
        "contact_point_1": _point_to_dict(proposal.contact_point_1),
        "contact_point_2": _point_to_dict(proposal.contact_point_2),
        "grasp_center": _point_to_dict(proposal.grasp_center),
        "approach_direction": _vector_to_dict(proposal.approach_direction),
        "closing_direction": _vector_to_dict(proposal.closing_direction),
        "grasp_pose": _pose_to_dict(proposal.grasp_pose),
        "pregrasp_pose": _pose_to_dict(proposal.pregrasp_pose),
        "grasp_width_m": float(proposal.grasp_width_m),
        "confidence_score": float(proposal.confidence_score),
        "semantic_score": float(proposal.semantic_score),
        "candidate_rank": int(proposal.candidate_rank),
        "task_constraint_tag": str(proposal.task_constraint_tag or ""),
    }


def _arm_state_to_dict(msg: ArmState) -> dict[str, Any]:
    return {
        "connected": bool(msg.connected),
        "moving": bool(msg.moving),
        "error": bool(msg.error),
        "error_message": str(msg.error_message or ""),
        "battery_voltage": float(msg.battery_voltage),
        "joint_positions": [float(item) for item in msg.joint_positions],
        "joint_angles": [float(item) for item in msg.joint_angles],
        "updated_at": _now_sec(),
        "stamp_sec": _msg_stamp_sec(msg),
    }


@dataclass
class StreamFrame:
    jpeg: Optional[bytes] = None
    updated_at: float = 0.0
    width: int = 0
    height: int = 0
    encoding: str = ""
    source_stamp_sec: float = 0.0


@dataclass
class RawImageFrame:
    data: bytes
    width: int
    height: int
    step: int
    encoding: str
    source_stamp_sec: float


def _frame_rate_from_times(update_times: list[float]) -> float:
    if len(update_times) < 2:
        return 0.0
    span = max(1e-6, update_times[-1] - update_times[0])
    return float(len(update_times) - 1) / span


def _image_msg_to_raw_frame(msg: Image) -> RawImageFrame:
    return RawImageFrame(
        data=bytes(msg.data),
        width=int(msg.width),
        height=int(msg.height),
        step=int(getattr(msg, "step", 0) or 0),
        encoding=str(msg.encoding or ""),
        source_stamp_sec=_msg_stamp_sec(msg),
    )


def _raw_image_to_jpeg(frame: RawImageFrame, jpeg_quality: int) -> bytes:
    width = int(frame.width)
    height = int(frame.height)
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image dimensions: {width}x{height}")

    encoding = str(frame.encoding or "")
    row_bytes = 0
    channels = 0
    convert_code: Optional[int] = None
    grayscale = False

    if encoding in ("rgb8", "bgr8"):
        channels = 3
        row_bytes = width * channels
        convert_code = cv2.COLOR_RGB2BGR if encoding == "rgb8" else None
    elif encoding in ("rgba8", "bgra8"):
        channels = 4
        row_bytes = width * channels
        convert_code = cv2.COLOR_RGBA2BGR if encoding == "rgba8" else cv2.COLOR_BGRA2BGR
    elif encoding in ("mono8", "8UC1"):
        channels = 1
        row_bytes = width
        grayscale = True
    else:
        raise ValueError(f"unsupported image encoding for MJPEG stream: {encoding}")

    step = int(frame.step or row_bytes)
    if step < row_bytes:
        raise ValueError(f"image step {step} is smaller than row bytes {row_bytes}")
    required_bytes = step * height
    if len(frame.data) < required_bytes:
        raise ValueError(
            f"image buffer too small: got {len(frame.data)} expected at least {required_bytes}"
        )

    flat = np.frombuffer(frame.data, dtype=np.uint8, count=required_bytes)
    rows = flat.reshape((height, step))
    pixels = rows[:, :row_bytes]
    if step != row_bytes:
        pixels = pixels.copy()
    if grayscale:
        array = cv2.cvtColor(pixels.reshape((height, width)), cv2.COLOR_GRAY2BGR)
    else:
        array = pixels.reshape((height, width, channels))
        if convert_code is not None:
            array = cv2.cvtColor(array, convert_code)

    ok, encoded = cv2.imencode(
        ".jpg",
        array,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
    )
    if not ok:
        raise ValueError("cv2.imencode returned false")
    return encoded.tobytes()


class MjpegFrameBuffer:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._frame = StreamFrame()
        self._frame_count = 0
        self._changed_frame_count = 0
        self._last_digest = ""
        self._update_times: deque[float] = deque(maxlen=180)

    def update(
        self,
        jpeg: bytes,
        *,
        width: int,
        height: int,
        encoding: str,
        source_stamp_sec: float,
    ) -> None:
        with self._condition:
            now_sec = _now_sec()
            digest = hashlib.sha1(jpeg).hexdigest()
            self._frame_count += 1
            if digest != self._last_digest:
                self._changed_frame_count += 1
                self._last_digest = digest
            self._update_times.append(now_sec)
            self._frame = StreamFrame(
                jpeg=jpeg,
                updated_at=now_sec,
                width=int(width),
                height=int(height),
                encoding=str(encoding or ""),
                source_stamp_sec=float(source_stamp_sec),
            )
            self._condition.notify_all()

    def wait_next(self, previous_update: float, timeout_sec: float = 1.0) -> StreamFrame:
        with self._condition:
            if self._frame.updated_at <= previous_update:
                self._condition.wait(timeout_sec)
            return self._frame

    def latest(self) -> StreamFrame:
        with self._condition:
            return self._frame

    def stats(self) -> dict[str, Any]:
        with self._condition:
            frame = self._frame
            update_times = list(self._update_times)
            frame_count = int(self._frame_count)
            changed_frame_count = int(self._changed_frame_count)
        fps = _frame_rate_from_times(update_times)
        changed_fps = 0.0
        if update_times:
            span = max(1e-6, update_times[-1] - update_times[0]) if len(update_times) >= 2 else 1.0
            changed_fps = float(min(changed_frame_count, len(update_times))) / span
        now_sec = _now_sec()
        return {
            "updated_at": float(frame.updated_at),
            "age_sec": max(0.0, now_sec - float(frame.updated_at)) if frame.updated_at else None,
            "frame_count": frame_count,
            "changed_frame_count": changed_frame_count,
            "approx_fps": fps,
            "approx_changed_fps": changed_fps,
            "width": int(frame.width),
            "height": int(frame.height),
            "encoding": str(frame.encoding or ""),
            "jpeg_bytes": len(frame.jpeg) if frame.jpeg is not None else 0,
            "source_stamp_sec": float(frame.source_stamp_sec),
            "source_age_sec": max(0.0, now_sec - float(frame.source_stamp_sec))
            if frame.source_stamp_sec
            else None,
        }


class AsyncImageEncoder:
    def __init__(
        self,
        *,
        stream_name: str,
        logger: Any,
        jpeg_quality: int,
        buffer: MjpegFrameBuffer,
    ) -> None:
        self._stream_name = str(stream_name)
        self._logger = logger
        self._jpeg_quality = int(jpeg_quality)
        self._buffer = buffer
        self._condition = threading.Condition()
        self._pending: Optional[RawImageFrame] = None
        self._stopped = False
        self._received_count = 0
        self._dropped_pending_count = 0
        self._encoded_count = 0
        self._receive_times: deque[float] = deque(maxlen=180)
        self._thread = threading.Thread(
            target=self._run,
            name=f"{self._stream_name}-jpeg-encoder",
            daemon=True,
        )
        self._thread.start()

    def submit(self, msg: Image) -> None:
        frame = _image_msg_to_raw_frame(msg)
        with self._condition:
            self._received_count += 1
            self._receive_times.append(_now_sec())
            if self._pending is not None:
                self._dropped_pending_count += 1
            self._pending = frame
            self._condition.notify()

    def close(self) -> None:
        with self._condition:
            self._stopped = True
            self._condition.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def stats(self) -> dict[str, Any]:
        base_stats = self._buffer.stats()
        with self._condition:
            receive_times = list(self._receive_times)
            received_count = int(self._received_count)
            dropped_pending_count = int(self._dropped_pending_count)
            encoded_count = int(self._encoded_count)
            has_pending = self._pending is not None
        return {
            **base_stats,
            "input_frame_count": received_count,
            "input_approx_fps": _frame_rate_from_times(receive_times),
            "encoded_frame_count": encoded_count,
            "dropped_pending_count": dropped_pending_count,
            "has_pending_frame": has_pending,
        }

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._stopped and self._pending is None:
                    self._condition.wait(timeout=1.0)
                if self._stopped:
                    return
                frame = self._pending
                self._pending = None
            if frame is None:
                continue
            try:
                jpeg = _raw_image_to_jpeg(frame, self._jpeg_quality)
            except Exception as exc:  # noqa: BLE001
                self._logger.warn(
                    f"failed to encode {self._stream_name} image for MJPEG stream: {exc}"
                )
                continue
            self._buffer.update(
                jpeg,
                width=frame.width,
                height=frame.height,
                encoding=frame.encoding,
                source_stamp_sec=frame.source_stamp_sec,
            )
            with self._condition:
                self._encoded_count += 1


def _dialog_status_label(status: str) -> str:
    labels = {
        "idle": "Idle",
        "thinking": "Thinking",
        "ready": "Ready",
        "awaiting_lock": "Waiting for Target Lock",
        "auto_executing": "Auto Executing",
        "error": "Error",
    }
    return labels.get(status, status or "Idle")


def _dialog_message_summary(
    semantic_payload: Optional[dict[str, Any]],
    requested_action: str,
) -> str:
    if not semantic_payload:
        return f"Action: {requested_action or 'review'}"
    target = str(
        semantic_payload.get("target_label")
        or semantic_payload.get("target_hint")
        or "--"
    )
    constraints = list(semantic_payload.get("constraints", []) or [])
    gripper = str(constraints[0] if constraints else "--")
    return (
        f"Task: {semantic_payload.get('task', 'pick')} | "
        f"Target: {target} | "
        f"Gripper: {gripper} | "
        f"Action: {requested_action or 'update_task'}"
    )


class TactileWebGateway(Node):
    def __init__(self) -> None:
        super().__init__("tactile_web_gateway")

        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 8765)
        self.declare_parameter("frontend_dist_dir", "")
        self.declare_parameter("prompt_topic", "/qwen/user_prompt")
        self.declare_parameter("semantic_task_topic", "/qwen/semantic_task")
        self.declare_parameter("semantic_result_topic", "/qwen/semantic_result")
        self.declare_parameter("color_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("detection_result_topic", "/perception/detection_result")
        self.declare_parameter("detection_debug_topic", "/perception/detection_debug")
        self.declare_parameter(
            "detection_debug_overlay_topic", "/perception/detection_debug_overlay"
        )
        self.declare_parameter("target_locked_topic", "/sim/perception/target_locked")
        self.declare_parameter("pick_active_topic", "/sim/task/pick_active")
        self.declare_parameter("pick_status_topic", "/sim/task/pick_status")
        self.declare_parameter(
            "candidate_grasp_proposal_array_topic", "/grasp/candidate_grasp_proposals"
        )
        self.declare_parameter("grasp_backend_debug_topic", "/grasp/backend_debug")
        self.declare_parameter("grasp_overlay_topic", "/grasp/ggcnn/grasp_overlay")
        self.declare_parameter("tactile_topic", "/tactile/raw")
        self.declare_parameter("health_topic", "/system/health")
        self.declare_parameter("arm_state_topic", "/arm/state")
        self.declare_parameter("execute_pick_service", "/task/execute_pick")
        self.declare_parameter("reset_pick_session_service", "/task/reset_pick_session")
        self.declare_parameter("start_search_sweep_service", "/task/start_search_sweep")
        self.declare_parameter("event_log_capacity", 240)
        self.declare_parameter("feedback_event_capacity", 48)
        self.declare_parameter("stream_jpeg_quality", 95)
        self.declare_parameter("dialog_model_endpoint", "http://127.0.0.1:8000/v1/chat/completions")
        self.declare_parameter("dialog_model_name", "Qwen/Qwen2.5-VL-3B-Instruct-AWQ")
        self.declare_parameter("dialog_api_key", "EMPTY")
        self.declare_parameter("dialog_request_timeout_sec", 25.0)
        self.declare_parameter("dialog_temperature", 0.1)
        self.declare_parameter("dialog_max_tokens", 384)
        self.declare_parameter("dialog_history_turns", 8)
        self.declare_parameter("dialog_auto_execute_confidence_threshold", 0.55)
        self.declare_parameter("default_task", "pick")
        self.declare_parameter("default_target_hint", "blue cylinder")
        self.declare_parameter("default_constraints", ["parallel_gripper"])

        self.host = str(self.get_parameter("host").value)
        self.port = int(self.get_parameter("port").value)
        self.frontend_dist_dir = str(self.get_parameter("frontend_dist_dir").value or "").strip()
        self.prompt_topic = str(self.get_parameter("prompt_topic").value)
        self.semantic_task_topic = str(self.get_parameter("semantic_task_topic").value)
        self.semantic_result_topic = str(self.get_parameter("semantic_result_topic").value)
        self.color_topic = str(self.get_parameter("color_topic").value)
        self.detection_result_topic = str(self.get_parameter("detection_result_topic").value)
        self.detection_debug_topic = str(self.get_parameter("detection_debug_topic").value)
        self.detection_debug_overlay_topic = str(
            self.get_parameter("detection_debug_overlay_topic").value
        )
        self.target_locked_topic = str(self.get_parameter("target_locked_topic").value)
        self.pick_active_topic = str(self.get_parameter("pick_active_topic").value)
        self.pick_status_topic = str(self.get_parameter("pick_status_topic").value)
        self.candidate_grasp_proposal_array_topic = str(
            self.get_parameter("candidate_grasp_proposal_array_topic").value
        )
        self.grasp_backend_debug_topic = str(
            self.get_parameter("grasp_backend_debug_topic").value
        )
        self.grasp_overlay_topic = str(self.get_parameter("grasp_overlay_topic").value)
        self.tactile_topic = str(self.get_parameter("tactile_topic").value)
        self.health_topic = str(self.get_parameter("health_topic").value)
        self.arm_state_topic = str(self.get_parameter("arm_state_topic").value)
        self.execute_pick_service = str(self.get_parameter("execute_pick_service").value)
        self.reset_pick_session_service = str(
            self.get_parameter("reset_pick_session_service").value
        )
        self.start_search_sweep_service = str(
            self.get_parameter("start_search_sweep_service").value
        )
        self.event_log_capacity = max(50, int(self.get_parameter("event_log_capacity").value))
        self.feedback_event_capacity = max(
            10, int(self.get_parameter("feedback_event_capacity").value)
        )
        self.stream_jpeg_quality = min(
            100,
            max(70, int(self.get_parameter("stream_jpeg_quality").value)),
        )
        self.dialog_model_endpoint = str(self.get_parameter("dialog_model_endpoint").value)
        self.dialog_model_name = str(self.get_parameter("dialog_model_name").value)
        self.dialog_api_key = str(self.get_parameter("dialog_api_key").value or "")
        self.dialog_request_timeout_sec = max(
            1.0, float(self.get_parameter("dialog_request_timeout_sec").value)
        )
        self.dialog_temperature = max(
            0.0, float(self.get_parameter("dialog_temperature").value)
        )
        self.dialog_max_tokens = max(128, int(self.get_parameter("dialog_max_tokens").value))
        self.dialog_history_turns = max(
            2, int(self.get_parameter("dialog_history_turns").value)
        )
        self.dialog_auto_execute_confidence_threshold = max(
            0.0,
            min(1.0, float(self.get_parameter("dialog_auto_execute_confidence_threshold").value)),
        )
        self.default_task = str(self.get_parameter("default_task").value).strip() or "pick"
        self.default_target_hint = (
            str(self.get_parameter("default_target_hint").value).strip() or "target object"
        )
        self.default_constraints = [
            str(item).strip()
            for item in list(self.get_parameter("default_constraints").value)
            if str(item).strip()
        ]
        if not self.default_constraints:
            self.default_constraints = ["parallel_gripper"]

        qos_sensor = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        qos_reliable = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        self.prompt_pub = self.create_publisher(String, self.prompt_topic, qos_reliable)
        self.semantic_override_pub = self.create_publisher(
            SemanticTask, self.semantic_task_topic, qos_reliable
        )
        self.execute_pick_client = self.create_client(Trigger, self.execute_pick_service)
        self.reset_pick_client = self.create_client(Trigger, self.reset_pick_session_service)
        self.start_search_sweep_client = self.create_client(
            Trigger, self.start_search_sweep_service
        )

        self.create_subscription(String, self.semantic_result_topic, self._on_semantic_result, qos_reliable)
        self.create_subscription(SemanticTask, self.semantic_task_topic, self._on_semantic_task, qos_reliable)
        self.create_subscription(Image, self.color_topic, self._on_color_image, qos_sensor)
        self.create_subscription(
            DetectionResult, self.detection_result_topic, self._on_detection_result, qos_reliable
        )
        self.create_subscription(String, self.detection_debug_topic, self._on_detection_debug, qos_reliable)
        self.create_subscription(
            Image, self.detection_debug_overlay_topic, self._on_detection_overlay, qos_sensor
        )
        self.create_subscription(Bool, self.target_locked_topic, self._on_target_locked, qos_reliable)
        self.create_subscription(Bool, self.pick_active_topic, self._on_pick_active, qos_reliable)
        self.create_subscription(String, self.pick_status_topic, self._on_pick_status, qos_reliable)
        self.create_subscription(
            GraspProposalArray,
            self.candidate_grasp_proposal_array_topic,
            self._on_grasp_proposals,
            qos_reliable,
        )
        self.create_subscription(String, self.grasp_backend_debug_topic, self._on_backend_debug, qos_reliable)
        self.create_subscription(Image, self.grasp_overlay_topic, self._on_grasp_overlay, qos_sensor)
        self.create_subscription(TactileRaw, self.tactile_topic, self._on_tactile, qos_sensor)
        self.create_subscription(SystemHealth, self.health_topic, self._on_health, qos_reliable)
        self.create_subscription(ArmState, self.arm_state_topic, self._on_arm_state, qos_reliable)

        self._lock = threading.Lock()
        self._state_version = 0
        self._event_id = 0
        self._events: deque[dict[str, Any]] = deque(maxlen=self.event_log_capacity)
        self._feedback_events: deque[dict[str, Any]] = deque(maxlen=self.feedback_event_capacity)
        self._last_prompt_text = ""
        self._intervention = {"active": False, "source": "", "label": ""}
        self._semantic_result: dict[str, Any] = {"raw": "", "updated_at": 0.0}
        self._semantic_task: dict[str, Any] = {"updated_at": 0.0}
        self._detection_result: dict[str, Any] = {"updated_at": 0.0}
        self._detection_debug: dict[str, Any] = {"updated_at": 0.0}
        self._grasp_backend_debug: dict[str, Any] = {"updated_at": 0.0}
        self._pick_status: dict[str, Any] = {"phase": "idle", "message": "idle", "updated_at": 0.0}
        self._grasp_proposals: dict[str, Any] = {
            "updated_at": 0.0,
            "proposals": [],
            "selected_index": 0,
            "count": 0,
        }
        self._target_locked = False
        self._pick_active = False
        self._tactile: dict[str, Any] = {"updated_at": 0.0}
        self._arm_state: dict[str, Any] = {"updated_at": 0.0}
        self._health_by_node: dict[str, dict[str, Any]] = {}
        self._dialog_message_id = 0
        self._dialog = {
            "session_id": uuid4().hex,
            "mode": "review",
            "status": "idle",
            "pending_auto_execute": False,
            "last_error": "",
            "messages": [],
            "updated_at": 0.0,
        }
        self._dialog_http = requests.Session()

        self.rgb_stream = MjpegFrameBuffer()
        self.detection_overlay_stream = MjpegFrameBuffer()
        self.grasp_overlay_stream = MjpegFrameBuffer()
        self.rgb_encoder = AsyncImageEncoder(
            stream_name="rgb",
            logger=self.get_logger(),
            jpeg_quality=self.stream_jpeg_quality,
            buffer=self.rgb_stream,
        )
        self.detection_overlay_encoder = AsyncImageEncoder(
            stream_name="detection_overlay",
            logger=self.get_logger(),
            jpeg_quality=self.stream_jpeg_quality,
            buffer=self.detection_overlay_stream,
        )
        self.grasp_overlay_encoder = AsyncImageEncoder(
            stream_name="grasp_overlay",
            logger=self.get_logger(),
            jpeg_quality=self.stream_jpeg_quality,
            buffer=self.grasp_overlay_stream,
        )

        self._last_pick_status_phase = "idle"
        self._last_pick_active = False
        self._last_target_locked = False

        self._record_event(
            "system",
            "info",
            f"web gateway ready on http://{self.host}:{self.port}",
            feedback=False,
        )
        self.get_logger().info(
            f"tactile_web_gateway ready: host={self.host} port={self.port} "
            f"prompt={self.prompt_topic} semantic_task={self.semantic_task_topic}"
        )

    def destroy_node(self) -> bool:
        self.rgb_encoder.close()
        self.detection_overlay_encoder.close()
        self.grasp_overlay_encoder.close()
        self._dialog_http.close()
        return super().destroy_node()

    def _record_event(
        self,
        category: str,
        level: str,
        message: str,
        *,
        data: Optional[dict[str, Any]] = None,
        feedback: bool = False,
    ) -> dict[str, Any]:
        event = {
            "id": self._event_id + 1,
            "created_at": _now_sec(),
            "category": category,
            "level": level,
            "message": message,
            "data": data or {},
        }
        with self._lock:
            self._event_id = int(event["id"])
            self._events.append(event)
            if feedback:
                self._feedback_events.append(event)
            self._state_version += 1
        return event

    def _set_intervention(self, active: bool, source: str = "", label: str = "") -> None:
        with self._lock:
            self._intervention = {
                "active": bool(active),
                "source": str(source or ""),
                "label": str(label or ""),
            }
            self._state_version += 1

    def _on_semantic_result(self, msg: String) -> None:
        parsed = _safe_json_loads(str(msg.data or ""))
        parsed["raw"] = str(msg.data or "")
        parsed["updated_at"] = _now_sec()
        with self._lock:
            self._semantic_result = parsed
            self._state_version += 1

    def _on_semantic_task(self, msg: SemanticTask) -> None:
        semantic = _semantic_task_to_dict(msg)
        prompt_text = str(semantic.get("prompt_text", "") or "").strip()
        if prompt_text:
            self._last_prompt_text = prompt_text
        with self._lock:
            self._semantic_task = semantic
            self._state_version += 1

    def _on_color_image(self, msg: Image) -> None:
        self.rgb_encoder.submit(msg)

    def _on_detection_result(self, msg: DetectionResult) -> None:
        detection = _detection_result_to_dict(msg)
        with self._lock:
            self._detection_result = detection
            self._state_version += 1

    def _on_detection_debug(self, msg: String) -> None:
        parsed = _safe_json_loads(str(msg.data or ""))
        parsed["updated_at"] = _now_sec()
        with self._lock:
            self._detection_debug = parsed
            self._state_version += 1

    def _on_detection_overlay(self, msg: Image) -> None:
        self.detection_overlay_encoder.submit(msg)

    def _on_target_locked(self, msg: Bool) -> None:
        locked = bool(msg.data)
        if locked != self._last_target_locked:
            self._last_target_locked = locked
            self._record_event(
                "execution",
                "info",
                "target locked" if locked else "target lock released",
                feedback=False,
            )
        with self._lock:
            self._target_locked = locked
            self._state_version += 1
        self._maybe_run_pending_auto_execute(locked)

    def _on_pick_active(self, msg: Bool) -> None:
        active = bool(msg.data)
        if active != self._last_pick_active:
            self._last_pick_active = active
            self._record_event(
                "execution",
                "info",
                "pick_active became true" if active else "pick_active returned false",
                feedback=False,
            )
        with self._lock:
            self._pick_active = active
            self._state_version += 1

    def _on_pick_status(self, msg: String) -> None:
        parsed = _safe_json_loads(str(msg.data or ""))
        parsed["raw"] = str(msg.data or "")
        parsed["updated_at"] = _now_sec()
        phase = str(parsed.get("phase", "idle") or "idle")
        phase_changed = phase != self._last_pick_status_phase
        if phase_changed:
            self._last_pick_status_phase = phase
            self._record_event(
                "execution",
                "error" if phase == "error" else "info",
                str(parsed.get("message") or phase),
                data={"phase": phase},
                feedback=phase in {"completed", "error"},
            )
        with self._lock:
            self._pick_status = parsed
            self._state_version += 1
        if phase_changed and phase == "completed":
            self._append_dialog_message("system", "Execution completed.")
            self._set_dialog_status(status="ready", pending_auto_execute=False, last_error="")
        elif phase_changed and phase == "error":
            self._append_dialog_message(
                "system",
                f"Execution error: {str(parsed.get('message') or 'unknown error')}",
            )
            self._set_dialog_status(
                status="error",
                pending_auto_execute=False,
                last_error=str(parsed.get("message") or "execution error"),
            )

    def _on_grasp_proposals(self, msg: GraspProposalArray) -> None:
        proposals = [_proposal_to_dict(item) for item in list(msg.proposals)[:8]]
        with self._lock:
            self._grasp_proposals = {
                "proposals": proposals,
                "selected_index": int(msg.selected_index),
                "count": len(msg.proposals),
                "updated_at": _now_sec(),
                "stamp_sec": _msg_stamp_sec(msg),
            }
            self._state_version += 1

    def _on_backend_debug(self, msg: String) -> None:
        parsed = _safe_json_loads(str(msg.data or ""))
        parsed["raw"] = str(msg.data or "")
        parsed["updated_at"] = _now_sec()
        with self._lock:
            self._grasp_backend_debug = parsed
            self._state_version += 1

    def _on_grasp_overlay(self, msg: Image) -> None:
        self.grasp_overlay_encoder.submit(msg)

    def _on_tactile(self, msg: TactileRaw) -> None:
        tactile = {
            "rows": int(msg.rows),
            "cols": int(msg.cols),
            "sequence_id": int(msg.sequence_id),
            "frame_id": str(msg.frame_id or ""),
            "forces": [float(item) for item in msg.forces],
            "forces_fx": [float(item) for item in msg.forces_fx],
            "forces_fy": [float(item) for item in msg.forces_fy],
            "forces_fz": [float(item) for item in msg.forces_fz],
            "updated_at": _now_sec(),
            "stamp_sec": _msg_stamp_sec(msg),
        }
        with self._lock:
            self._tactile = tactile
            self._state_version += 1

    def _on_health(self, msg: SystemHealth) -> None:
        item = {
            "node_name": str(msg.node_name or ""),
            "healthy": bool(msg.healthy),
            "level": int(msg.level),
            "message": str(msg.message or ""),
            "cpu_percent": float(msg.cpu_percent),
            "memory_percent": float(msg.memory_percent),
            "updated_at": _now_sec(),
            "stamp_sec": _msg_stamp_sec(msg),
        }
        with self._lock:
            self._health_by_node[item["node_name"] or f"node-{len(self._health_by_node) + 1}"] = item
            self._state_version += 1
        if not item["healthy"] or item["level"] >= 2:
            self._record_event(
                "health",
                "error" if item["level"] >= 2 else "warn",
                f"{item['node_name']}: {item['message']}",
                data={"level": item["level"]},
                feedback=item["level"] >= 2,
            )

    def _on_arm_state(self, msg: ArmState) -> None:
        arm_state = _arm_state_to_dict(msg)
        with self._lock:
            self._arm_state = arm_state
            self._state_version += 1

    def _append_dialog_message(
        self,
        role: str,
        text: str,
        *,
        meta: str = "",
        semantic_task: Optional[dict[str, Any]] = None,
        requested_action: str = "",
    ) -> dict[str, Any]:
        message = {
            "id": 0,
            "role": str(role or "system"),
            "text": str(text or "").strip(),
            "meta": str(meta or ""),
            "semantic_task": copy.deepcopy(semantic_task) if semantic_task else None,
            "requested_action": str(requested_action or ""),
            "created_at": _now_sec(),
        }
        with self._lock:
            self._dialog_message_id += 1
            message["id"] = int(self._dialog_message_id)
            self._dialog["messages"].append(message)
            self._dialog["messages"] = list(self._dialog["messages"][-40:])
            self._dialog["updated_at"] = float(message["created_at"])
            self._state_version += 1
        return copy.deepcopy(message)

    def _set_dialog_status(
        self,
        *,
        status: Optional[str] = None,
        mode: Optional[str] = None,
        pending_auto_execute: Optional[bool] = None,
        last_error: Optional[str] = None,
    ) -> None:
        with self._lock:
            if mode is not None:
                self._dialog["mode"] = "auto" if str(mode).lower() == "auto" else "review"
            if status is not None:
                self._dialog["status"] = str(status or "idle")
            if pending_auto_execute is not None:
                self._dialog["pending_auto_execute"] = bool(pending_auto_execute)
            if last_error is not None:
                self._dialog["last_error"] = str(last_error or "")
            self._dialog["updated_at"] = _now_sec()
            self._state_version += 1

    def _reset_dialog_session(self) -> None:
        with self._lock:
            current_mode = str(self._dialog.get("mode", "review") or "review")
            self._dialog = {
                "session_id": uuid4().hex,
                "mode": current_mode,
                "status": "idle",
                "pending_auto_execute": False,
                "last_error": "",
                "messages": [],
                "updated_at": _now_sec(),
            }
            self._state_version += 1

    def _dialog_context_snapshot(self) -> dict[str, Any]:
        with self._lock:
            dialog = copy.deepcopy(self._dialog)
            semantic = copy.deepcopy(self._semantic_task)
            detection = copy.deepcopy(self._detection_result)
            detection_debug = copy.deepcopy(self._detection_debug)
            execution = {
                "phase": self._derive_phase(
                    semantic=self._semantic_task,
                    detection=self._detection_result,
                    detection_debug=self._detection_debug,
                    target_locked=bool(self._target_locked),
                    pick_active=bool(self._pick_active),
                    pick_status=self._pick_status,
                ),
                "target_locked": bool(self._target_locked),
                "pick_active": bool(self._pick_active),
                "pick_status": copy.deepcopy(self._pick_status),
            }
        top_candidates = []
        for item in list(detection_debug.get("top_candidates", []) or [])[:5]:
            if not isinstance(item, dict):
                continue
            top_candidates.append(
                {
                    "label": str(item.get("label", "") or ""),
                    "confidence": float(item.get("confidence", 0.0) or 0.0),
                    "status": str(item.get("status", "") or ""),
                    "bbox_xyxy": list(item.get("bbox_xyxy", []) or []),
                }
            )
        return {
            "dialog": dialog,
            "semantic": semantic,
            "detection": detection,
            "top_candidates": top_candidates,
            "execution": execution,
        }

    def _build_dialog_context_text(self, snapshot: dict[str, Any], mode: str) -> str:
        semantic = _semantic_payload_defaults(snapshot.get("semantic", {}))
        detection = snapshot.get("detection", {}) or {}
        execution = snapshot.get("execution", {}) or {}
        top_candidates = snapshot.get("top_candidates", []) or []
        context = {
            "mode": "auto" if str(mode).lower() == "auto" else "review",
            "current_task": semantic,
            "vision": {
                "accepted": bool(detection.get("accepted", False)),
                "target_label": str(detection.get("target_label", "") or ""),
                "confidence": float(detection.get("confidence", 0.0) or 0.0),
                "image_width": int(detection.get("image_width", 0) or 0),
                "image_height": int(detection.get("image_height", 0) or 0),
                "top_candidates": top_candidates,
            },
            "execution": {
                "phase": str(execution.get("phase", "idle") or "idle"),
                "target_locked": bool(execution.get("target_locked", False)),
                "pick_active": bool(execution.get("pick_active", False)),
                "pick_status_message": str(
                    (execution.get("pick_status", {}) or {}).get("message", "") or ""
                ),
            },
        }
        return (
            "Robot UI context. Use this to resolve references and keep continuity between turns. "
            + _compact_json(context)
        )

    def _dialog_history_messages(self, dialog_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        history: list[dict[str, Any]] = []
        conversational_messages = [
            item
            for item in dialog_messages
            if str(item.get("role", "")) in {"user", "assistant"}
        ]
        for item in conversational_messages[-self.dialog_history_turns * 2 :]:
            role = str(item.get("role", "user") or "user")
            text = str(item.get("text", "") or "").strip()
            semantic_task = item.get("semantic_task")
            if role == "assistant" and isinstance(semantic_task, dict):
                text = (
                    f"{text}\nStructured task snapshot: "
                    f"{_compact_json(_semantic_payload_defaults(semantic_task))}"
                ).strip()
            if not text:
                continue
            history.append({"role": role, "content": [{"type": "text", "text": text}]})
        return history

    def _query_dialog_model(
        self,
        *,
        user_text: str,
        mode: str,
        snapshot: dict[str, Any],
        history_messages: list[dict[str, Any]],
    ) -> tuple[str, dict[str, Any]]:
        system_prompt = (
            "You are the multi-turn dialog controller for a robot manipulation UI. "
            "Return exactly one JSON object with no markdown. "
            "Required keys: assistant_text, requested_action, task, target_label, target_hint, "
            "constraints, excluded_labels, confidence, need_human_confirm, reason. "
            "requested_action must be one of update_task, execute, clarify, cancel. "
            "Keep continuity with the current task unless the user explicitly changes it. "
            "If the user only changes one field, keep the others consistent. "
            "If the request is ambiguous, ask one concise clarification question and set "
            "requested_action to clarify. "
            "In review mode, do not promise automatic execution. "
            "In auto mode, you may say execution will start only when the command is clear and safe."
        )
        headers = {"Content-Type": "application/json"}
        if self.dialog_api_key:
            headers["Authorization"] = f"Bearer {self.dialog_api_key}"
        payload = {
            "model": self.dialog_model_name,
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": self._build_dialog_context_text(snapshot, mode),
                        }
                    ],
                },
                *history_messages,
                {"role": "user", "content": [{"type": "text", "text": str(user_text or "").strip()}]},
            ],
            "temperature": self.dialog_temperature,
            "max_tokens": self.dialog_max_tokens,
        }
        response = self._dialog_http.post(
            self.dialog_model_endpoint,
            headers=headers,
            json=payload,
            timeout=self.dialog_request_timeout_sec,
        )
        response.raise_for_status()
        raw_text = _extract_model_message_text(response.json())
        return raw_text, _extract_first_json_object(raw_text)

    def _normalize_dialog_response(
        self,
        response: dict[str, Any],
        *,
        snapshot: dict[str, Any],
        user_text: str,
    ) -> dict[str, Any]:
        semantic_defaults = _semantic_payload_defaults(snapshot.get("semantic", {}))
        constraints = response.get("constraints", semantic_defaults["constraints"])
        if not isinstance(constraints, list):
            constraints = semantic_defaults["constraints"]
        normalized_constraints = [
            str(item).strip() for item in constraints if str(item).strip()
        ]
        gripper = str(response.get("gripper", "") or "").strip()
        if gripper:
            normalized_constraints = [
                gripper,
                *[item for item in normalized_constraints if item != gripper],
            ]
        excluded_labels = response.get(
            "excluded_labels", semantic_defaults["excluded_labels"]
        )
        if not isinstance(excluded_labels, list):
            excluded_labels = semantic_defaults["excluded_labels"]
        target_label = str(
            response.get("target_label")
            or response.get("target")
            or semantic_defaults["target_label"]
            or ""
        ).strip()
        target_hint = str(
            response.get("target_hint")
            or target_label
            or semantic_defaults["target_hint"]
            or self.default_target_hint
        ).strip()
        confidence_raw = response.get("confidence", semantic_defaults["confidence"])
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = float(semantic_defaults["confidence"])
        confidence = max(0.0, min(1.0, confidence))
        requested_action = str(
            response.get("requested_action")
            or response.get("action")
            or response.get("intent")
            or "update_task"
        ).strip().lower()
        action_aliases = {
            "review": "update_task",
            "update": "update_task",
            "execute_task": "execute",
            "run": "execute",
            "ask": "clarify",
            "question": "clarify",
            "stop": "cancel",
        }
        requested_action = action_aliases.get(requested_action, requested_action)
        if requested_action not in {"update_task", "execute", "clarify", "cancel"}:
            requested_action = "update_task"
        assistant_text = str(
            response.get("assistant_text")
            or response.get("reply")
            or response.get("message")
            or response.get("assistant")
            or ""
        ).strip()
        need_human_confirm = bool(
            response.get(
                "need_human_confirm",
                requested_action == "clarify" or confidence < 0.5,
            )
        )
        reason = str(response.get("reason", "") or "").strip()
        if not assistant_text:
            if requested_action == "clarify":
                assistant_text = "I need one more detail before updating the task."
            elif requested_action == "execute":
                assistant_text = "I updated the task and prepared execution."
            elif requested_action == "cancel":
                assistant_text = "I left the current task unchanged."
            else:
                assistant_text = "I updated the structured task."
        return {
            "assistant_text": assistant_text,
            "requested_action": requested_action,
            "task": str(response.get("task") or semantic_defaults["task"] or self.default_task),
            "target_label": target_label,
            "target_hint": target_hint,
            "constraints": normalized_constraints or list(self.default_constraints),
            "excluded_labels": [
                str(item).strip() for item in excluded_labels if str(item).strip()
            ],
            "confidence": confidence,
            "need_human_confirm": need_human_confirm,
            "reason": reason or f"dialog turn interpreted from: {str(user_text or '').strip()}",
        }

    def _publish_semantic_payload(
        self,
        payload: dict[str, Any],
        *,
        prompt_text: str,
        raw_json: str,
        intervention_active: bool,
        intervention_source: str = "",
        intervention_label: str = "",
    ) -> dict[str, Any]:
        constraints = [
            str(item).strip() for item in payload.get("constraints", []) if str(item).strip()
        ]
        gripper = str(payload.get("gripper", "") or "").strip()
        if gripper and gripper not in constraints:
            constraints = [gripper, *constraints]
        task_msg = SemanticTask()
        task_msg.header.stamp = self.get_clock().now().to_msg()
        task_msg.task = str(payload.get("task", "pick") or "pick")
        task_msg.target_label = str(payload.get("target_label", "") or "").strip()
        task_msg.target_hint = str(
            payload.get("target_hint", task_msg.target_label) or task_msg.target_label
        ).strip()
        task_msg.constraints = constraints
        task_msg.excluded_labels = [
            str(item).strip()
            for item in payload.get("excluded_labels", [])
            if str(item).strip()
        ]
        task_msg.confidence = float(payload.get("confidence", 1.0) or 1.0)
        task_msg.need_human_confirm = bool(payload.get("need_human_confirm", False))
        task_msg.reason = str(payload.get("reason", "") or "")
        task_msg.prompt_text = str(prompt_text or "")
        task_msg.raw_json = str(raw_json or "")
        self.semantic_override_pub.publish(task_msg)

        semantic = _semantic_task_to_dict(task_msg)
        with self._lock:
            self._semantic_task = semantic
            self._semantic_result = {
                **copy.deepcopy(payload),
                "prompt_text": task_msg.prompt_text,
                "raw_json": task_msg.raw_json,
                "updated_at": _now_sec(),
            }
            self._state_version += 1
        self._last_prompt_text = task_msg.prompt_text
        if intervention_active:
            label = intervention_label or task_msg.target_label or task_msg.target_hint
            self._set_intervention(True, intervention_source, label)
        else:
            self._set_intervention(False)
        return semantic

    def set_dialog_mode(self, mode: str) -> dict[str, Any]:
        normalized = "auto" if str(mode).lower() == "auto" else "review"
        self._set_dialog_status(mode=normalized, status="idle", pending_auto_execute=False, last_error="")
        self._append_dialog_message("system", f"Dialog mode switched to {normalized.title()}.")
        return self.build_state()

    def reset_dialog_session(self) -> dict[str, Any]:
        self._reset_dialog_session()
        return self.build_state()

    def dialog_send_message(self, payload: dict[str, Any]) -> dict[str, Any]:
        user_text = str(payload.get("message") or payload.get("prompt") or "").strip()
        if not user_text:
            raise ValueError("message is required")
        requested_mode = str(payload.get("mode", "") or "").strip().lower()
        if requested_mode not in {"review", "auto"}:
            with self._lock:
                requested_mode = str(self._dialog.get("mode", "review") or "review")

        snapshot = self._dialog_context_snapshot()
        history_messages = self._dialog_history_messages(snapshot.get("dialog", {}).get("messages", []))
        with self._lock:
            self._dialog["mode"] = requested_mode
            self._dialog["status"] = "thinking"
            self._dialog["pending_auto_execute"] = False
            self._dialog["last_error"] = ""
            self._dialog["updated_at"] = _now_sec()
            self._dialog_message_id += 1
            self._dialog["messages"].append(
                {
                    "id": int(self._dialog_message_id),
                    "role": "user",
                    "text": user_text,
                    "meta": "",
                    "semantic_task": None,
                    "requested_action": "",
                    "created_at": _now_sec(),
                }
            )
            self._dialog["messages"] = list(self._dialog["messages"][-40:])
            self._state_version += 1

        try:
            raw_text, parsed = self._query_dialog_model(
                user_text=user_text,
                mode=requested_mode,
                snapshot=snapshot,
                history_messages=history_messages,
            )
            normalized = self._normalize_dialog_response(
                parsed,
                snapshot=snapshot,
                user_text=user_text,
            )
        except Exception as exc:  # noqa: BLE001
            error_message = f"dialog inference failed: {exc}"
            self._set_dialog_status(status="error", pending_auto_execute=False, last_error=error_message)
            self._append_dialog_message("system", error_message)
            raise

        requested_action = str(normalized["requested_action"])
        semantic_payload: Optional[dict[str, Any]] = None
        if requested_action not in {"clarify", "cancel"}:
            semantic_payload = {
                key: normalized[key]
                for key in (
                    "task",
                    "target_label",
                    "target_hint",
                    "constraints",
                    "excluded_labels",
                    "confidence",
                    "need_human_confirm",
                    "reason",
                )
            }
            semantic_payload = self._publish_semantic_payload(
                semantic_payload,
                prompt_text=user_text,
                raw_json=raw_text,
                intervention_active=False,
            )
            self._record_event(
                "dialog",
                "info",
                f"dialog updated task: {semantic_payload.get('target_label') or semantic_payload.get('target_hint') or '<unset>'}",
                data={"mode": requested_mode, "requested_action": requested_action},
                feedback=False,
            )

        self._append_dialog_message(
            "assistant",
            str(normalized["assistant_text"]),
            meta=_dialog_message_summary(semantic_payload, requested_action),
            semantic_task=semantic_payload,
            requested_action=requested_action,
        )

        if requested_action == "cancel":
            self._set_dialog_status(status="ready", pending_auto_execute=False, last_error="")
            self._append_dialog_message("system", "Current structured task was left unchanged.")
            return self.build_state()

        if requested_mode == "auto" and requested_action == "execute":
            auto_note = self._configure_auto_execute(
                semantic_payload=semantic_payload,
                user_text=user_text,
                confidence=float(normalized["confidence"]),
                need_human_confirm=bool(normalized["need_human_confirm"]),
            )
            self._append_dialog_message("system", auto_note)
            return self.build_state()

        self._set_dialog_status(status="ready", pending_auto_execute=False, last_error="")
        return self.build_state()

    def _configure_auto_execute(
        self,
        *,
        semantic_payload: Optional[dict[str, Any]],
        user_text: str,
        confidence: float,
        need_human_confirm: bool,
    ) -> str:
        if semantic_payload is None:
            self._set_dialog_status(status="error", pending_auto_execute=False, last_error="no semantic task to execute")
            return "Auto mode could not start because no structured task was produced."
        target_name = str(
            semantic_payload.get("target_label")
            or semantic_payload.get("target_hint")
            or ""
        ).strip()
        if not target_name:
            self._set_dialog_status(status="ready", pending_auto_execute=False, last_error="")
            return "Auto mode held execution because the target is still unspecified."
        if need_human_confirm or confidence < self.dialog_auto_execute_confidence_threshold:
            self._set_dialog_status(status="ready", pending_auto_execute=False, last_error="")
            return (
                "Auto mode updated the task but held execution because the instruction is still ambiguous "
                "or low-confidence."
            )

        sweep_ok, sweep_message = self._call_trigger(
            self.start_search_sweep_client,
            timeout_sec=1.0,
        )
        if not sweep_ok:
            self._set_dialog_status(status="error", pending_auto_execute=False, last_error=sweep_message)
            return f"Auto mode could not start the search sweep: {sweep_message}"

        with self._lock:
            target_locked = bool(self._target_locked)
            pick_active = bool(self._pick_active)
        if pick_active:
            self._set_dialog_status(status="auto_executing", pending_auto_execute=False, last_error="")
            return "Auto mode kept the updated task, but a pick is already running."

        if target_locked:
            self._set_dialog_status(status="auto_executing", pending_auto_execute=False, last_error="")
            threading.Thread(
                target=self._run_auto_execute,
                args=("target already locked",),
                daemon=True,
            ).start()
            return "Auto mode accepted the task and started execution immediately."

        self._set_dialog_status(status="awaiting_lock", pending_auto_execute=True, last_error="")
        return "Auto mode accepted the task and is waiting for target lock before executing."

    def _run_auto_execute(self, reason: str) -> None:
        ok, message, _ = self.execute_pick()
        if ok:
            self._append_dialog_message("system", f"Auto execute requested: {reason}.")
            self._set_dialog_status(status="auto_executing", pending_auto_execute=False, last_error="")
            return
        self._append_dialog_message("system", f"Auto execute failed: {message}")
        self._set_dialog_status(status="error", pending_auto_execute=False, last_error=message)

    def _maybe_run_pending_auto_execute(self, locked: bool) -> None:
        if not locked:
            return
        with self._lock:
            should_execute = bool(self._dialog.get("pending_auto_execute", False)) and not bool(
                self._pick_active
            )
            if should_execute:
                self._dialog["pending_auto_execute"] = False
                self._dialog["status"] = "auto_executing"
                self._dialog["updated_at"] = _now_sec()
                self._state_version += 1
        if should_execute:
            threading.Thread(
                target=self._run_auto_execute,
                args=("target lock acquired",),
                daemon=True,
            ).start()

    def _call_trigger(self, client: Any, timeout_sec: float = 4.0) -> tuple[bool, str]:
        if not client.wait_for_service(timeout_sec=timeout_sec):
            return False, "service is unavailable"
        future = client.call_async(Trigger.Request())
        done = threading.Event()
        future.add_done_callback(lambda _: done.set())
        if not done.wait(timeout_sec):
            return False, "service call timed out"
        try:
            result = future.result()
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)
        return bool(result.success), str(result.message or "")

    def publish_prompt(self, prompt: str) -> dict[str, Any]:
        message = String()
        message.data = str(prompt or "").strip()
        self.prompt_pub.publish(message)
        self._last_prompt_text = message.data
        self._set_intervention(False)
        sweep_ok, sweep_message = self._call_trigger(self.start_search_sweep_client, timeout_sec=1.0)
        self._record_event(
            "prompt",
            "info",
            f"prompt submitted: {message.data or '<empty>'}",
            data={
                "search_sweep_started": sweep_ok,
                "search_sweep_message": sweep_message,
            },
            feedback=True,
        )
        if not sweep_ok:
            self._record_event(
                "execution",
                "warn",
                f"search sweep did not start after prompt: {sweep_message}",
                feedback=False,
            )
        return self.build_state()

    def publish_override(self, payload: dict[str, Any]) -> dict[str, Any]:
        target_label = str(payload.get("target_label", "") or "").strip()
        target_hint = str(payload.get("target_hint", "") or target_label).strip()
        semantic = self._publish_semantic_payload(
            {
                **payload,
                "reason": str(payload.get("reason", "manual override from web ui") or ""),
            },
            prompt_text=str(payload.get("prompt_text", self._last_prompt_text) or ""),
            raw_json=str(payload.get("raw_json", "") or ""),
            intervention_active=True,
            intervention_source="override_api",
            intervention_label=target_label or target_hint,
        )
        self._record_event(
            "override",
            "info",
            f"override applied: {target_label or target_hint or '<unset>'}",
            data={"label": target_label or target_hint, "task": semantic.get("task", "pick")},
            feedback=True,
        )
        return self.build_state()

    def replan(self) -> tuple[bool, str, dict[str, Any]]:
        ok, message = self._call_trigger(self.reset_pick_client)
        if not ok:
            self._record_event("execution", "error", f"replan failed: {message}", feedback=True)
            return False, message, self.build_state()

        replay_prompt = self._last_prompt_text or str(
            self._semantic_task.get("prompt_text", "") or ""
        )
        self._set_intervention(False)
        if replay_prompt:
            prompt_msg = String()
            prompt_msg.data = replay_prompt
            self.prompt_pub.publish(prompt_msg)
            self._record_event(
                "replan",
                "info",
                f"re-plan requested; replayed prompt: {replay_prompt}",
                feedback=True,
            )
            return True, "pick session reset and prompt replayed", self.build_state()

        self._record_event(
            "replan",
            "info",
            "pick session reset without prompt replay",
            feedback=True,
        )
        return True, "pick session reset", self.build_state()

    def execute_pick(self) -> tuple[bool, str, dict[str, Any]]:
        ok, message = self._call_trigger(self.execute_pick_client)
        self._record_event(
            "execution",
            "info" if ok else "error",
            "execute request accepted" if ok else f"execute request failed: {message}",
            feedback=True,
        )
        return ok, message, self.build_state()

    def build_state(self) -> dict[str, Any]:
        with self._lock:
            semantic = copy.deepcopy(self._semantic_task)
            semantic_result = copy.deepcopy(self._semantic_result)
            detection = copy.deepcopy(self._detection_result)
            detection_debug = copy.deepcopy(self._detection_debug)
            pick_status = copy.deepcopy(self._pick_status)
            proposals = copy.deepcopy(self._grasp_proposals)
            backend_debug = copy.deepcopy(self._grasp_backend_debug)
            tactile = copy.deepcopy(self._tactile)
            arm_state = copy.deepcopy(self._arm_state)
            intervention = copy.deepcopy(self._intervention)
            health_by_node = copy.deepcopy(self._health_by_node)
            dialog = copy.deepcopy(self._dialog)
            events = list(self._events)
            feedback_events = list(self._feedback_events)
            target_locked = bool(self._target_locked)
            pick_active = bool(self._pick_active)
            state_version = int(self._state_version)

        health_items = sorted(
            health_by_node.values(),
            key=lambda item: (
                int(item.get("level", 0)) * -1,
                float(item.get("updated_at", 0.0)) * -1,
            ),
        )
        health_issues = [
            item
            for item in health_items
            if (not item.get("healthy", True)) or int(item.get("level", 0)) > 0
        ]

        vision_updated_at = max(
            float(detection.get("updated_at", 0.0)),
            float(detection_debug.get("updated_at", 0.0)),
            float(self.rgb_stream.latest().updated_at),
            float(self.detection_overlay_stream.latest().updated_at),
            float(self.grasp_overlay_stream.latest().updated_at),
        )
        execution_updated_at = max(
            float(pick_status.get("updated_at", 0.0)),
            float(detection.get("updated_at", 0.0)),
            float(proposals.get("updated_at", 0.0)),
        )

        phase = self._derive_phase(
            semantic=semantic,
            detection=detection,
            detection_debug=detection_debug,
            target_locked=target_locked,
            pick_active=pick_active,
            pick_status=pick_status,
        )
        return _json_compatible(
            {
                "connection": {
                    "backend_ready": True,
                    "backend_time": _now_sec(),
                    "state_version": state_version,
                    "host": self.host,
                    "port": self.port,
                },
                "semantic": {
                    **semantic,
                    "result": semantic_result,
                    "updated_at": float(semantic.get("updated_at", 0.0)),
                },
                "vision": {
                    "detection": detection,
                    "debug": detection_debug,
                    "debug_candidates": detection_debug.get("debug_candidates", []),
                    "selected_candidate": detection_debug.get("selected_candidate"),
                    "candidate_summary": detection_debug.get("candidate_summary", ""),
                    "image_width": int(
                        detection.get("image_width", 0)
                        or detection_debug.get("image_width", 0)
                        or 0
                    ),
                    "image_height": int(
                        detection.get("image_height", 0)
                        or detection_debug.get("image_height", 0)
                        or 0
                    ),
                    "updated_at": vision_updated_at,
                },
                "execution": {
                    "phase": phase,
                    "target_locked": target_locked,
                    "pick_active": pick_active,
                    "pick_status": pick_status,
                    "intervention": intervention,
                    "grasp_proposals": proposals,
                    "backend_debug": backend_debug,
                    "updated_at": execution_updated_at,
                },
                "tactile": tactile,
                "health": {
                    "healthy": not health_issues,
                    "issues": health_issues,
                    "latest": health_items[:12],
                    "arm_state": arm_state,
                    "updated_at": max(
                        float(arm_state.get("updated_at", 0.0)),
                        max(
                            (float(item.get("updated_at", 0.0)) for item in health_items),
                            default=0.0,
                        ),
                    ),
                },
                "logs": {
                    "stepper_phase": phase,
                    "intervention_badge": bool(intervention.get("active", False)),
                    "events": events,
                    "updated_at": max(
                        (float(item.get("created_at", 0.0)) for item in events), default=0.0
                    ),
                },
                "ui_feedback": {
                    "events": feedback_events,
                    "last_event_id": int(feedback_events[-1]["id"]) if feedback_events else 0,
                },
                "dialog": {
                    **dialog,
                    "status_label": _dialog_status_label(str(dialog.get("status", "idle"))),
                },
            }
        )

    def _derive_phase(
        self,
        *,
        semantic: dict[str, Any],
        detection: dict[str, Any],
        detection_debug: dict[str, Any],
        target_locked: bool,
        pick_active: bool,
        pick_status: dict[str, Any],
    ) -> str:
        pick_phase = str(pick_status.get("phase", "") or "").strip()
        if pick_phase in {"waiting_execute", "planning", "executing", "completed", "error"}:
            return pick_phase
        if pick_active:
            return "executing"
        if target_locked:
            return "target_locked"
        if bool(detection.get("accepted")) or bool(detection_debug.get("debug_candidates")):
            return "vision_ready"
        if bool(semantic.get("task")) or bool(semantic.get("target_label")) or bool(
            semantic.get("target_hint")
        ):
            return "semantic_ready"
        return "idle"


def _frontend_root_candidates(bridge: TactileWebGateway) -> list[Path]:
    candidates: list[Path] = []
    seen: set[str] = set()

    def add_candidate(path: Path) -> None:
        resolved = path.expanduser().resolve()
        key = resolved.as_posix()
        if key in seen:
            return
        seen.add(key)
        candidates.append(resolved)

    if bridge.frontend_dist_dir:
        add_candidate(Path(bridge.frontend_dist_dir))

    if get_package_share_directory is not None:
        try:
            share_dir = Path(get_package_share_directory("tactile_web_bridge"))
            add_candidate(share_dir / "frontend" / "dist")
        except PackageNotFoundError:
            pass
        except Exception:  # noqa: BLE001
            pass

    add_candidate(Path(__file__).resolve().parents[1] / "frontend" / "dist")
    return candidates


def _frontend_root_dir(bridge: TactileWebGateway) -> Path:
    candidates = _frontend_root_candidates(bridge)
    for candidate in candidates:
        if (candidate / "index.html").exists():
            return candidate
    return candidates[0]


def create_app(bridge: TactileWebGateway) -> Any:
    app = FastAPI(title="Programme UI Gateway")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5173",
            "http://localhost:5173",
            "http://127.0.0.1:8765",
            "http://localhost:8765",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    dist_dir = _frontend_root_dir(bridge)
    assets_dir = dist_dir / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/api/bootstrap")
    async def bootstrap() -> JSONResponse:
        return JSONResponse(
            {
                "state": bridge.build_state(),
                "streams": {
                    "rgb": "/api/streams/rgb.mjpeg",
                    "detection_overlay": "/api/streams/detection_overlay.mjpeg",
                    "grasp_overlay": "/api/streams/grasp_overlay.mjpeg",
                },
                "frontend_ready": (dist_dir / "index.html").exists(),
            }
        )

    @app.post("/api/prompt")
    async def post_prompt(payload: dict[str, Any]) -> JSONResponse:
        prompt = str(payload.get("prompt", "") or "").strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="prompt is required")
        return JSONResponse({"ok": True, "state": bridge.publish_prompt(prompt)})

    @app.post("/api/dialog/message")
    async def post_dialog_message(payload: dict[str, Any]) -> JSONResponse:
        try:
            state = bridge.dialog_send_message(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except requests.HTTPError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return JSONResponse({"ok": True, "state": state})

    @app.post("/api/dialog/mode")
    async def post_dialog_mode(payload: dict[str, Any]) -> JSONResponse:
        mode = str(payload.get("mode", "") or "").strip().lower()
        if mode not in {"review", "auto"}:
            raise HTTPException(status_code=400, detail="mode must be review or auto")
        return JSONResponse({"ok": True, "state": bridge.set_dialog_mode(mode)})

    @app.post("/api/dialog/reset")
    async def post_dialog_reset() -> JSONResponse:
        return JSONResponse({"ok": True, "state": bridge.reset_dialog_session()})

    @app.post("/api/task/override")
    async def post_override(payload: dict[str, Any]) -> JSONResponse:
        return JSONResponse({"ok": True, "state": bridge.publish_override(payload)})

    @app.post("/api/task/replan")
    async def post_replan() -> JSONResponse:
        ok, message, state = bridge.replan()
        status_code = 200 if ok else 503
        return JSONResponse(
            {"ok": ok, "message": message, "state": state},
            status_code=status_code,
        )

    @app.post("/api/execution/execute")
    async def post_execute() -> JSONResponse:
        ok, message, state = bridge.execute_pick()
        status_code = 200 if ok else 409
        return JSONResponse(
            {"ok": ok, "message": message, "state": state},
            status_code=status_code,
        )

    def _stream_generator(buffer: MjpegFrameBuffer):
        last_update = 0.0
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        while True:
            frame = buffer.wait_next(last_update, timeout_sec=1.0)
            if frame.jpeg is None or frame.updated_at <= last_update:
                continue
            last_update = frame.updated_at
            yield boundary + frame.jpeg + b"\r\n"

    @app.get("/api/streams/rgb.mjpeg")
    async def stream_rgb() -> StreamingResponse:
        return StreamingResponse(
            _stream_generator(bridge.rgb_stream),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/api/streams/detection_overlay.mjpeg")
    async def stream_detection_overlay() -> StreamingResponse:
        return StreamingResponse(
            _stream_generator(bridge.detection_overlay_stream),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/api/streams/grasp_overlay.mjpeg")
    async def stream_grasp_overlay() -> StreamingResponse:
        return StreamingResponse(
            _stream_generator(bridge.grasp_overlay_stream),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/api/diagnostics/streams")
    async def get_stream_diagnostics() -> JSONResponse:
        return JSONResponse(
            {
                "rgb": bridge.rgb_encoder.stats(),
                "detection_overlay": bridge.detection_overlay_encoder.stats(),
                "grasp_overlay": bridge.grasp_overlay_encoder.stats(),
                "jpeg_quality": int(bridge.stream_jpeg_quality),
            }
        )

    @app.websocket("/ws/live")
    async def ws_state(websocket: WebSocket) -> None:
        await websocket.accept()
        last_state_version = -1
        last_sent_at = 0.0
        try:
            while True:
                state = bridge.build_state()
                state_version = int(state["connection"]["state_version"])
                now_sec = _now_sec()
                if state_version != last_state_version or (now_sec - last_sent_at) >= 1.5:
                    await websocket.send_json(state)
                    last_state_version = state_version
                    last_sent_at = now_sec
                await asyncio.sleep(0.5)
        except (WebSocketDisconnect, RuntimeError):
            return

    def _fallback_html() -> str:
        return """
        <!doctype html>
        <html lang="en">
          <head>
            <meta charset="utf-8" />
            <title>Programme UI Gateway</title>
            <style>
              body { background: #0d1524; color: #e8edf7; font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif; margin: 0; padding: 48px; }
              .panel { max-width: 820px; margin: 0 auto; padding: 32px; border-radius: 22px; background: linear-gradient(160deg, rgba(28,45,74,0.95), rgba(10,17,30,0.96)); box-shadow: 0 24px 80px rgba(0,0,0,0.35); }
              h1 { margin: 0 0 12px; font-size: 34px; }
              p { line-height: 1.6; color: #bfd0ea; }
              code { background: rgba(255,255,255,0.08); padding: 2px 6px; border-radius: 6px; }
            </style>
          </head>
          <body>
            <div class="panel">
              <h1>Programme UI Gateway</h1>
              <p>The backend is running, but the built frontend bundle was not found.</p>
              <p>Run the Vite app under <code>src/tactile_web_bridge/frontend</code> for development, or build it so the bundle appears under <code>frontend/dist</code>.</p>
            </div>
          </body>
        </html>
        """

    @app.get("/")
    async def root() -> HTMLResponse:
        index_path = dist_dir / "index.html"
        if index_path.exists():
            return HTMLResponse(index_path.read_text(encoding="utf-8"))
        return HTMLResponse(_fallback_html())

    @app.get("/{full_path:path}")
    async def spa_fallback(full_path: str) -> HTMLResponse:
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="not found")
        index_path = dist_dir / "index.html"
        if index_path.exists():
            return HTMLResponse(index_path.read_text(encoding="utf-8"))
        return HTMLResponse(_fallback_html(), status_code=200)

    return app


def main(args: Optional[list[str]] = None) -> None:
    try:
        import uvicorn
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "tactile_web_gateway requires fastapi and uvicorn. "
            "Install the tactile_web_bridge Python dependencies first."
        ) from exc

    rclpy.init(args=args)
    node = TactileWebGateway()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    app = create_app(node)
    try:
        uvicorn.run(app, host=node.host, port=node.port, log_level="info", ws="wsproto")
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
