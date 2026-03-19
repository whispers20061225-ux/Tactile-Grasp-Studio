from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import requests
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, RegionOfInterest
from std_msgs.msg import Bool, String
from tactile_interfaces.msg import DetectionResult, SemanticTask

from tactile_vision.modular_common import (
    coerce_bbox,
    coerce_point,
    compact_json,
    decode_color_image,
    decode_png_base64_mask,
    encode_image_to_base64_jpeg,
    make_mono8_image,
    make_rgb8_image,
)

COLOR_HINT_TOKENS = {
    "black",
    "blue",
    "brown",
    "gray",
    "green",
    "grey",
    "orange",
    "pink",
    "purple",
    "red",
    "white",
    "yellow",
}

SEMANTIC_ALIAS_LABELS = {
    "container": {"bottle", "can", "cup", "vase", "wine glass"},
    "cylinder": {"bottle", "can", "cup", "vase", "wine glass"},
    "cylindrical": {"bottle", "can", "cup", "vase", "wine glass"},
    "drinkware": {"bottle", "can", "cup", "wine glass"},
    "tube": {"bottle", "can", "cup", "vase"},
}


class DetectorSegNode(Node):
    def __init__(self) -> None:
        super().__init__("detector_seg_node")

        self.declare_parameter("color_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("semantic_task_topic", "/qwen/semantic_task")
        self.declare_parameter("detection_result_topic", "/perception/detection_result")
        self.declare_parameter(
            "candidate_visible_topic", "/sim/perception/target_candidate_visible"
        )
        self.declare_parameter("detection_debug_topic", "/perception/detection_debug")
        self.declare_parameter(
            "detection_debug_overlay_topic", "/perception/detection_debug_overlay"
        )
        self.declare_parameter("backend", "ultralytics_local")
        self.declare_parameter("backend_url", "")
        self.declare_parameter("request_timeout_sec", 10.0)
        self.declare_parameter("max_inference_rate_hz", 2.0)
        self.declare_parameter("enabled", True)
        self.declare_parameter("ultralytics_model_path", "yolo11s-seg.pt")
        self.declare_parameter("ultralytics_runtime_preference", "auto")
        self.declare_parameter("ultralytics_device", "")
        self.declare_parameter("ultralytics_imgsz", 960)
        self.declare_parameter("confidence_threshold", 0.25)
        self.declare_parameter("iou_threshold", 0.55)
        self.declare_parameter("candidate_complete_edge_margin_px", 16)
        self.declare_parameter("jpeg_quality", 90)
        self.declare_parameter("publish_debug_overlay", True)
        self.declare_parameter("debug_top_k_candidates", 5)
        self.declare_parameter("debug_candidate_confidence_floor", 0.05)
        self.declare_parameter("semantic_match_confidence_floor", 0.05)
        self.declare_parameter("log_interval_sec", 10.0)

        self.color_topic = str(self.get_parameter("color_topic").value)
        self.semantic_task_topic = str(self.get_parameter("semantic_task_topic").value)
        self.detection_result_topic = str(self.get_parameter("detection_result_topic").value)
        self.candidate_visible_topic = str(self.get_parameter("candidate_visible_topic").value)
        self.detection_debug_topic = str(self.get_parameter("detection_debug_topic").value)
        self.detection_debug_overlay_topic = str(
            self.get_parameter("detection_debug_overlay_topic").value
        )
        self.backend = str(self.get_parameter("backend").value).strip().lower()
        self.backend_url = str(self.get_parameter("backend_url").value).strip()
        self.request_timeout_sec = max(
            0.5, float(self.get_parameter("request_timeout_sec").value)
        )
        self.max_inference_rate_hz = max(
            0.2, float(self.get_parameter("max_inference_rate_hz").value)
        )
        self.enabled = bool(self.get_parameter("enabled").value)
        self.ultralytics_model_path = str(
            self.get_parameter("ultralytics_model_path").value
        ).strip()
        self.ultralytics_runtime_preference = str(
            self.get_parameter("ultralytics_runtime_preference").value
        ).strip().lower()
        self.ultralytics_device = str(self.get_parameter("ultralytics_device").value).strip()
        self.ultralytics_imgsz = max(320, int(self.get_parameter("ultralytics_imgsz").value))
        self.confidence_threshold = max(
            0.0, min(1.0, float(self.get_parameter("confidence_threshold").value))
        )
        self.iou_threshold = max(
            0.05, min(0.95, float(self.get_parameter("iou_threshold").value))
        )
        self.candidate_complete_edge_margin_px = max(
            0, int(self.get_parameter("candidate_complete_edge_margin_px").value)
        )
        self.jpeg_quality = max(50, min(100, int(self.get_parameter("jpeg_quality").value)))
        self.publish_debug_overlay = bool(self.get_parameter("publish_debug_overlay").value)
        self.debug_top_k_candidates = max(
            1, int(self.get_parameter("debug_top_k_candidates").value)
        )
        self.debug_candidate_confidence_floor = max(
            0.0,
            min(1.0, float(self.get_parameter("debug_candidate_confidence_floor").value)),
        )
        self.semantic_match_confidence_floor = max(
            0.0,
            min(1.0, float(self.get_parameter("semantic_match_confidence_floor").value)),
        )
        self.log_interval_sec = max(1.0, float(self.get_parameter("log_interval_sec").value))

        self._session = requests.Session()
        self._request_lock = threading.Lock()
        self._semantic_lock = threading.Lock()
        self._color_lock = threading.Lock()
        self._request_in_flight = False
        self._pending_request = False
        self._latest_color_msg: Optional[Image] = None
        self._semantic_task: Optional[SemanticTask] = None
        self._last_terminal_summary = ""
        self._last_candidate_summary = ""
        self._model = None
        self._resolved_ultralytics_model_path = ""
        self._ultralytics_runtime = "torch"
        self._effective_ultralytics_imgsz = 0

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

        self.create_subscription(Image, self.color_topic, self._on_color_image, qos_sensor)
        self.create_subscription(
            SemanticTask, self.semantic_task_topic, self._on_semantic_task, qos_reliable
        )

        self.detection_pub = self.create_publisher(
            DetectionResult, self.detection_result_topic, qos_reliable
        )
        self.candidate_visible_pub = self.create_publisher(
            Bool, self.candidate_visible_topic, qos_reliable
        )
        self.debug_pub = self.create_publisher(String, self.detection_debug_topic, qos_reliable)
        self.debug_overlay_pub = None
        if self.publish_debug_overlay:
            self.debug_overlay_pub = self.create_publisher(
                Image,
                self.detection_debug_overlay_topic,
                qos_reliable,
            )
        if self.backend in ("http_json", "ultralytics_local"):
            self.create_timer(1.0 / self.max_inference_rate_hz, self._maybe_run_inference)
        if self.backend == "ultralytics_local":
            self._prepare_ultralytics_runtime_env()
            self._log_ultralytics_runtime_status()

        self.get_logger().info(
            "detector_seg_node started: "
            f"backend={self.backend}, result={self.detection_result_topic}, "
            f"candidate_visible={self.candidate_visible_topic}"
        )

    def _log_ultralytics_runtime_status(self) -> None:
        try:
            import ultralytics  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(
                "detector backend is ultralytics_local, but ultralytics is unavailable: "
                f"{exc}. Install it before launching YOLO11-seg."
            )
            return

        self.get_logger().info(
            "ultralytics_local backend ready: "
            f"model={self.ultralytics_model_path}, device={self.ultralytics_device or 'auto'}, "
            f"runtime_pref={self.ultralytics_runtime_preference or 'auto'}"
        )

    def _prepare_ultralytics_runtime_env(self) -> None:
        os.environ.setdefault("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS", "1")
        if not self._should_prefer_onnx_runtime():
            return
        for key in (
            "CUDA_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
            "MESA_D3D12_DEFAULT_ADAPTER_NAME",
        ):
            os.environ.pop(key, None)

    def _should_prefer_onnx_runtime(self) -> bool:
        if self.ultralytics_runtime_preference == "onnx":
            return True
        if self.ultralytics_runtime_preference == "torch":
            return False
        device = self.ultralytics_device.strip().lower()
        return device in {"", "cpu"}

    def _resolve_ultralytics_model_path(self) -> tuple[str, str]:
        model_path = Path(self.ultralytics_model_path).expanduser()
        if not model_path.is_absolute():
            model_path = (Path.cwd() / model_path).resolve()
        else:
            model_path = model_path.resolve()

        if not model_path.exists():
            raise FileNotFoundError(f"ultralytics model not found: {model_path}")

        if model_path.suffix.lower() == ".onnx":
            return model_path.as_posix(), "onnx"
        if model_path.suffix.lower() != ".pt" or not self._should_prefer_onnx_runtime():
            return model_path.as_posix(), "torch"

        onnx_path = model_path.with_suffix(".onnx")
        if onnx_path.exists():
            return onnx_path.as_posix(), "onnx"

        try:
            import onnx  # noqa: F401
            import onnxruntime  # noqa: F401
            from ultralytics import YOLO
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(
                "ONNX runtime fallback unavailable; staying on PyTorch inference: "
                f"{exc}"
            )
            return model_path.as_posix(), "torch"

        self.get_logger().info(
            "exporting ONNX segmentation model for stable CPU inference: "
            f"{model_path.name} -> {onnx_path.name}"
        )
        exported_path = YOLO(model_path.as_posix()).export(
            format="onnx",
            imgsz=self.ultralytics_imgsz,
            simplify=False,
        )
        resolved_export = Path(str(exported_path)).expanduser().resolve()
        return resolved_export.as_posix(), "onnx"

    def _resolve_onnx_input_imgsz(self, model_path: str) -> int:
        try:
            import onnxruntime as ort
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(
                f"failed to inspect ONNX input size for {model_path}: {exc}"
            )
            return int(self.ultralytics_imgsz)

        try:
            session = ort.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"],
            )
            inputs = session.get_inputs()
            if not inputs:
                return int(self.ultralytics_imgsz)
            shape = list(inputs[0].shape)
            if len(shape) < 4:
                return int(self.ultralytics_imgsz)
            height = shape[2]
            width = shape[3]
            if isinstance(height, int) and isinstance(width, int) and height == width and height > 0:
                return int(height)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(
                f"failed to read ONNX model metadata from {model_path}: {exc}"
            )
        return int(self.ultralytics_imgsz)

    def _on_color_image(self, msg: Image) -> None:
        with self._color_lock:
            self._latest_color_msg = msg
        if self.backend in ("http_json", "ultralytics_local"):
            self._pending_request = True

    def _on_semantic_task(self, msg: SemanticTask) -> None:
        with self._semantic_lock:
            self._semantic_task = msg
        if self.backend == "disabled":
            self._publish_detection(
                self._build_empty_result(
                    task=msg.task,
                    target_label=msg.target_label,
                    reason="detector backend disabled; configure YOLO11 segmentation backend",
                )
            )
        else:
            self._pending_request = True

    def _maybe_run_inference(self) -> None:
        if not self.enabled or self.backend not in ("http_json", "ultralytics_local"):
            return
        if not self._pending_request:
            return

        with self._request_lock:
            if self._request_in_flight:
                return
            self._request_in_flight = True
            self._pending_request = False

        with self._color_lock:
            color_msg = self._latest_color_msg
        with self._semantic_lock:
            semantic_task = self._semantic_task

        self._run_inference(color_msg, semantic_task)

    def _run_inference(
        self,
        color_msg: Optional[Image],
        semantic_task: Optional[SemanticTask],
    ) -> None:
        try:
            if color_msg is None:
                raise ValueError("missing color image for detector inference")
            image_rgb = decode_color_image(color_msg)
            if image_rgb is None:
                raise ValueError(f"unsupported color image encoding: {color_msg.encoding}")
            image_rgb = np.array(image_rgb, dtype=np.uint8, copy=True, order="C")

            if self.backend == "http_json":
                payload = self._run_http_inference(color_msg, image_rgb, semantic_task)
            elif self.backend == "ultralytics_local":
                payload = self._run_ultralytics_inference(color_msg, image_rgb, semantic_task)
            else:
                payload = self._build_empty_result(
                    task=semantic_task.task if semantic_task is not None else "pick",
                    target_label=semantic_task.target_label if semantic_task is not None else "",
                    reason=f"unsupported detector backend: {self.backend}",
                    image_width=int(color_msg.width),
                    image_height=int(color_msg.height),
                    frame_id=color_msg.header.frame_id,
                )

            self._publish_detection(payload)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"detector inference failed: {exc}")
            self._publish_detection(
                self._build_empty_result(
                    task=semantic_task.task if semantic_task is not None else "pick",
                    target_label=semantic_task.target_label if semantic_task is not None else "",
                    reason=str(exc),
                    image_width=int(color_msg.width) if color_msg is not None else 0,
                    image_height=int(color_msg.height) if color_msg is not None else 0,
                    frame_id=color_msg.header.frame_id if color_msg is not None else "",
                )
            )
        finally:
            with self._request_lock:
                self._request_in_flight = False

    def _run_http_inference(
        self,
        color_msg: Image,
        image_rgb: np.ndarray,
        semantic_task: Optional[SemanticTask],
    ) -> dict[str, Any]:
        if not self.backend_url:
            raise ValueError("detector backend_url is empty")

        image_b64 = encode_image_to_base64_jpeg(image_rgb, self.jpeg_quality)
        semantic_payload = self._semantic_payload(semantic_task)
        response = self._session.post(
            self.backend_url,
            json={
                "image_jpeg_b64": image_b64,
                "semantic_task": semantic_payload,
            },
            timeout=self.request_timeout_sec,
        )
        response.raise_for_status()
        parsed = response.json()
        if not isinstance(parsed, dict):
            raise ValueError("detector response is not a JSON object")

        bbox_xyxy = coerce_bbox(parsed.get("bbox_xyxy", parsed.get("bbox")), int(color_msg.width), int(color_msg.height))
        point_px = coerce_point(parsed.get("point_px", parsed.get("point")), int(color_msg.width), int(color_msg.height))
        if point_px is None and bbox_xyxy is not None:
            point_px = self._bbox_center_point(bbox_xyxy)

        mask = None
        mask_png_b64 = parsed.get("mask_png_b64")
        if isinstance(mask_png_b64, str) and mask_png_b64.strip():
            mask = decode_png_base64_mask(mask_png_b64)
            if mask is not None and mask.shape[:2] != image_rgb.shape[:2]:
                mask = cv2.resize(
                    mask,
                    (image_rgb.shape[1], image_rgb.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

        return self._build_detection_payload(
            backend="http_json",
            task=str(parsed.get("task") or semantic_payload["task"]),
            target_label=str(
                parsed.get("target_label")
                or parsed.get("target")
                or parsed.get("label")
                or semantic_payload["target_label"]
                or semantic_payload["target_hint"]
            ).strip(),
            confidence=float(parsed.get("confidence", parsed.get("score", 0.0)) or 0.0),
            need_human_confirm=bool(parsed.get("need_human_confirm", False)),
            reason=str(parsed.get("reason") or ""),
            image_width=int(color_msg.width),
            image_height=int(color_msg.height),
            frame_id=color_msg.header.frame_id,
            bbox_xyxy=bbox_xyxy,
            point_px=point_px,
            mask_u8=mask,
        )

    def _ensure_ultralytics_model(self):
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLO
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "ultralytics is not installed; install it before using backend=ultralytics_local"
            ) from exc

        self._resolved_ultralytics_model_path, self._ultralytics_runtime = (
            self._resolve_ultralytics_model_path()
        )
        self._effective_ultralytics_imgsz = int(self.ultralytics_imgsz)
        if self._ultralytics_runtime == "onnx":
            self._effective_ultralytics_imgsz = self._resolve_onnx_input_imgsz(
                self._resolved_ultralytics_model_path
            )
        self._model = YOLO(self._resolved_ultralytics_model_path)
        self.get_logger().info(
            "ultralytics model loaded: "
            f"runtime={self._ultralytics_runtime}, "
            f"model={self._resolved_ultralytics_model_path}, "
            f"imgsz={self._effective_ultralytics_imgsz}"
        )
        return self._model

    def _run_ultralytics_inference(
        self,
        color_msg: Image,
        image_rgb: np.ndarray,
        semantic_task: Optional[SemanticTask],
    ) -> dict[str, Any]:
        model = self._ensure_ultralytics_model()
        prediction_kwargs: dict[str, Any] = {
            "source": image_rgb,
            "imgsz": self._effective_ultralytics_imgsz or self.ultralytics_imgsz,
            "conf": min(
                self.confidence_threshold,
                self.debug_candidate_confidence_floor,
                self.semantic_match_confidence_floor,
            ),
            "iou": self.iou_threshold,
            "verbose": False,
        }
        if self._ultralytics_runtime != "onnx" and self.ultralytics_device:
            prediction_kwargs["device"] = self.ultralytics_device
        prediction = model.predict(**prediction_kwargs)
        if not prediction:
            return self._build_empty_result(
                task=semantic_task.task if semantic_task is not None else "pick",
                target_label=semantic_task.target_label if semantic_task is not None else "",
                reason="no detections returned by YOLO11-seg",
                image_width=int(color_msg.width),
                image_height=int(color_msg.height),
                frame_id=color_msg.header.frame_id,
            )

        result = prediction[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return self._build_empty_result(
                task=semantic_task.task if semantic_task is not None else "pick",
                target_label=semantic_task.target_label if semantic_task is not None else "",
                reason="YOLO11-seg found no candidate instances",
                image_width=int(color_msg.width),
                image_height=int(color_msg.height),
                frame_id=color_msg.header.frame_id,
            )

        names = getattr(result, "names", {})
        masks_data = None
        masks_obj = getattr(result, "masks", None)
        if masks_obj is not None and getattr(masks_obj, "data", None) is not None:
            masks_data = masks_obj.data
            if hasattr(masks_data, "cpu"):
                masks_data = masks_data.cpu().numpy()
            else:
                masks_data = np.asarray(masks_data)

        candidates: list[dict[str, Any]] = []
        raw_candidates: list[dict[str, Any]] = []
        for idx in range(len(boxes)):
            cls_tensor = boxes.cls[idx]
            conf_tensor = boxes.conf[idx]
            cls_id = int(cls_tensor.item() if hasattr(cls_tensor, "item") else cls_tensor)
            confidence = float(
                conf_tensor.item() if hasattr(conf_tensor, "item") else conf_tensor
            )
            label = (
                str(names.get(cls_id, cls_id))
                if isinstance(names, dict)
                else str(names[cls_id])
            )
            semantic_bonus = self._semantic_match_bonus(label, semantic_task)
            bbox_xyxy = self._extract_bbox_xyxy(
                boxes,
                idx,
                int(color_msg.width),
                int(color_msg.height),
            )
            mask_u8 = self._extract_mask_u8(masks_data, idx, image_rgb.shape[:2])
            mask_pixels = (
                int(np.count_nonzero(mask_u8)) if isinstance(mask_u8, np.ndarray) else 0
            )
            score = confidence + semantic_bonus
            confidence_floor = self._candidate_confidence_floor(semantic_bonus, semantic_task)
            status = "selectable"
            if confidence < confidence_floor:
                status = "filtered_low_confidence"
            elif semantic_bonus <= -1.0:
                status = "filtered_excluded_label"

            candidate = {
                "index": int(idx),
                "class_id": int(cls_id),
                "label": label,
                "confidence": float(confidence),
                "semantic_bonus": float(semantic_bonus),
                "score": float(score),
                "confidence_floor": float(confidence_floor),
                "bbox_xyxy": bbox_xyxy,
                "mask_u8": mask_u8,
                "mask_pixels": int(mask_pixels),
                "status": status,
            }
            raw_candidates.append(candidate)
            if status == "selectable":
                candidates.append(candidate)

        if not candidates:
            top_candidates = self._select_debug_candidates(raw_candidates, None)
            payload = self._build_empty_result(
                task=semantic_task.task if semantic_task is not None else "pick",
                target_label=semantic_task.target_label if semantic_task is not None else "",
                reason="YOLO11-seg found detections, but none passed selection filters",
                image_width=int(color_msg.width),
                image_height=int(color_msg.height),
                frame_id=color_msg.header.frame_id,
            )
            payload["semantic_target"] = self._semantic_payload(semantic_task)
            payload["debug_candidates"] = [
                self._candidate_debug_public(candidate) for candidate in top_candidates
            ]
            payload["selected_candidate"] = None
            payload["candidate_summary"] = self._format_candidate_summary(
                semantic_task,
                top_candidates,
                None,
            )
            if self.publish_debug_overlay:
                payload["debug_overlay_rgb"] = self._render_debug_overlay(
                    image_rgb,
                    top_candidates,
                    None,
                    semantic_task,
                )
            return payload

        if self._semantic_task_requires_match(semantic_task):
            semantically_relevant = [
                item for item in candidates if float(item["semantic_bonus"]) > 0.0
            ]
            if semantically_relevant:
                candidates = semantically_relevant

        selected_candidate = max(candidates, key=lambda item: float(item["score"]))
        top_candidates = self._select_debug_candidates(raw_candidates, selected_candidate)
        payload = self._build_detection_payload(
            backend=(
                f"ultralytics_{self._ultralytics_runtime}:"
                f"{Path(self._resolved_ultralytics_model_path or self.ultralytics_model_path).name}"
            ),
            task=semantic_task.task if semantic_task is not None else "pick",
            target_label=str(selected_candidate["label"]),
            confidence=float(selected_candidate["confidence"]),
            need_human_confirm=False,
            reason="",
            image_width=int(color_msg.width),
            image_height=int(color_msg.height),
            frame_id=color_msg.header.frame_id,
            bbox_xyxy=selected_candidate["bbox_xyxy"],
            point_px=self._bbox_center_point(selected_candidate["bbox_xyxy"]),
            mask_u8=selected_candidate["mask_u8"],
        )
        payload["semantic_target"] = self._semantic_payload(semantic_task)
        payload["debug_candidates"] = [
            self._candidate_debug_public(candidate) for candidate in top_candidates
        ]
        payload["selected_candidate"] = self._candidate_debug_public(selected_candidate)
        payload["candidate_summary"] = self._format_candidate_summary(
            semantic_task,
            top_candidates,
            selected_candidate,
        )
        if self.publish_debug_overlay:
            payload["debug_overlay_rgb"] = self._render_debug_overlay(
                image_rgb,
                top_candidates,
                selected_candidate,
                semantic_task,
            )
        return payload

    def _semantic_payload(self, semantic_task: Optional[SemanticTask]) -> dict[str, Any]:
        return {
            "task": semantic_task.task if semantic_task is not None else "pick",
            "target_label": semantic_task.target_label if semantic_task is not None else "",
            "target_hint": semantic_task.target_hint if semantic_task is not None else "",
            "constraints": list(semantic_task.constraints) if semantic_task is not None else [],
            "excluded_labels": list(semantic_task.excluded_labels)
            if semantic_task is not None
            else [],
        }

    def _semantic_match_bonus(
        self,
        detected_label: str,
        semantic_task: Optional[SemanticTask],
    ) -> float:
        label = detected_label.strip().lower()
        if not label or semantic_task is None:
            return 0.0

        bonus = 0.0
        target_label = str(semantic_task.target_label or "").strip().lower()
        target_hint = str(semantic_task.target_hint or "").strip().lower()
        excluded = {str(item).strip().lower() for item in semantic_task.excluded_labels}
        if label in excluded:
            return -1.0
        if target_label:
            if target_label == label:
                bonus += 0.35
            elif label in target_label or target_label in label:
                bonus += 0.20
        if target_hint:
            hint_tokens = self._semantic_tokens(target_hint)
            label_tokens = set(self._semantic_tokens(label))
            overlap = sum(1 for token in hint_tokens if token in label_tokens)
            bonus += min(0.15, 0.05 * overlap)
        alias_labels = self._semantic_alias_labels(target_label, target_hint)
        if any(self._labels_related(label, alias) for alias in alias_labels):
            bonus += 0.25
        if (target_label or target_hint) and bonus == 0.0:
            bonus -= 0.05
        return bonus

    def _semantic_task_requires_match(
        self,
        semantic_task: Optional[SemanticTask],
    ) -> bool:
        if semantic_task is None:
            return False
        return bool(
            str(semantic_task.target_label or "").strip()
            or str(semantic_task.target_hint or "").strip()
        )

    def _candidate_confidence_floor(
        self,
        semantic_bonus: float,
        semantic_task: Optional[SemanticTask],
    ) -> float:
        if self._semantic_task_requires_match(semantic_task) and semantic_bonus > 0.0:
            return min(self.confidence_threshold, self.semantic_match_confidence_floor)
        return self.confidence_threshold

    def _semantic_tokens(self, text: str) -> list[str]:
        normalized = text.replace(",", " ").replace("-", " ").lower()
        return [
            token
            for token in normalized.split()
            if token and token not in COLOR_HINT_TOKENS
        ]

    def _semantic_alias_labels(self, *semantic_texts: str) -> set[str]:
        aliases: set[str] = set()
        for text in semantic_texts:
            for token in self._semantic_tokens(text):
                aliases.update(SEMANTIC_ALIAS_LABELS.get(token, set()))
        return aliases

    def _labels_related(self, detected_label: str, semantic_label: str) -> bool:
        left = detected_label.strip().lower()
        right = semantic_label.strip().lower()
        if not left or not right:
            return False
        return bool(left == right or left in right or right in left)

    def _extract_bbox_xyxy(
        self,
        boxes: Any,
        index: int,
        image_width: int,
        image_height: int,
    ) -> Optional[list[int]]:
        xyxy_tensor = boxes.xyxy[index]
        bbox_xyxy = [
            float(v)
            for v in (
                xyxy_tensor.cpu().numpy().tolist()
                if hasattr(xyxy_tensor, "cpu")
                else list(xyxy_tensor)
            )
        ]
        return coerce_bbox(bbox_xyxy, image_width, image_height)

    def _extract_mask_u8(
        self,
        masks_data: Optional[np.ndarray],
        index: int,
        image_shape: tuple[int, int],
    ) -> Optional[np.ndarray]:
        if masks_data is None or index >= int(masks_data.shape[0]):
            return None
        mask = np.asarray(masks_data[index], dtype=np.float32)
        if mask.shape[:2] != image_shape:
            mask = cv2.resize(
                mask,
                (int(image_shape[1]), int(image_shape[0])),
                interpolation=cv2.INTER_NEAREST,
            )
        return np.where(mask > 0.5, 255, 0).astype(np.uint8)

    def _select_debug_candidates(
        self,
        raw_candidates: list[dict[str, Any]],
        selected_candidate: Optional[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        ranked = sorted(
            raw_candidates,
            key=lambda item: (
                float(item["confidence"]),
                float(item["score"]),
            ),
            reverse=True,
        )
        top_candidates = ranked[: self.debug_top_k_candidates]
        if (
            selected_candidate is not None and
            all(int(item["index"]) != int(selected_candidate["index"]) for item in top_candidates)
        ):
            remaining = [
                item
                for item in ranked
                if int(item["index"]) != int(selected_candidate["index"])
            ]
            top_candidates = [selected_candidate] + remaining[: self.debug_top_k_candidates - 1]
        return top_candidates

    def _candidate_debug_public(self, candidate: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if candidate is None:
            return None
        bbox_xyxy = candidate.get("bbox_xyxy")
        bbox_list = list(bbox_xyxy) if isinstance(bbox_xyxy, list) else None
        return {
            "index": int(candidate.get("index", -1)),
            "label": str(candidate.get("label", "")),
            "confidence": round(float(candidate.get("confidence", 0.0)), 4),
            "confidence_floor": round(float(candidate.get("confidence_floor", 0.0)), 4),
            "semantic_bonus": round(float(candidate.get("semantic_bonus", 0.0)), 4),
            "score": round(float(candidate.get("score", 0.0)), 4),
            "bbox_xyxy": bbox_list,
            "mask_pixels": int(candidate.get("mask_pixels", 0)),
            "status": str(candidate.get("status", "")),
        }

    def _format_candidate_summary(
        self,
        semantic_task: Optional[SemanticTask],
        top_candidates: list[dict[str, Any]],
        selected_candidate: Optional[dict[str, Any]],
    ) -> str:
        target_hint = (
            str(semantic_task.target_hint or "").strip()
            if semantic_task is not None
            else ""
        )
        target_label = (
            str(semantic_task.target_label or "").strip()
            if semantic_task is not None
            else ""
        )
        selected_label = (
            str(selected_candidate.get("label", ""))
            if selected_candidate is not None
            else "<none>"
        )
        parts: list[str] = []
        for rank, candidate in enumerate(top_candidates[: self.debug_top_k_candidates], start=1):
            parts.append(
                f"#{rank}:{str(candidate['label'])}"
                f"(c={float(candidate['confidence']):.3f},"
                f"b={float(candidate['semantic_bonus']):+.3f},"
                f"s={float(candidate['score']):.3f},"
                f"status={str(candidate['status'])})"
            )
        return (
            "detector top-k: "
            f"target_label={target_label or '<none>'} "
            f"target_hint={target_hint or '<none>'} "
            f"selected={selected_label or '<none>'} "
            f"candidates=[{'; '.join(parts)}]"
        )

    def _render_debug_overlay(
        self,
        image_rgb: np.ndarray,
        top_candidates: list[dict[str, Any]],
        selected_candidate: Optional[dict[str, Any]],
        semantic_task: Optional[SemanticTask],
    ) -> np.ndarray:
        canvas = np.asarray(image_rgb, dtype=np.uint8).copy()
        selected_index = int(selected_candidate["index"]) if selected_candidate is not None else -1

        for candidate in top_candidates:
            color = self._candidate_overlay_color(candidate, selected_index)
            mask_u8 = candidate.get("mask_u8")
            if isinstance(mask_u8, np.ndarray) and mask_u8.shape[:2] == canvas.shape[:2]:
                mask = mask_u8 > 0
                if np.any(mask):
                    color_array = np.asarray(color, dtype=np.float32)
                    blended = canvas[mask].astype(np.float32) * 0.65 + color_array * 0.35
                    canvas[mask] = np.clip(blended, 0.0, 255.0).astype(np.uint8)

        for rank, candidate in enumerate(top_candidates[: self.debug_top_k_candidates], start=1):
            bbox_xyxy = candidate.get("bbox_xyxy")
            if not isinstance(bbox_xyxy, list) or len(bbox_xyxy) != 4:
                continue
            x1, y1, x2, y2 = [int(value) for value in bbox_xyxy]
            color = self._candidate_overlay_color(candidate, selected_index)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            label_text = (
                f"#{rank} {str(candidate['label'])} "
                f"c={float(candidate['confidence']):.2f} "
                f"s={float(candidate['score']):.2f}"
            )
            text_y = y1 - 8 if y1 > 20 else y1 + 18
            cv2.putText(
                canvas,
                label_text,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        header_lines = [
            (
                "target="
                f"{str(getattr(semantic_task, 'target_hint', '') or '<none>')}"
                f" selected={str(selected_candidate.get('label', '<none>') if selected_candidate else '<none>')}"
            ),
            "top-k="
            + " | ".join(
                f"{str(candidate['label'])}:{float(candidate['confidence']):.2f}"
                for candidate in top_candidates[: self.debug_top_k_candidates]
            ),
        ]
        bar_height = 24 * len(header_lines) + 8
        cv2.rectangle(
            canvas,
            (0, 0),
            (int(canvas.shape[1]), int(bar_height)),
            (0, 0, 0),
            thickness=-1,
        )
        for line_index, line in enumerate(header_lines, start=1):
            cv2.putText(
                canvas,
                line,
                (8, 20 * line_index),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return canvas

    def _candidate_overlay_color(
        self,
        candidate: dict[str, Any],
        selected_index: int,
    ) -> tuple[int, int, int]:
        if int(candidate.get("index", -1)) == selected_index:
            return (32, 255, 96)
        status = str(candidate.get("status", ""))
        if status == "filtered_low_confidence":
            return (180, 180, 180)
        if status == "filtered_excluded_label":
            return (255, 96, 96)
        return (255, 176, 64)

    def _bbox_center_point(self, bbox_xyxy: Optional[list[int]]) -> Optional[list[int]]:
        if bbox_xyxy is None:
            return None
        return [
            int(round((bbox_xyxy[0] + bbox_xyxy[2]) * 0.5)),
            int(round((bbox_xyxy[1] + bbox_xyxy[3]) * 0.5)),
        ]

    def _is_candidate_complete(
        self,
        bbox_xyxy: Optional[list[int]],
        image_width: int,
        image_height: int,
    ) -> bool:
        if bbox_xyxy is None:
            return False
        margin = self.candidate_complete_edge_margin_px
        x1, y1, x2, y2 = bbox_xyxy
        return bool(
            x1 > margin
            and y1 > margin
            and x2 < (image_width - margin)
            and y2 < (image_height - margin)
        )

    def _build_detection_payload(
        self,
        *,
        backend: str,
        task: str,
        target_label: str,
        confidence: float,
        need_human_confirm: bool,
        reason: str,
        image_width: int,
        image_height: int,
        frame_id: str,
        bbox_xyxy: Optional[list[int]],
        point_px: Optional[list[int]],
        mask_u8: Optional[np.ndarray],
    ) -> dict[str, Any]:
        accepted = bool(target_label and bbox_xyxy is not None and confidence > 0.0)
        return {
            "backend": backend,
            "accepted": accepted,
            "candidate_visible": accepted,
            "candidate_complete": self._is_candidate_complete(
                bbox_xyxy, image_width, image_height
            ),
            "task": str(task or "pick"),
            "target_label": str(target_label or ""),
            "confidence": max(0.0, min(1.0, float(confidence))),
            "need_human_confirm": bool(need_human_confirm),
            "reason": str(reason or ""),
            "image_width": int(image_width),
            "image_height": int(image_height),
            "frame_id": str(frame_id or ""),
            "bbox_xyxy": bbox_xyxy,
            "point_px": point_px,
            "mask_u8": mask_u8,
        }

    def _build_empty_result(
        self,
        *,
        task: str,
        target_label: str,
        reason: str,
        image_width: int = 0,
        image_height: int = 0,
        frame_id: str = "",
    ) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "accepted": False,
            "candidate_visible": False,
            "candidate_complete": False,
            "task": str(task or "pick"),
            "target_label": str(target_label or ""),
            "confidence": 0.0,
            "need_human_confirm": True,
            "reason": str(reason or ""),
            "image_width": int(image_width),
            "image_height": int(image_height),
            "frame_id": str(frame_id or ""),
            "bbox_xyxy": None,
            "point_px": None,
            "mask_u8": None,
        }

    def _publish_detection(self, payload: dict[str, Any]) -> None:
        detection_msg = DetectionResult()
        detection_msg.header.stamp = self.get_clock().now().to_msg()
        detection_msg.header.frame_id = str(payload.get("frame_id", "") or "")
        detection_msg.backend = str(payload.get("backend", ""))
        detection_msg.accepted = bool(payload.get("accepted", False))
        detection_msg.candidate_visible = bool(payload.get("candidate_visible", False))
        detection_msg.candidate_complete = bool(payload.get("candidate_complete", False))
        detection_msg.task = str(payload.get("task", ""))
        detection_msg.target_label = str(payload.get("target_label", ""))
        detection_msg.confidence = float(payload.get("confidence", 0.0))
        detection_msg.need_human_confirm = bool(payload.get("need_human_confirm", False))
        detection_msg.reason = str(payload.get("reason", ""))
        detection_msg.image_width = int(payload.get("image_width", 0))
        detection_msg.image_height = int(payload.get("image_height", 0))

        bbox_xyxy = payload.get("bbox_xyxy")
        if isinstance(bbox_xyxy, list) and len(bbox_xyxy) == 4:
            detection_msg.has_bbox = True
            roi = RegionOfInterest()
            roi.x_offset = int(bbox_xyxy[0])
            roi.y_offset = int(bbox_xyxy[1])
            roi.width = max(0, int(bbox_xyxy[2]) - int(bbox_xyxy[0]))
            roi.height = max(0, int(bbox_xyxy[3]) - int(bbox_xyxy[1]))
            roi.do_rectify = False
            detection_msg.bbox = roi
        else:
            detection_msg.has_bbox = False

        point_px = payload.get("point_px")
        if isinstance(point_px, list) and len(point_px) == 2:
            detection_msg.has_point = True
            detection_msg.point_px = [int(point_px[0]), int(point_px[1])]
        else:
            detection_msg.has_point = False
            detection_msg.point_px = [0, 0]

        mask_u8 = payload.get("mask_u8")
        if isinstance(mask_u8, np.ndarray) and mask_u8.size > 0:
            detection_msg.has_mask = True
            detection_msg.mask = make_mono8_image(
                mask_u8,
                frame_id=detection_msg.header.frame_id,
                stamp=detection_msg.header.stamp,
            )
        else:
            detection_msg.has_mask = False
            detection_msg.mask = Image()

        self.detection_pub.publish(detection_msg)
        self.candidate_visible_pub.publish(
            Bool(data=bool(payload.get("candidate_visible", False)))
        )

        debug_payload = {
            "backend": detection_msg.backend,
            "accepted": detection_msg.accepted,
            "label": detection_msg.target_label,
            "confidence": round(float(detection_msg.confidence), 4),
            "has_bbox": bool(detection_msg.has_bbox),
            "has_mask": bool(detection_msg.has_mask),
            "reason": detection_msg.reason,
            "semantic_target": payload.get("semantic_target", {}),
            "selected_candidate": payload.get("selected_candidate"),
            "top_candidates": payload.get("debug_candidates", []),
        }
        debug_msg = String()
        debug_msg.data = compact_json(debug_payload)
        self.debug_pub.publish(debug_msg)

        if self.debug_overlay_pub is not None:
            debug_overlay_rgb = payload.get("debug_overlay_rgb")
            if isinstance(debug_overlay_rgb, np.ndarray) and debug_overlay_rgb.size > 0:
                self.debug_overlay_pub.publish(
                    make_rgb8_image(
                        np.asarray(debug_overlay_rgb, dtype=np.uint8),
                        frame_id=detection_msg.header.frame_id,
                        stamp=detection_msg.header.stamp,
                    )
                )

        summary = (
            "detector decision: "
            f"accepted={detection_msg.accepted} "
            f"label={detection_msg.target_label or '<none>'} "
            f"visible={bool(payload.get('candidate_visible', False))} "
            f"mask={detection_msg.has_mask}"
        )
        if summary != self._last_terminal_summary:
            self._last_terminal_summary = summary
            self.get_logger().info(summary)

        candidate_summary = str(payload.get("candidate_summary", "") or "")
        if candidate_summary and candidate_summary != self._last_candidate_summary:
            self._last_candidate_summary = candidate_summary
            self.get_logger().info(candidate_summary)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DetectorSegNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
