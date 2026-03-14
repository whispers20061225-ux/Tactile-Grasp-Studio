from __future__ import annotations

import json
import threading
import time
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
)


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
        self.declare_parameter("backend", "ultralytics_local")
        self.declare_parameter("backend_url", "")
        self.declare_parameter("request_timeout_sec", 10.0)
        self.declare_parameter("max_inference_rate_hz", 2.0)
        self.declare_parameter("enabled", True)
        self.declare_parameter("ultralytics_model_path", "yolo11s-seg.pt")
        self.declare_parameter("ultralytics_device", "")
        self.declare_parameter("ultralytics_imgsz", 960)
        self.declare_parameter("confidence_threshold", 0.25)
        self.declare_parameter("iou_threshold", 0.55)
        self.declare_parameter("candidate_complete_edge_margin_px", 16)
        self.declare_parameter("jpeg_quality", 90)
        self.declare_parameter("log_interval_sec", 10.0)

        self.color_topic = str(self.get_parameter("color_topic").value)
        self.semantic_task_topic = str(self.get_parameter("semantic_task_topic").value)
        self.detection_result_topic = str(self.get_parameter("detection_result_topic").value)
        self.candidate_visible_topic = str(self.get_parameter("candidate_visible_topic").value)
        self.detection_debug_topic = str(self.get_parameter("detection_debug_topic").value)
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
        self.log_interval_sec = max(1.0, float(self.get_parameter("log_interval_sec").value))

        self._session = requests.Session()
        self._request_lock = threading.Lock()
        self._semantic_lock = threading.Lock()
        self._color_lock = threading.Lock()
        self._request_in_flight = False
        self._pending_request = False
        self._latest_color_msg: Optional[Image] = None
        self._semantic_task: Optional[SemanticTask] = None
        self._last_log_ts = 0.0
        self._model = None

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
        if self.backend in ("http_json", "ultralytics_local"):
            self.create_timer(1.0 / self.max_inference_rate_hz, self._maybe_run_inference)
        if self.backend == "ultralytics_local":
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
            f"model={self.ultralytics_model_path}, device={self.ultralytics_device or 'auto'}"
        )

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

        worker = threading.Thread(
            target=self._run_inference,
            args=(color_msg, semantic_task),
            daemon=True,
        )
        worker.start()

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

        self._model = YOLO(self.ultralytics_model_path)
        return self._model

    def _run_ultralytics_inference(
        self,
        color_msg: Image,
        image_rgb: np.ndarray,
        semantic_task: Optional[SemanticTask],
    ) -> dict[str, Any]:
        model = self._ensure_ultralytics_model()
        prediction = model.predict(
            source=image_rgb,
            imgsz=self.ultralytics_imgsz,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.ultralytics_device or None,
            verbose=False,
        )
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

        best_idx = -1
        best_score = -1.0
        best_label = ""
        for idx in range(len(boxes)):
            cls_tensor = boxes.cls[idx]
            conf_tensor = boxes.conf[idx]
            cls_id = int(cls_tensor.item() if hasattr(cls_tensor, "item") else cls_tensor)
            confidence = float(
                conf_tensor.item() if hasattr(conf_tensor, "item") else conf_tensor
            )
            label = str(names.get(cls_id, cls_id)) if isinstance(names, dict) else str(names[cls_id])
            semantic_bonus = self._semantic_match_bonus(label, semantic_task)
            if semantic_bonus <= -1.0:
                continue
            score = confidence + semantic_bonus
            if score > best_score:
                best_score = score
                best_idx = idx
                best_label = label

        if best_idx < 0:
            return self._build_empty_result(
                task=semantic_task.task if semantic_task is not None else "pick",
                target_label=semantic_task.target_label if semantic_task is not None else "",
                reason="YOLO11-seg found detections, but none satisfied semantic filters",
                image_width=int(color_msg.width),
                image_height=int(color_msg.height),
                frame_id=color_msg.header.frame_id,
            )

        xyxy_tensor = boxes.xyxy[best_idx]
        bbox_xyxy = [
            float(v)
            for v in (
                xyxy_tensor.cpu().numpy().tolist()
                if hasattr(xyxy_tensor, "cpu")
                else list(xyxy_tensor)
            )
        ]
        bbox_xyxy = coerce_bbox(bbox_xyxy, int(color_msg.width), int(color_msg.height))
        point_px = self._bbox_center_point(bbox_xyxy) if bbox_xyxy is not None else None
        conf_tensor = boxes.conf[best_idx]
        confidence = float(conf_tensor.item() if hasattr(conf_tensor, "item") else conf_tensor)

        mask_u8 = None
        if masks_data is not None and best_idx < int(masks_data.shape[0]):
            mask = np.asarray(masks_data[best_idx], dtype=np.float32)
            if mask.shape[:2] != image_rgb.shape[:2]:
                mask = cv2.resize(
                    mask,
                    (image_rgb.shape[1], image_rgb.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            mask_u8 = np.where(mask > 0.5, 255, 0).astype(np.uint8)

        return self._build_detection_payload(
            backend=f"ultralytics_local:{self.ultralytics_model_path}",
            task=semantic_task.task if semantic_task is not None else "pick",
            target_label=best_label,
            confidence=confidence,
            need_human_confirm=False,
            reason="",
            image_width=int(color_msg.width),
            image_height=int(color_msg.height),
            frame_id=color_msg.header.frame_id,
            bbox_xyxy=bbox_xyxy,
            point_px=point_px,
            mask_u8=mask_u8,
        )

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
            hint_tokens = [token for token in target_hint.replace(",", " ").split() if token]
            label_tokens = set(label.replace(",", " ").split())
            overlap = sum(1 for token in hint_tokens if token in label_tokens)
            bonus += min(0.15, 0.05 * overlap)
        return bonus

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
        }
        debug_msg = String()
        debug_msg.data = compact_json(debug_payload)
        self.debug_pub.publish(debug_msg)

        now = time.time()
        if now - self._last_log_ts >= self.log_interval_sec:
            self._last_log_ts = now
            self.get_logger().info(
                "detector result: "
                f"backend={detection_msg.backend} "
                f"accepted={detection_msg.accepted} "
                f"label={detection_msg.target_label} "
                f"has_mask={detection_msg.has_mask}"
            )


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
