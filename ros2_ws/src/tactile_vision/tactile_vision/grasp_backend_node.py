from __future__ import annotations

import threading
import time
from typing import Any, Optional

import cv2
import numpy as np
import requests
import rclpy
from geometry_msgs.msg import Point, Vector3
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_msgs.msg import String
from tactile_interfaces.msg import DetectionResult, GraspProposal, GraspProposalArray, SemanticTask

from tactile_vision.modular_common import (
    compact_json,
    decode_depth_image,
    decode_mono8_image,
    encode_depth_m_to_base64_png,
    encode_png_base64,
)


def point_cloud2_to_xyz_array(msg: PointCloud2) -> np.ndarray:
    if int(msg.point_step) < 12 or int(msg.width) <= 0:
        return np.empty((0, 3), dtype=np.float32)
    point_step_floats = max(3, int(msg.point_step) // 4)
    points = np.frombuffer(msg.data, dtype=np.float32)
    if points.size < int(msg.width) * point_step_floats:
        return np.empty((0, 3), dtype=np.float32)
    points = points.reshape((-1, point_step_floats))
    return points[:, :3].astype(np.float32)


def to_point(values: Any) -> Point:
    point = Point()
    if isinstance(values, (list, tuple)) and len(values) >= 3:
        point.x = float(values[0])
        point.y = float(values[1])
        point.z = float(values[2])
    return point


def to_vector3(values: Any) -> Vector3:
    vector = Vector3()
    if isinstance(values, (list, tuple)) and len(values) >= 3:
        vector.x = float(values[0])
        vector.y = float(values[1])
        vector.z = float(values[2])
    return vector


class GraspBackendNode(Node):
    def __init__(self) -> None:
        super().__init__("grasp_backend_node")

        self.declare_parameter("semantic_task_topic", "/qwen/semantic_task")
        self.declare_parameter("target_cloud_topic", "/perception/target_cloud")
        self.declare_parameter("detection_result_topic", "/perception/detection_result")
        self.declare_parameter(
            "depth_topic", "/camera/camera/aligned_depth_to_color/image_raw"
        )
        self.declare_parameter(
            "camera_info_topic", "/camera/camera/aligned_depth_to_color/camera_info"
        )
        self.declare_parameter(
            "candidate_grasp_proposal_array_topic", "/grasp/candidate_grasp_proposals"
        )
        self.declare_parameter("backend", "disabled")
        self.declare_parameter("backend_url", "")
        self.declare_parameter("request_timeout_sec", 10.0)
        self.declare_parameter("max_inference_rate_hz", 1.0)
        self.declare_parameter("sensor_stale_sec", 1.5)
        self.declare_parameter("max_points", 2048)
        self.declare_parameter("min_points_required", 64)
        self.declare_parameter("publish_empty_on_failure", False)
        self.declare_parameter("default_task_constraint_tag", "pick")
        self.declare_parameter("contact_graspnet_segmap_id", 1)
        self.declare_parameter("contact_graspnet_local_regions", True)
        self.declare_parameter("contact_graspnet_filter_grasps", True)
        self.declare_parameter("contact_graspnet_skip_border_objects", False)
        self.declare_parameter("contact_graspnet_forward_passes", 1)
        self.declare_parameter("contact_graspnet_z_range", [0.2, 1.8])
        self.declare_parameter("contact_graspnet_pregrasp_offset_m", 0.06)
        self.declare_parameter("contact_graspnet_max_proposals", 0)
        self.declare_parameter("contact_graspnet_min_mask_pixels", 96)
        self.declare_parameter("contact_graspnet_visualize", False)
        self.declare_parameter("log_interval_sec", 8.0)

        self.semantic_task_topic = str(self.get_parameter("semantic_task_topic").value)
        self.target_cloud_topic = str(self.get_parameter("target_cloud_topic").value)
        self.detection_result_topic = str(self.get_parameter("detection_result_topic").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        self.candidate_grasp_proposal_array_topic = str(
            self.get_parameter("candidate_grasp_proposal_array_topic").value
        )
        self.backend = str(self.get_parameter("backend").value).strip().lower()
        self.backend_url = str(self.get_parameter("backend_url").value).strip()
        self.request_timeout_sec = max(
            0.5, float(self.get_parameter("request_timeout_sec").value)
        )
        self.max_inference_rate_hz = max(
            0.2, float(self.get_parameter("max_inference_rate_hz").value)
        )
        self.sensor_stale_sec = max(0.1, float(self.get_parameter("sensor_stale_sec").value))
        self.max_points = max(64, int(self.get_parameter("max_points").value))
        self.min_points_required = max(
            8, int(self.get_parameter("min_points_required").value)
        )
        self.publish_empty_on_failure = bool(
            self.get_parameter("publish_empty_on_failure").value
        )
        self.default_task_constraint_tag = str(
            self.get_parameter("default_task_constraint_tag").value
        ).strip()
        self.contact_graspnet_segmap_id = max(
            1, int(self.get_parameter("contact_graspnet_segmap_id").value)
        )
        self.contact_graspnet_local_regions = bool(
            self.get_parameter("contact_graspnet_local_regions").value
        )
        self.contact_graspnet_filter_grasps = bool(
            self.get_parameter("contact_graspnet_filter_grasps").value
        )
        self.contact_graspnet_skip_border_objects = bool(
            self.get_parameter("contact_graspnet_skip_border_objects").value
        )
        self.contact_graspnet_forward_passes = max(
            1, int(self.get_parameter("contact_graspnet_forward_passes").value)
        )
        z_range_values = list(self.get_parameter("contact_graspnet_z_range").value)
        if len(z_range_values) >= 2:
            z0 = float(z_range_values[0])
            z1 = float(z_range_values[1])
        else:
            z0, z1 = 0.2, 1.8
        self.contact_graspnet_z_range = [min(z0, z1), max(z0, z1)]
        self.contact_graspnet_pregrasp_offset_m = max(
            0.01, float(self.get_parameter("contact_graspnet_pregrasp_offset_m").value)
        )
        self.contact_graspnet_max_proposals = max(
            0, int(self.get_parameter("contact_graspnet_max_proposals").value)
        )
        self.contact_graspnet_min_mask_pixels = max(
            16, int(self.get_parameter("contact_graspnet_min_mask_pixels").value)
        )
        self.contact_graspnet_visualize = bool(
            self.get_parameter("contact_graspnet_visualize").value
        )
        self.log_interval_sec = max(1.0, float(self.get_parameter("log_interval_sec").value))

        self._session = requests.Session()
        self._semantic_lock = threading.Lock()
        self._sensor_lock = threading.Lock()
        self._request_lock = threading.Lock()
        self._semantic_task: Optional[SemanticTask] = None
        self._latest_target_cloud: Optional[PointCloud2] = None
        self._latest_target_cloud_ts = 0.0
        self._latest_detection: Optional[DetectionResult] = None
        self._latest_detection_ts = 0.0
        self._latest_depth_msg: Optional[Image] = None
        self._latest_depth_ts = 0.0
        self._latest_camera_info: Optional[CameraInfo] = None
        self._latest_camera_info_ts = 0.0
        self._pending_request = False
        self._request_in_flight = False
        self._last_log_ts = 0.0

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

        self.create_subscription(
            SemanticTask, self.semantic_task_topic, self._on_semantic_task, qos_reliable
        )
        self.create_subscription(
            PointCloud2, self.target_cloud_topic, self._on_target_cloud, qos_reliable
        )
        self.create_subscription(
            DetectionResult, self.detection_result_topic, self._on_detection_result, qos_reliable
        )
        self.create_subscription(Image, self.depth_topic, self._on_depth_image, qos_sensor)
        self.create_subscription(
            CameraInfo, self.camera_info_topic, self._on_camera_info, qos_sensor
        )
        self.proposal_pub = self.create_publisher(
            GraspProposalArray, self.candidate_grasp_proposal_array_topic, qos_reliable
        )
        self.debug_pub = self.create_publisher(String, "/grasp/backend_debug", qos_reliable)
        self.create_timer(1.0 / self.max_inference_rate_hz, self._maybe_run_inference)

        self.get_logger().info(
            "grasp_backend_node started: "
            f"backend={self.backend}, target_cloud={self.target_cloud_topic}, "
            f"detection={self.detection_result_topic}, proposal_topic={self.candidate_grasp_proposal_array_topic}"
        )

    def _on_semantic_task(self, msg: SemanticTask) -> None:
        with self._semantic_lock:
            self._semantic_task = msg
        self._pending_request = True

    def _on_target_cloud(self, msg: PointCloud2) -> None:
        with self._sensor_lock:
            self._latest_target_cloud = msg
            self._latest_target_cloud_ts = time.time()
        self._pending_request = True

    def _on_detection_result(self, msg: DetectionResult) -> None:
        with self._sensor_lock:
            self._latest_detection = msg
            self._latest_detection_ts = time.time()
        self._pending_request = True

    def _on_depth_image(self, msg: Image) -> None:
        with self._sensor_lock:
            self._latest_depth_msg = msg
            self._latest_depth_ts = time.time()
        self._pending_request = True

    def _on_camera_info(self, msg: CameraInfo) -> None:
        with self._sensor_lock:
            self._latest_camera_info = msg
            self._latest_camera_info_ts = time.time()
        self._pending_request = True

    def _maybe_run_inference(self) -> None:
        if not self._pending_request:
            return

        if self.backend == "disabled":
            self._pending_request = False
            self._publish_debug(
                {
                    "status": "backend_disabled",
                    "reason": "configure Contact-GraspNet service endpoint",
                }
            )
            return

        with self._request_lock:
            if self._request_in_flight:
                return
            self._request_in_flight = True
            self._pending_request = False

        worker = threading.Thread(target=self._run_backend, daemon=True)
        worker.start()

    def _run_backend(self) -> None:
        try:
            with self._semantic_lock:
                semantic_task = self._semantic_task
            with self._sensor_lock:
                target_cloud = self._latest_target_cloud
                target_cloud_ts = self._latest_target_cloud_ts
                detection = self._latest_detection
                detection_ts = self._latest_detection_ts
                depth_msg = self._latest_depth_msg
                depth_ts = self._latest_depth_ts
                camera_info = self._latest_camera_info
                camera_info_ts = self._latest_camera_info_ts

            if self.backend == "http_json":
                parsed = self._run_point_cloud_http_backend(
                    semantic_task=semantic_task,
                    cloud_msg=target_cloud,
                    cloud_ts=target_cloud_ts,
                )
            elif self.backend == "contact_graspnet_http":
                parsed = self._run_contact_graspnet_http_backend(
                    semantic_task=semantic_task,
                    detection=detection,
                    detection_ts=detection_ts,
                    depth_msg=depth_msg,
                    depth_ts=depth_ts,
                    camera_info=camera_info,
                    camera_info_ts=camera_info_ts,
                )
            else:
                raise ValueError(f"unsupported grasp backend: {self.backend}")

            proposal_array = self._parse_proposal_array(parsed)
            self.proposal_pub.publish(proposal_array)
            self._publish_debug(
                {
                    "status": "ok",
                    "backend": self.backend,
                    "proposal_count": len(proposal_array.proposals),
                    "frame_id": proposal_array.header.frame_id,
                }
            )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"grasp backend inference failed: {exc}")
            self._publish_debug({"status": "error", "backend": self.backend, "reason": str(exc)})
            if self.publish_empty_on_failure:
                empty = GraspProposalArray()
                self.proposal_pub.publish(empty)
        finally:
            with self._request_lock:
                self._request_in_flight = False

    def _run_point_cloud_http_backend(
        self,
        *,
        semantic_task: Optional[SemanticTask],
        cloud_msg: Optional[PointCloud2],
        cloud_ts: float,
    ) -> dict[str, Any]:
        if not self.backend_url:
            raise ValueError("grasp backend_url is empty")
        if cloud_msg is None:
            raise ValueError("missing target cloud for http_json grasp backend")
        if time.time() - cloud_ts > self.sensor_stale_sec:
            raise ValueError("target cloud is stale")

        points = point_cloud2_to_xyz_array(cloud_msg)
        if points.shape[0] < self.min_points_required:
            raise ValueError(
                f"target cloud has too few points: {points.shape[0]} < {self.min_points_required}"
            )
        if points.shape[0] > self.max_points:
            step = max(1, points.shape[0] // self.max_points)
            points = points[::step][: self.max_points]

        payload = {
            "frame_id": cloud_msg.header.frame_id,
            "points_xyz": points.astype(float).tolist(),
            "task": semantic_task.task if semantic_task is not None else "pick",
            "target_label": semantic_task.target_label if semantic_task is not None else "",
            "target_hint": semantic_task.target_hint if semantic_task is not None else "",
            "constraints": list(semantic_task.constraints) if semantic_task is not None else [],
        }
        response = self._session.post(
            self.backend_url,
            json=payload,
            timeout=self.request_timeout_sec,
        )
        response.raise_for_status()
        parsed = response.json()
        if not isinstance(parsed, dict):
            raise ValueError("grasp backend response is not a JSON object")
        return parsed

    def _run_contact_graspnet_http_backend(
        self,
        *,
        semantic_task: Optional[SemanticTask],
        detection: Optional[DetectionResult],
        detection_ts: float,
        depth_msg: Optional[Image],
        depth_ts: float,
        camera_info: Optional[CameraInfo],
        camera_info_ts: float,
    ) -> dict[str, Any]:
        if not self.backend_url:
            raise ValueError("grasp backend_url is empty")
        if detection is None or depth_msg is None or camera_info is None:
            raise ValueError("missing detection/depth/camera info for Contact-GraspNet backend")
        now = time.time()
        if now - detection_ts > self.sensor_stale_sec:
            raise ValueError("detection result is stale")
        if now - depth_ts > self.sensor_stale_sec:
            raise ValueError("depth image is stale")
        if now - camera_info_ts > self.sensor_stale_sec:
            raise ValueError("camera info is stale")
        if not bool(detection.accepted):
            raise ValueError(str(detection.reason or "detection_not_accepted"))
        if not bool(detection.has_mask):
            raise ValueError("Contact-GraspNet backend requires an instance mask")

        depth_image = decode_depth_image(depth_msg)
        if depth_image is None:
            raise ValueError(f"unsupported depth image encoding: {depth_msg.encoding}")

        mask_image = decode_mono8_image(detection.mask)
        if mask_image is None:
            raise ValueError("failed to decode detection mask")
        if mask_image.shape[:2] != depth_image.shape[:2]:
            mask_image = cv2.resize(
                mask_image,
                (depth_image.shape[1], depth_image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        segmap = np.where(mask_image > 0, self.contact_graspnet_segmap_id, 0).astype(np.uint8)
        mask_pixels = int(np.count_nonzero(segmap))
        if mask_pixels < self.contact_graspnet_min_mask_pixels:
            raise ValueError(
                f"mask too sparse for Contact-GraspNet: {mask_pixels} < {self.contact_graspnet_min_mask_pixels}"
            )

        if len(camera_info.k) < 9:
            raise ValueError("camera info K matrix is incomplete")
        k_matrix = [
            float(camera_info.k[0]), float(camera_info.k[1]), float(camera_info.k[2]),
            float(camera_info.k[3]), float(camera_info.k[4]), float(camera_info.k[5]),
            float(camera_info.k[6]), float(camera_info.k[7]), float(camera_info.k[8]),
        ]

        payload: dict[str, Any] = {
            "frame_id": camera_info.header.frame_id or depth_msg.header.frame_id or detection.header.frame_id,
            "task": semantic_task.task if semantic_task is not None else "pick",
            "target_label": semantic_task.target_label if semantic_task is not None else "",
            "target_hint": semantic_task.target_hint if semantic_task is not None else "",
            "constraints": list(semantic_task.constraints) if semantic_task is not None else [],
            "excluded_labels": list(semantic_task.excluded_labels) if semantic_task is not None else [],
            "depth_png_b64": encode_depth_m_to_base64_png(depth_image),
            "depth_scale_m_per_unit": 0.001,
            "segmap_png_b64": encode_png_base64(segmap),
            "segmap_id": self.contact_graspnet_segmap_id,
            "image_width": int(depth_image.shape[1]),
            "image_height": int(depth_image.shape[0]),
            "K": k_matrix,
            "local_regions": self.contact_graspnet_local_regions,
            "filter_grasps": self.contact_graspnet_filter_grasps,
            "skip_border_objects": self.contact_graspnet_skip_border_objects,
            "forward_passes": self.contact_graspnet_forward_passes,
            "z_range": list(self.contact_graspnet_z_range),
            "pregrasp_offset_m": self.contact_graspnet_pregrasp_offset_m,
            "max_proposals": self.contact_graspnet_max_proposals,
            "task_constraint_tag": self.default_task_constraint_tag,
            "visualize": self.contact_graspnet_visualize,
        }
        if bool(detection.has_bbox):
            payload["bbox_xyxy"] = [
                int(detection.bbox.x_offset),
                int(detection.bbox.y_offset),
                int(detection.bbox.x_offset + detection.bbox.width),
                int(detection.bbox.y_offset + detection.bbox.height),
            ]

        response = self._session.post(
            self.backend_url,
            json=payload,
            timeout=self.request_timeout_sec,
        )
        response.raise_for_status()
        parsed = response.json()
        if not isinstance(parsed, dict):
            raise ValueError("Contact-GraspNet response is not a JSON object")
        parsed.setdefault(
            "debug",
            {
                "mask_pixels": mask_pixels,
                "segmap_id": self.contact_graspnet_segmap_id,
                "visualize": self.contact_graspnet_visualize,
            },
        )
        return parsed

    def _parse_proposal_array(self, parsed: dict[str, Any]) -> GraspProposalArray:
        proposals_raw = parsed.get("proposals", [])
        if not isinstance(proposals_raw, list):
            raise ValueError("grasp backend response missing proposals array")

        proposal_array = GraspProposalArray()
        proposal_array.header.frame_id = str(parsed.get("frame_id") or "")
        for item in proposals_raw:
            if not isinstance(item, dict):
                continue
            proposal = GraspProposal()
            proposal.header.frame_id = str(item.get("frame_id") or proposal_array.header.frame_id)
            proposal.contact_point_1 = to_point(item.get("contact_point_1"))
            proposal.contact_point_2 = to_point(item.get("contact_point_2"))
            proposal.grasp_center = to_point(item.get("grasp_center"))
            proposal.closing_direction = to_vector3(item.get("closing_direction"))
            proposal.approach_direction = to_vector3(item.get("approach_direction"))
            grasp_position = to_point(item.get("grasp_pose_position"))
            proposal.grasp_pose.position.x = grasp_position.x
            proposal.grasp_pose.position.y = grasp_position.y
            proposal.grasp_pose.position.z = grasp_position.z
            pregrasp_position = to_point(item.get("pregrasp_pose_position"))
            proposal.pregrasp_pose.position.x = pregrasp_position.x
            proposal.pregrasp_pose.position.y = pregrasp_position.y
            proposal.pregrasp_pose.position.z = pregrasp_position.z

            grasp_orientation = item.get("grasp_pose_orientation")
            if isinstance(grasp_orientation, (list, tuple)) and len(grasp_orientation) >= 4:
                proposal.grasp_pose.orientation.x = float(grasp_orientation[0])
                proposal.grasp_pose.orientation.y = float(grasp_orientation[1])
                proposal.grasp_pose.orientation.z = float(grasp_orientation[2])
                proposal.grasp_pose.orientation.w = float(grasp_orientation[3])
            else:
                proposal.grasp_pose.orientation.w = 1.0

            pregrasp_orientation = item.get("pregrasp_pose_orientation")
            if isinstance(pregrasp_orientation, (list, tuple)) and len(pregrasp_orientation) >= 4:
                proposal.pregrasp_pose.orientation.x = float(pregrasp_orientation[0])
                proposal.pregrasp_pose.orientation.y = float(pregrasp_orientation[1])
                proposal.pregrasp_pose.orientation.z = float(pregrasp_orientation[2])
                proposal.pregrasp_pose.orientation.w = float(pregrasp_orientation[3])
            else:
                proposal.pregrasp_pose.orientation.x = proposal.grasp_pose.orientation.x
                proposal.pregrasp_pose.orientation.y = proposal.grasp_pose.orientation.y
                proposal.pregrasp_pose.orientation.z = proposal.grasp_pose.orientation.z
                proposal.pregrasp_pose.orientation.w = proposal.grasp_pose.orientation.w

            proposal.grasp_width_m = float(item.get("grasp_width_m", 0.0))
            proposal.confidence_score = float(item.get("confidence_score", 0.0))
            proposal.semantic_score = float(item.get("semantic_score", 0.0))
            proposal.candidate_rank = int(item.get("candidate_rank", 0))
            proposal.task_constraint_tag = str(
                item.get("task_constraint_tag") or self.default_task_constraint_tag
            )
            proposal_array.proposals.append(proposal)

        return proposal_array

    def _publish_debug(self, payload: dict[str, Any]) -> None:
        msg = String()
        msg.data = compact_json(payload)
        self.debug_pub.publish(msg)
        now = time.time()
        if now - self._last_log_ts >= self.log_interval_sec:
            self._last_log_ts = now
            self.get_logger().info(f"grasp backend: {msg.data}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GraspBackendNode()
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
