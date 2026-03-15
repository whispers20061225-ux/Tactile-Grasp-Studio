#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import math
import subprocess
import tempfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2
import numpy as np


def decode_png_base64(payload_b64: str, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    encoded = base64.b64decode(payload_b64)
    image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), flags)
    if image is None:
        raise ValueError("failed to decode PNG payload")
    return image


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def normalize_vector(values: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape((3,))
    norm = np.linalg.norm(vector)
    if norm <= 1e-8:
        return np.asarray(fallback, dtype=np.float64).reshape((3,))
    return vector / norm


def rotation_matrix_to_quaternion_xyzw(rotation: np.ndarray) -> list[float]:
    matrix = np.asarray(rotation, dtype=np.float64).reshape((3, 3))
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (matrix[2, 1] - matrix[1, 2]) / s
        y = (matrix[0, 2] - matrix[2, 0]) / s
        z = (matrix[1, 0] - matrix[0, 1]) / s
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        s = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / s
        x = 0.25 * s
        y = (matrix[0, 1] + matrix[1, 0]) / s
        z = (matrix[0, 2] + matrix[2, 0]) / s
    elif matrix[1, 1] > matrix[2, 2]:
        s = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / s
        x = (matrix[0, 1] + matrix[1, 0]) / s
        y = 0.25 * s
        z = (matrix[1, 2] + matrix[2, 1]) / s
    else:
        s = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / s
        x = (matrix[0, 2] + matrix[2, 0]) / s
        y = (matrix[1, 2] + matrix[2, 1]) / s
        z = 0.25 * s

    quat = np.asarray([x, y, z, w], dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm <= 1e-8:
        return [0.0, 0.0, 0.0, 1.0]
    quat /= norm
    return quat.astype(float).tolist()


def load_npz_object(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.dtype == object:
        if value.shape == ():
            return value.item()
        if value.size == 1:
            return value.reshape(()).item()
    return value


def confidence_from_score(score: float) -> float:
    if 0.0 <= score <= 1.0:
        return float(score)
    return float(1.0 / (1.0 + math.exp(-score)))


def segment_key_matches(key: Any, preferred_segmap_id: int) -> bool:
    if key == preferred_segmap_id:
        return True
    try:
        return int(round(float(key))) == int(preferred_segmap_id)
    except (TypeError, ValueError):
        normalized = str(key).strip()
        return normalized in {str(preferred_segmap_id), f"{preferred_segmap_id}.0"}


class ContactGraspNetService:
    def __init__(
        self,
        *,
        contact_graspnet_root: Path,
        checkpoint_dir: Path,
        python_bin: str,
        default_pregrasp_offset_m: float,
        default_max_proposals: int,
        default_z_range: tuple[float, float],
        default_forward_passes: int,
        default_visualize: bool,
    ) -> None:
        self.contact_graspnet_root = contact_graspnet_root
        self.checkpoint_dir = checkpoint_dir
        self.python_bin = python_bin
        self.default_pregrasp_offset_m = default_pregrasp_offset_m
        self.default_max_proposals = default_max_proposals
        self.default_z_range = default_z_range
        self.default_forward_passes = default_forward_passes
        self.default_visualize = default_visualize
        self.inference_module_dir = contact_graspnet_root / "contact_graspnet"
        self.inference_script = self.inference_module_dir / "inference.py"
        if not self.inference_script.is_file():
            raise FileNotFoundError(f"inference.py not found at {self.inference_script}")

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        depth_png_b64 = str(payload.get("depth_png_b64") or "").strip()
        segmap_png_b64 = str(payload.get("segmap_png_b64") or "").strip()
        if not depth_png_b64 or not segmap_png_b64:
            raise ValueError("depth_png_b64 and segmap_png_b64 are required")

        depth_scale = float(payload.get("depth_scale_m_per_unit", 0.001) or 0.001)
        depth_raw = decode_png_base64(depth_png_b64, cv2.IMREAD_UNCHANGED)
        segmap_raw = decode_png_base64(segmap_png_b64, cv2.IMREAD_UNCHANGED)
        if depth_raw.ndim != 2:
            raise ValueError("depth image must be single-channel")
        if segmap_raw.ndim == 3:
            segmap_raw = segmap_raw[:, :, 0]
        if segmap_raw.shape[:2] != depth_raw.shape[:2]:
            raise ValueError("depth image and segmap shape mismatch")

        depth_m = depth_raw.astype(np.float32) * depth_scale
        segmap = np.asarray(segmap_raw, dtype=np.uint8)

        k_values = payload.get("K")
        if not isinstance(k_values, list) or len(k_values) != 9:
            raise ValueError("K must be a 9-element list")
        k_matrix = np.asarray(k_values, dtype=np.float32).reshape((3, 3))

        segmap_id = int(payload.get("segmap_id", 1) or 1)
        local_regions = bool(payload.get("local_regions", True))
        filter_grasps = bool(payload.get("filter_grasps", True))
        skip_border_objects = bool(payload.get("skip_border_objects", False))
        forward_passes = max(1, int(payload.get("forward_passes", self.default_forward_passes)))
        visualize = to_bool(payload.get("visualize"), self.default_visualize)
        z_range_raw = payload.get("z_range", list(self.default_z_range))
        if not isinstance(z_range_raw, list) or len(z_range_raw) < 2:
            z_range_raw = list(self.default_z_range)
        z_range = (float(z_range_raw[0]), float(z_range_raw[1]))
        pregrasp_offset_m = max(
            0.01,
            float(payload.get("pregrasp_offset_m", self.default_pregrasp_offset_m)),
        )
        requested_max_proposals = int(payload.get("max_proposals", self.default_max_proposals))
        max_proposals = max(0, requested_max_proposals)
        task_constraint_tag = str(payload.get("task_constraint_tag") or "pick")
        frame_id = str(payload.get("frame_id") or "")

        with tempfile.TemporaryDirectory(prefix="contact_graspnet_") as tmpdir:
            tmpdir_path = Path(tmpdir)
            input_path = tmpdir_path / "scene_input.npz"
            np.savez(
                input_path,
                depth=depth_m.astype(np.float32),
                K=k_matrix.astype(np.float32),
                segmap=np.where(segmap > 0, segmap_id, 0).astype(np.uint8),
            )

            completed = self._run_official_inference(
                tmpdir_path=tmpdir_path,
                input_path=input_path,
                segmap_id=segmap_id,
                local_regions=local_regions,
                filter_grasps=filter_grasps,
                skip_border_objects=skip_border_objects,
                forward_passes=forward_passes,
                z_range=z_range,
                visualize=visualize,
            )
            if completed.returncode != 0:
                stderr_tail = (completed.stderr or completed.stdout or "").strip()[-1200:]
                raise RuntimeError(
                    f"Contact-GraspNet inference failed with exit code {completed.returncode}: {stderr_tail}"
                )

            prediction_name = f"predictions_{input_path.name.replace('png', 'npz').replace('npy', 'npz')}"
            prediction_path = tmpdir_path / "results" / prediction_name
            if not prediction_path.is_file():
                raise FileNotFoundError(f"prediction file not found: {prediction_path}")

            prediction_data = np.load(prediction_path, allow_pickle=True)
            pred_grasps_cam = load_npz_object(prediction_data["pred_grasps_cam"])
            scores = load_npz_object(prediction_data["scores"])
            contact_pts = load_npz_object(prediction_data["contact_pts"])

            proposals = self._build_proposals(
                pred_grasps_cam=pred_grasps_cam,
                scores=scores,
                contact_pts=contact_pts,
                preferred_segmap_id=segmap_id,
                pregrasp_offset_m=pregrasp_offset_m,
                max_proposals=max_proposals,
                task_constraint_tag=task_constraint_tag,
                frame_id=frame_id,
            )
            source_segment_ids = sorted(
                {int(item["source_segment_id"]) for item in proposals if "source_segment_id" in item}
            )

            return {
                "frame_id": frame_id,
                "proposals": proposals,
                "debug": {
                    "proposal_count": len(proposals),
                    "selected_segment_ids": source_segment_ids,
                    "selected_object_only": True,
                    "local_regions": local_regions,
                    "filter_grasps": filter_grasps,
                    "forward_passes": forward_passes,
                    "segmap_pixels": int(np.count_nonzero(segmap)),
                    "segmap_id": segmap_id,
                    "visualize": visualize,
                },
            }

    def _run_official_inference(
        self,
        *,
        tmpdir_path: Path,
        input_path: Path,
        segmap_id: int,
        local_regions: bool,
        filter_grasps: bool,
        skip_border_objects: bool,
        forward_passes: int,
        z_range: tuple[float, float],
        visualize: bool,
    ) -> subprocess.CompletedProcess[str]:
        runner_path = tmpdir_path / "run_contact_graspnet.py"
        runner_path.write_text(
            self._build_runner_script(
                input_path=input_path,
                segmap_id=segmap_id,
                local_regions=local_regions,
                filter_grasps=filter_grasps,
                skip_border_objects=skip_border_objects,
                forward_passes=forward_passes,
                z_range=z_range,
                visualize=visualize,
            ),
            encoding="utf-8",
        )
        return subprocess.run(
            [self.python_bin, str(runner_path)],
            cwd=str(tmpdir_path),
            capture_output=True,
            text=True,
            check=False,
        )

    def _build_runner_script(
        self,
        *,
        input_path: Path,
        segmap_id: int,
        local_regions: bool,
        filter_grasps: bool,
        skip_border_objects: bool,
        forward_passes: int,
        z_range: tuple[float, float],
        visualize: bool,
    ) -> str:
        sys_paths = [
            str(self.inference_module_dir),
            str(self.contact_graspnet_root / "pointnet2" / "utils"),
            str(self.contact_graspnet_root),
        ]
        lines = [
            "import sys",
            f"sys.path = {sys_paths!r} + [p for p in sys.path if p not in {sys_paths!r}]",
            "import inference as inf",
            "import config_utils",
            f"visualize = {visualize!r}",
            "if not visualize:",
            "    inf.show_image = lambda *args, **kwargs: None",
            "    inf.visualize_grasps = lambda *args, **kwargs: None",
            f"ckpt_dir = {str(self.checkpoint_dir)!r}",
            f"input_path = {str(input_path)!r}",
            f"global_config = config_utils.load_config(ckpt_dir, batch_size={int(forward_passes)}, arg_configs=[])",
            "inf.inference(",
            "    global_config,",
            "    ckpt_dir,",
            "    input_path,",
            "    K=None,",
            f"    local_regions={bool(local_regions)!r},",
            f"    skip_border_objects={bool(skip_border_objects)!r},",
            f"    filter_grasps={bool(filter_grasps)!r},",
            f"    segmap_id={int(segmap_id)},",
            f"    z_range={[float(min(z_range[0], z_range[1])), float(max(z_range[0], z_range[1]))]!r},",
            f"    forward_passes={int(forward_passes)},",
            ")",
        ]
        return "\n".join(lines) + "\n"

    def _build_proposals(
        self,
        *,
        pred_grasps_cam: Any,
        scores: Any,
        contact_pts: Any,
        preferred_segmap_id: int,
        pregrasp_offset_m: float,
        max_proposals: int,
        task_constraint_tag: str,
        frame_id: str,
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for key, grasps, score_values, contact_values in self._iterate_grasp_sets(
            pred_grasps_cam=pred_grasps_cam,
            scores=scores,
            contact_pts=contact_pts,
            preferred_segmap_id=preferred_segmap_id,
        ):
            grasp_array = np.asarray(grasps, dtype=np.float64)
            if grasp_array.ndim == 2 and grasp_array.shape == (4, 4):
                grasp_array = grasp_array[np.newaxis, ...]
            score_array = np.asarray(score_values, dtype=np.float64).reshape((-1,))
            contact_array = None
            if contact_values is not None:
                contact_array = np.asarray(contact_values, dtype=np.float64)
                if contact_array.ndim == 1 and contact_array.shape[0] == 3:
                    contact_array = contact_array[np.newaxis, ...]

            count = min(grasp_array.shape[0], score_array.shape[0])
            for idx in range(count):
                transform = np.asarray(grasp_array[idx], dtype=np.float64).reshape((4, 4))
                score = float(score_array[idx])
                contact_point = None
                if contact_array is not None and idx < int(contact_array.shape[0]):
                    candidate_contact = np.asarray(contact_array[idx], dtype=np.float64).reshape((-1,))
                    if candidate_contact.size >= 3:
                        contact_point = candidate_contact[:3]

                proposal = self._normalize_grasp(
                    transform=transform,
                    score=score,
                    contact_point=contact_point,
                    pregrasp_offset_m=pregrasp_offset_m,
                    task_constraint_tag=task_constraint_tag,
                    frame_id=frame_id,
                    source_segment_id=key,
                )
                if proposal is not None:
                    candidates.append(proposal)

        candidates.sort(key=lambda item: float(item["confidence_score"]), reverse=True)
        trimmed = candidates if max_proposals <= 0 else candidates[:max_proposals]
        for rank, item in enumerate(trimmed, start=1):
            item["candidate_rank"] = rank
        return trimmed

    def _iterate_grasp_sets(
        self,
        *,
        pred_grasps_cam: Any,
        scores: Any,
        contact_pts: Any,
        preferred_segmap_id: int,
    ) -> Iterable[tuple[int, Any, Any, Any]]:
        pred_grasps_cam = load_npz_object(pred_grasps_cam)
        scores = load_npz_object(scores)
        contact_pts = load_npz_object(contact_pts)

        if isinstance(pred_grasps_cam, dict):
            keys = list(pred_grasps_cam.keys())
            preferred_keys = [key for key in keys if segment_key_matches(key, preferred_segmap_id)]
            if not preferred_keys and len(keys) == 1:
                preferred_keys = list(keys)
            for key in preferred_keys:
                key_scores = scores.get(key, []) if isinstance(scores, dict) else scores
                key_contacts = contact_pts.get(key, None) if isinstance(contact_pts, dict) else contact_pts
                try:
                    normalized_key = int(round(float(key)))
                except (TypeError, ValueError):
                    normalized_key = preferred_segmap_id
                yield normalized_key, pred_grasps_cam[key], key_scores, key_contacts
            return

        yield preferred_segmap_id, pred_grasps_cam, scores, contact_pts

    def _normalize_grasp(
        self,
        *,
        transform: np.ndarray,
        score: float,
        contact_point: Optional[np.ndarray],
        pregrasp_offset_m: float,
        task_constraint_tag: str,
        frame_id: str,
        source_segment_id: int,
    ) -> Optional[dict[str, Any]]:
        if transform.shape != (4, 4):
            return None

        rotation = transform[:3, :3]
        position = transform[:3, 3]
        closing_direction = normalize_vector(rotation[:, 0], np.array([1.0, 0.0, 0.0]))
        approach_direction = normalize_vector(rotation[:, 2], np.array([0.0, 0.0, 1.0]))

        default_half_width = 0.025
        if contact_point is not None:
            contact_point = np.asarray(contact_point, dtype=np.float64).reshape((3,))
            projected_half_width = abs(float(np.dot(contact_point - position, closing_direction)))
            half_width = max(0.005, projected_half_width)
        else:
            half_width = default_half_width

        contact_point_1 = position - closing_direction * half_width
        contact_point_2 = position + closing_direction * half_width
        grasp_center = 0.5 * (contact_point_1 + contact_point_2)
        pregrasp_position = position - approach_direction * pregrasp_offset_m
        quaternion_xyzw = rotation_matrix_to_quaternion_xyzw(rotation)
        confidence_score = clamp(confidence_from_score(score), 0.0, 1.0)

        return {
            "frame_id": frame_id,
            "contact_point_1": contact_point_1.astype(float).tolist(),
            "contact_point_2": contact_point_2.astype(float).tolist(),
            "grasp_center": grasp_center.astype(float).tolist(),
            "closing_direction": closing_direction.astype(float).tolist(),
            "approach_direction": approach_direction.astype(float).tolist(),
            "grasp_pose_position": position.astype(float).tolist(),
            "grasp_pose_orientation": quaternion_xyzw,
            "pregrasp_pose_position": pregrasp_position.astype(float).tolist(),
            "pregrasp_pose_orientation": quaternion_xyzw,
            "grasp_width_m": float(2.0 * half_width),
            "confidence_score": confidence_score,
            "semantic_score": 1.0,
            "task_constraint_tag": task_constraint_tag,
            "source_segment_id": int(source_segment_id),
            "raw_score": float(score),
        }


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "ContactGraspNetHTTP/0.1"

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/infer":
            self.send_error(HTTPStatus.NOT_FOUND, "unknown path")
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0") or 0)
            raw_body = self.rfile.read(content_length)
            payload = json.loads(raw_body.decode("utf-8"))
            response_payload = self.server.service.run(payload)  # type: ignore[attr-defined]
            response_body = json.dumps(response_payload, ensure_ascii=True).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)
        except Exception as exc:  # noqa: BLE001
            body = json.dumps(
                {"status": "error", "reason": str(exc)},
                ensure_ascii=True,
            ).encode("utf-8")
            self.send_response(HTTPStatus.INTERNAL_SERVER_ERROR)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(fmt % args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HTTP sidecar for Contact-GraspNet inference")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--python-bin", default="python")
    parser.add_argument("--contact-graspnet-root", required=True)
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--default-pregrasp-offset-m", type=float, default=0.06)
    parser.add_argument("--default-max-proposals", type=int, default=16)
    parser.add_argument("--default-z-min", type=float, default=0.2)
    parser.add_argument("--default-z-max", type=float, default=1.8)
    parser.add_argument("--default-forward-passes", type=int, default=1)
    parser.add_argument("--default-visualize", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = ContactGraspNetService(
        contact_graspnet_root=Path(args.contact_graspnet_root).resolve(),
        checkpoint_dir=Path(args.ckpt_dir).resolve(),
        python_bin=args.python_bin,
        default_pregrasp_offset_m=float(args.default_pregrasp_offset_m),
        default_max_proposals=max(0, int(args.default_max_proposals)),
        default_z_range=(float(args.default_z_min), float(args.default_z_max)),
        default_forward_passes=max(1, int(args.default_forward_passes)),
        default_visualize=bool(args.default_visualize),
    )
    server = ThreadingHTTPServer((args.host, int(args.port)), RequestHandler)
    server.service = service  # type: ignore[attr-defined]
    print(
        f"contact_graspnet_http_service listening on http://{args.host}:{args.port}/infer "
        f"(root={service.contact_graspnet_root}, ckpt={service.checkpoint_dir})"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
