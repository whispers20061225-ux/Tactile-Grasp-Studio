from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from tactile_interfaces.msg import SystemHealth, TactileRaw, TaskExecutionStatus, TaskGoal
from tactile_control.grasp_profile_store import GraspProfileStore
from tactile_control.planner_protocol import (
    DEFAULT_PLANNER_STRATEGY,
    normalize_planner_strategy,
)


def _safe_json_loads(raw_text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_text)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(item) for item in values)
    if len(ordered) == 1:
        return ordered[0]
    index = max(0.0, min(1.0, float(quantile))) * float(len(ordered) - 1)
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return ordered[lower]
    blend = index - float(lower)
    return ordered[lower] * (1.0 - blend) + ordered[upper] * blend


class GraspProfileNode(Node):
    def __init__(self) -> None:
        super().__init__("grasp_profile_node")

        self.declare_parameter("task_goal_topic", "/task/goal")
        self.declare_parameter("task_execution_status_topic", "/task/execution_status")
        self.declare_parameter("tactile_topic", "/tactile/raw")
        self.declare_parameter("pick_status_topic", "/sim/task/pick_status")
        self.declare_parameter("profile_topic", "/control/gripper/profile_json")
        self.declare_parameter("planner_strategy_topic", "/control/gripper/planner_strategy")
        self.declare_parameter("default_planner_strategy", DEFAULT_PLANNER_STRATEGY)
        self.declare_parameter("health_topic", "/system/health")
        self.declare_parameter("database_path", "~/.local/share/programme/grasp_profiles.db")
        self.declare_parameter("seed_data_path", "")
        self.declare_parameter("trial_artifact_dir", "~/.local/share/programme/grasp_trials")
        self.declare_parameter("aggregation_min_success_trials", 2)
        self.declare_parameter("trial_trace_max_points", 240)

        self.task_goal_topic = str(self.get_parameter("task_goal_topic").value)
        self.task_execution_status_topic = str(
            self.get_parameter("task_execution_status_topic").value
        )
        self.tactile_topic = str(self.get_parameter("tactile_topic").value)
        self.pick_status_topic = str(self.get_parameter("pick_status_topic").value)
        self.profile_topic = str(self.get_parameter("profile_topic").value)
        self.planner_strategy_topic = str(self.get_parameter("planner_strategy_topic").value)
        self.health_topic = str(self.get_parameter("health_topic").value)
        self._planner_strategy = normalize_planner_strategy(
            self.get_parameter("default_planner_strategy").value
        )
        database_path = Path(str(self.get_parameter("database_path").value or "").strip()).expanduser()
        seed_data_path = str(self.get_parameter("seed_data_path").value or "").strip()
        trial_artifact_dir = Path(
            str(self.get_parameter("trial_artifact_dir").value or "").strip()
        ).expanduser()
        aggregation_min_success_trials = int(
            self.get_parameter("aggregation_min_success_trials").value
        )
        self.trial_trace_max_points = max(32, int(self.get_parameter("trial_trace_max_points").value))
        if not seed_data_path:
            seed_data_path = str(
                Path(__file__).resolve().parent / "data" / "grasp_profile_seed.json"
            )
        self._profile_store = GraspProfileStore(
            database_path=database_path,
            seed_data_path=Path(seed_data_path).expanduser(),
            artifact_dir=trial_artifact_dir,
            aggregation_min_success_trials=aggregation_min_success_trials,
        )

        qos_reliable = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        self.profile_pub = self.create_publisher(String, self.profile_topic, qos_reliable)
        self.health_pub = self.create_publisher(SystemHealth, self.health_topic, qos_reliable)
        self.create_subscription(TaskGoal, self.task_goal_topic, self._on_task_goal, qos_reliable)
        self.create_subscription(
            String,
            self.planner_strategy_topic,
            self._on_planner_strategy,
            qos_reliable,
        )
        self.create_subscription(
            TaskExecutionStatus,
            self.task_execution_status_topic,
            self._on_task_execution_status,
            qos_reliable,
        )
        self.create_subscription(TactileRaw, self.tactile_topic, self._on_tactile_raw, qos_reliable)
        self.create_subscription(String, self.pick_status_topic, self._on_pick_status, qos_reliable)
        self.create_timer(1.0, self._publish_health)

        self._last_goal_snapshot: dict[str, Any] = self._snapshot_goal(TaskGoal())
        self._last_profile: dict[str, Any] = self._build_profile(self._last_goal_snapshot)
        self._goal_context_by_id: dict[str, dict[str, Any]] = {}
        self._profile_by_goal_id: dict[str, dict[str, Any]] = {}
        self._active_trial: dict[str, Any] | None = None
        self._publish_profile()
        self.get_logger().info(
            "grasp_profile_node started: "
            f"task_goal_topic={self.task_goal_topic} "
            f"task_execution_status_topic={self.task_execution_status_topic} "
            f"tactile_topic={self.tactile_topic} "
            f"pick_status_topic={self.pick_status_topic} "
            f"profile_topic={self.profile_topic} "
            f"planner_strategy_topic={self.planner_strategy_topic} "
            f"default_planner_strategy={self._planner_strategy} "
            f"database_path={database_path}"
        )

    def _on_task_goal(self, msg: TaskGoal) -> None:
        self._last_goal_snapshot = self._snapshot_goal(msg)
        goal_id = str(msg.goal_id or "").strip()
        self._last_profile = self._build_profile(self._last_goal_snapshot)
        if goal_id:
            self._goal_context_by_id[goal_id] = dict(self._last_goal_snapshot)
            self._profile_by_goal_id[goal_id] = dict(self._last_profile)
            if len(self._goal_context_by_id) > 16:
                oldest_key = next(iter(self._goal_context_by_id))
                self._goal_context_by_id.pop(oldest_key, None)
                self._profile_by_goal_id.pop(oldest_key, None)
        self._publish_profile()

    def _on_planner_strategy(self, msg: String) -> None:
        normalized = normalize_planner_strategy(str(msg.data or "").strip())
        if normalized == self._planner_strategy:
            return
        self._planner_strategy = normalized
        goal_id = str(self._last_goal_snapshot.get("goal_id") or "").strip()
        self._last_profile = self._build_profile(self._last_goal_snapshot)
        if goal_id:
            self._profile_by_goal_id[goal_id] = dict(self._last_profile)
        self._publish_profile()
        self.get_logger().info(f"planner strategy updated: {self._planner_strategy}")

    def _on_task_execution_status(self, msg: TaskExecutionStatus) -> None:
        goal_id = str(msg.goal_id or "").strip()
        if bool(msg.active):
            self._ensure_active_trial(goal_id=goal_id, status=msg)
            return
        phase = str(msg.phase or "").strip().lower()
        if phase not in {"completed", "error", "cancelled"}:
            return
        goal_context = dict(self._goal_context_by_id.get(goal_id) or self._last_goal_snapshot)
        profile_context = dict(self._profile_by_goal_id.get(goal_id) or self._last_profile)
        status_context = self._snapshot_status(msg)
        artifact_payload: dict[str, Any] | None = None
        if self._active_trial is not None and str(self._active_trial.get("goal_id") or "") == goal_id:
            self._active_trial["status_events"].append(
                {
                    "ts": time.time(),
                    "phase": str(msg.phase or "").strip(),
                    "success": bool(msg.success),
                    "message": str(msg.message or "").strip(),
                    "progress": float(msg.progress or 0.0),
                }
            )
            tactile_summary, tactile_trace = self._summarize_tactile_samples(
                list(self._active_trial.get("tactile_samples") or [])
            )
            pick_summary = self._summarize_pick_status_events(
                list(self._active_trial.get("pick_status_events") or [])
            )
            artifact_payload = {
                "trial_start_time": float(self._active_trial.get("start_time", 0.0) or 0.0),
                "trial_end_time": time.time(),
                "tactile_summary": tactile_summary,
                "tactile_trace": tactile_trace,
                "pick_status_summary": pick_summary,
                "pick_status_events": self._decimate_trace(
                    list(self._active_trial.get("pick_status_events") or []),
                    self.trial_trace_max_points,
                ),
                "status_events": self._decimate_trace(
                    list(self._active_trial.get("status_events") or []),
                    self.trial_trace_max_points,
                ),
            }
            status_context["tactile_summary"] = tactile_summary
            status_context["pick_status_summary"] = pick_summary
        status_signature = "|".join(
            [
                goal_id or "<no-goal-id>",
                phase or "<no-phase>",
                str(int(bool(msg.success))),
                str(int(msg.retry_count or 0)),
                str(msg.error_code or "").strip(),
                str(msg.current_skill or "").strip(),
            ]
        )
        inserted = self._profile_store.record_trial(
            status_signature=status_signature,
            goal_context=goal_context,
            status_context=status_context,
            profile=profile_context,
            artifact_payload=artifact_payload,
        )
        if inserted:
            self.get_logger().info(
                "grasp trial recorded: "
                f"goal_id={goal_id or 'unset'} phase={phase} success={bool(msg.success)} "
                f"profile={profile_context.get('profile_id', 'unset')}"
            )
        if self._active_trial is not None and str(self._active_trial.get("goal_id") or "") == goal_id:
            self._active_trial = None

    def _on_tactile_raw(self, msg: TactileRaw) -> None:
        if self._active_trial is None:
            return
        stamp_sec = float(getattr(msg.header.stamp, "sec", 0)) + (
            float(getattr(msg.header.stamp, "nanosec", 0)) / 1e9
        )
        effective_ts = stamp_sec if stamp_sec > 0.0 else time.time()
        self._active_trial["tactile_samples"].append(
            {
                "ts": effective_ts,
                "total_fx": float(msg.total_fx or 0.0),
                "total_fy": float(msg.total_fy or 0.0),
                "total_fz": float(msg.total_fz or 0.0),
                "contact_active": bool(msg.contact_active),
                "contact_score": float(msg.contact_score or 0.0),
                "source_connected": bool(msg.source_connected),
                "publish_rate_hz": float(msg.publish_rate_hz or 0.0),
                "transport_rate_hz": float(msg.transport_rate_hz or 0.0),
            }
        )

    def _on_pick_status(self, msg: String) -> None:
        if self._active_trial is None:
            return
        parsed = _safe_json_loads(str(msg.data or ""))
        parsed["raw"] = str(msg.data or "")
        parsed["ts"] = time.time()
        self._active_trial["pick_status_events"].append(parsed)

    def _build_profile(self, goal: TaskGoal | dict[str, Any]) -> dict[str, Any]:
        goal_context = dict(goal) if isinstance(goal, dict) else self._snapshot_goal(goal)
        return self._profile_store.build_plan(
            goal_context=goal_context,
            planner_strategy=self._planner_strategy,
        )

    def _snapshot_goal(self, goal: TaskGoal) -> dict[str, Any]:
        return {
            "goal_id": str(goal.goal_id or "").strip(),
            "target_label": str(goal.target_label or "").strip(),
            "target_hint": str(goal.target_hint or "").strip(),
            "target_part_query": str(goal.target_part_query or "").strip(),
            "grasp_region": str(goal.grasp_region or "").strip(),
            "preferred_grasp_family": str(goal.preferred_grasp_family or "").strip(),
            "target_instance_track_id": int(goal.target_instance_track_id or 0),
            "confidence": float(goal.confidence or 0.0),
            "handoff_context_json": str(goal.handoff_context_json or "").strip(),
            "raw_json": str(goal.raw_json or "").strip(),
        }

    def _snapshot_status(self, status: TaskExecutionStatus) -> dict[str, Any]:
        return {
            "goal_id": str(status.goal_id or "").strip(),
            "phase": str(status.phase or "").strip(),
            "current_skill": str(status.current_skill or "").strip(),
            "skill_status_code": str(status.skill_status_code or "").strip(),
            "message": str(status.message or "").strip(),
            "error_code": str(status.error_code or "").strip(),
            "target_label": str(status.target_label or "").strip(),
            "target_hint": str(status.target_hint or "").strip(),
            "success": bool(status.success),
            "target_locked": bool(status.target_locked),
            "target_candidate_visible": bool(status.target_candidate_visible),
            "pick_active": bool(status.pick_active),
            "retry_count": int(status.retry_count or 0),
            "max_retries": int(status.max_retries or 0),
            "grounded_track_id": int(status.grounded_track_id or 0),
            "selected_candidate_id": int(status.selected_candidate_id or 0),
            "progress": float(status.progress or 0.0),
            "grounding_confidence": float(status.grounding_confidence or 0.0),
            "best_grasp_quality": float(status.best_grasp_quality or 0.0),
            "best_affordance_score": float(status.best_affordance_score or 0.0),
            "verification_score": float(status.verification_score or 0.0),
            "recommended_recovery": str(status.recommended_recovery or "").strip(),
        }

    def _ensure_active_trial(self, *, goal_id: str, status: TaskExecutionStatus) -> None:
        if self._active_trial is not None and str(self._active_trial.get("goal_id") or "") == goal_id:
            self._active_trial["status_events"].append(
                {
                    "ts": time.time(),
                    "phase": str(status.phase or "").strip(),
                    "success": bool(status.success),
                    "message": str(status.message or "").strip(),
                    "progress": float(status.progress or 0.0),
                }
            )
            return
        self._active_trial = {
            "goal_id": goal_id,
            "start_time": time.time(),
            "goal_context": dict(self._goal_context_by_id.get(goal_id) or self._last_goal_snapshot),
            "profile": dict(self._profile_by_goal_id.get(goal_id) or self._last_profile),
            "status_events": [
                {
                    "ts": time.time(),
                    "phase": str(status.phase or "").strip(),
                    "success": bool(status.success),
                    "message": str(status.message or "").strip(),
                    "progress": float(status.progress or 0.0),
                }
            ],
            "pick_status_events": [],
            "tactile_samples": [],
        }

    def _summarize_tactile_samples(
        self,
        samples: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if not samples:
            return {
                "sample_count": 0,
                "duration_sec": 0.0,
                "mean_total_fz": 0.0,
                "std_total_fz": 0.0,
                "peak_total_fz": 0.0,
                "p90_total_fz": 0.0,
                "p95_total_fz": 0.0,
                "final_total_fz": 0.0,
                "contact_ratio": 0.0,
                "first_contact_sec": -1.0,
                "source_connected_ratio": 0.0,
                "avg_publish_rate_hz": 0.0,
                "avg_transport_rate_hz": 0.0,
                "peak_contact_score": 0.0,
            }, []
        start_ts = float(samples[0].get("ts", 0.0) or 0.0)
        end_ts = float(samples[-1].get("ts", start_ts) or start_ts)
        total_fz = [float(item.get("total_fz", 0.0) or 0.0) for item in samples]
        contact_score = [float(item.get("contact_score", 0.0) or 0.0) for item in samples]
        publish_rates = [float(item.get("publish_rate_hz", 0.0) or 0.0) for item in samples]
        transport_rates = [float(item.get("transport_rate_hz", 0.0) or 0.0) for item in samples]
        source_connected = [1.0 if bool(item.get("source_connected")) else 0.0 for item in samples]
        contact_indices = [idx for idx, item in enumerate(samples) if bool(item.get("contact_active"))]
        mean_force = statistics.fmean(total_fz)
        std_force = statistics.pstdev(total_fz) if len(total_fz) > 1 else 0.0
        first_contact_sec = -1.0
        if contact_indices:
            first_contact_sec = float(samples[contact_indices[0]].get("ts", start_ts) or start_ts) - start_ts
        summary = {
            "sample_count": len(samples),
            "duration_sec": round(max(0.0, end_ts - start_ts), 4),
            "mean_total_fz": round(mean_force, 4),
            "std_total_fz": round(std_force, 4),
            "peak_total_fz": round(max(total_fz), 4),
            "p90_total_fz": round(_percentile(total_fz, 0.90), 4),
            "p95_total_fz": round(_percentile(total_fz, 0.95), 4),
            "final_total_fz": round(total_fz[-1], 4),
            "contact_ratio": round(float(len(contact_indices)) / float(max(1, len(samples))), 4),
            "first_contact_sec": round(first_contact_sec, 4),
            "source_connected_ratio": round(statistics.fmean(source_connected), 4),
            "avg_publish_rate_hz": round(statistics.fmean(publish_rates), 4),
            "avg_transport_rate_hz": round(statistics.fmean(transport_rates), 4),
            "peak_contact_score": round(max(contact_score), 4),
        }
        trace = self._decimate_trace(
            [
                {
                    "t": round(float(item.get("ts", start_ts) or start_ts) - start_ts, 4),
                    "total_fz": round(float(item.get("total_fz", 0.0) or 0.0), 4),
                    "contact_score": round(float(item.get("contact_score", 0.0) or 0.0), 4),
                    "contact_active": bool(item.get("contact_active")),
                }
                for item in samples
            ],
            self.trial_trace_max_points,
        )
        return summary, trace

    def _summarize_pick_status_events(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        if not events:
            return {
                "event_count": 0,
                "final_phase": "",
                "max_progress_percent": 0,
            }
        progress_percent = [
            int(item.get("progress_percent", 0) or 0)
            for item in events
            if str(item.get("phase", "") or "").strip().lower() == "planning"
        ]
        final_phase = str(events[-1].get("phase", "") or "").strip()
        return {
            "event_count": len(events),
            "final_phase": final_phase,
            "max_progress_percent": max(progress_percent) if progress_percent else 0,
        }

    def _decimate_trace(self, items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        if len(items) <= limit:
            return items
        stride = max(1, len(items) // max(1, limit))
        reduced = items[::stride]
        if reduced[-1] != items[-1]:
            reduced.append(items[-1])
        return reduced[:limit]

    def _publish_profile(self) -> None:
        msg = String()
        msg.data = json.dumps(self._last_profile, ensure_ascii=False, separators=(",", ":"))
        self.profile_pub.publish(msg)

    def _publish_health(self) -> None:
        msg = SystemHealth()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "system"
        msg.node_name = self.get_name()
        msg.healthy = True
        msg.level = 0
        msg.cpu_percent = 0.0
        msg.memory_percent = 0.0
        msg.message = (
            f"profile={self._last_profile.get('profile_id', 'unset')} "
            f"planner={self._planner_strategy}"
        )
        self.health_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GraspProfileNode()
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
