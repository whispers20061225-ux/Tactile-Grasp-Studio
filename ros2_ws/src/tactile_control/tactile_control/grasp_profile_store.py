from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
import statistics
import time
from pathlib import Path
from typing import Any

from tactile_control.planner_protocol import (
    DEFAULT_PLANNER_STRATEGY,
    PLANNER_DB_ONLY,
    PLANNER_NUMERIC_FIELDS,
    PLANNER_RETRIEVAL_AUG,
    normalize_planner_strategy,
    planner_promotes_rollup,
)


def _safe_json_loads(raw_text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_text)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _clamp01(value: Any, default: float) -> float:
    try:
        numeric = float(value)
    except Exception:
        return float(default)
    return max(0.0, min(1.0, numeric))


def _positive(value: Any, default: float) -> float:
    try:
        numeric = float(value)
    except Exception:
        return float(default)
    return float(default) if numeric <= 0.0 else numeric


def _normalized_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _normalized_aliases(*values: Any) -> list[str]:
    seen: set[str] = set()
    aliases: list[str] = []
    for value in values:
        if isinstance(value, list):
            for item in value:
                token = _normalized_text(item)
                if token and token not in seen:
                    seen.add(token)
                    aliases.append(token)
            continue
        token = _normalized_text(value)
        if token and token not in seen:
            seen.add(token)
            aliases.append(token)
    return aliases


def _midpoint(minimum: Any, maximum: Any, fallback: float) -> float:
    low = _positive(minimum, fallback)
    high = _positive(maximum, low)
    if high < low:
        low, high = high, low
    return 0.5 * (low + high)


class GraspProfileStore:
    def __init__(
        self,
        *,
        database_path: Path,
        seed_data_path: Path | None = None,
        artifact_dir: Path | None = None,
        aggregation_min_success_trials: int = 2,
    ) -> None:
        self.database_path = Path(database_path).expanduser()
        self.seed_data_path = Path(seed_data_path).expanduser() if seed_data_path else None
        self.artifact_dir = Path(artifact_dir).expanduser() if artifact_dir else None
        self.trial_artifact_dir = self.artifact_dir / "trials" if self.artifact_dir else None
        self.rollup_artifact_dir = self.artifact_dir / "rollups" if self.artifact_dir else None
        self.aggregation_min_success_trials = max(1, int(aggregation_min_success_trials))
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        if self.trial_artifact_dir is not None:
            self.trial_artifact_dir.mkdir(parents=True, exist_ok=True)
        if self.rollup_artifact_dir is not None:
            self.rollup_artifact_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()
        self._ensure_seed_data()

    def build_profile(self, *, target_label: str, target_hint: str, confidence: float, raw_context: str) -> dict[str, Any]:
        return self.build_plan(
            goal_context={
                "target_label": str(target_label or "").strip(),
                "target_hint": str(target_hint or "").strip(),
                "confidence": float(confidence or 0.0),
                "handoff_context_json": str(raw_context or "").strip(),
            },
            planner_strategy=DEFAULT_PLANNER_STRATEGY,
        )

    def build_plan(
        self,
        *,
        goal_context: dict[str, Any],
        planner_strategy: str,
    ) -> dict[str, Any]:
        normalized_planner = normalize_planner_strategy(planner_strategy)
        raw_context = str(goal_context.get("handoff_context_json") or "").strip()
        context = _safe_json_loads(raw_context)
        merged_context = self._resolve_context(
            target_label=str(goal_context.get("target_label") or "").strip(),
            target_hint=str(goal_context.get("target_hint") or "").strip(),
            confidence=float(goal_context.get("confidence", 0.0) or 0.0),
            context=context,
        )
        db_profile = self._apply_rollup(
            self._build_profile_from_context(merged_context, raw_context=raw_context)
        )
        planner_features = self._build_planner_features(
            goal_context=goal_context,
            profile=db_profile,
            resolved_context=merged_context,
        )
        planned_profile = dict(db_profile)
        planned_profile["db_prior"] = copy.deepcopy(db_profile)
        planned_profile["planner_strategy"] = normalized_planner
        planned_profile["planner_source"] = "database_only"
        planned_profile["planner_features"] = planner_features
        planned_profile["planner_confidence"] = self._estimate_db_planner_confidence(planned_profile)
        planned_profile["promote_rollup"] = planner_promotes_rollup(normalized_planner)
        planned_profile["retrieval"] = {
            "applied": False,
            "fallback_used": normalized_planner == PLANNER_RETRIEVAL_AUG,
            "neighbor_count": 0,
            "neighbors": [],
            "aggregated_params": {},
            "blend_alpha": 0.0,
            "confidence": 0.0,
        }
        if normalized_planner == PLANNER_RETRIEVAL_AUG:
            planned_profile = self._apply_retrieval_augmented_plan(
                goal_context=goal_context,
                db_profile=planned_profile,
            )
        planned_profile["updated_at"] = time.time()
        return planned_profile

    def record_trial(
        self,
        *,
        status_signature: str,
        goal_context: dict[str, Any],
        status_context: dict[str, Any],
        profile: dict[str, Any],
        artifact_payload: dict[str, Any] | None = None,
    ) -> bool:
        signature = str(status_signature or "").strip()
        if not signature:
            return False

        target_label = str(
            goal_context.get("target_label")
            or status_context.get("target_label")
            or profile.get("target_label")
            or ""
        ).strip()
        strategy = str(
            profile.get("preferred_strategy")
            or goal_context.get("preferred_grasp_family")
            or ""
        ).strip()
        planner_strategy = normalize_planner_strategy(profile.get("planner_strategy"))
        planner_confidence = float(profile.get("planner_confidence", 0.0) or 0.0)
        planner_source = str(profile.get("planner_source") or "").strip()
        planner_features = (
            copy.deepcopy(profile.get("planner_features"))
            if isinstance(profile.get("planner_features"), dict)
            else {}
        )
        promote_rollup = 1 if bool(profile.get("promote_rollup", planner_promotes_rollup(planner_strategy))) else 0
        params_json = {
            "goal": copy.deepcopy(goal_context),
            "profile": copy.deepcopy(profile),
        }
        metrics_json = copy.deepcopy(status_context)
        key_metrics = self._extract_key_metrics(
            status_context=status_context,
            payload=artifact_payload,
        )
        if key_metrics:
            metrics_json["key_metrics"] = key_metrics
        artifact_path = self._write_trial_artifact(
            goal_context=goal_context,
            status_context=status_context,
            profile=profile,
            payload=artifact_payload,
            key_metrics=key_metrics,
        )
        if artifact_path:
            metrics_json["artifact_path"] = artifact_path
        now_sec = time.time()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO grasp_trials(
                    created_at,
                    goal_id,
                    target_label,
                    strategy,
                    planner_strategy,
                    planner_confidence,
                    planner_source,
                    planner_feature_json,
                    promote_rollup,
                    object_profile_id,
                    material_profile_id,
                    success,
                    params_json,
                    metrics_json,
                    note,
                    status_signature,
                    phase,
                    error_code
                )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now_sec,
                    str(goal_context.get("goal_id") or "").strip(),
                    target_label,
                    strategy,
                    planner_strategy,
                    planner_confidence,
                    planner_source,
                    json.dumps(planner_features, ensure_ascii=False, separators=(",", ":")),
                    promote_rollup,
                    str(profile.get("object_profile_id") or "").strip(),
                    str(profile.get("material_id") or "").strip(),
                    1 if bool(status_context.get("success")) else 0,
                    json.dumps(params_json, ensure_ascii=False, separators=(",", ":")),
                    json.dumps(metrics_json, ensure_ascii=False, separators=(",", ":")),
                    str(status_context.get("message") or "").strip(),
                    signature,
                    str(status_context.get("phase") or "").strip(),
                    str(status_context.get("error_code") or "").strip(),
                ),
            )
            inserted = int(cursor.rowcount or 0) > 0
            if inserted:
                self._record_planner_decision(
                    connection,
                    status_signature=signature,
                    goal_context=goal_context,
                    status_context=status_context,
                    profile=profile,
                    key_metrics=key_metrics,
                )
            connection.commit()
        if inserted and promote_rollup:
            self._refresh_rollups_for_profile(profile)
        return inserted

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.database_path))
        connection.row_factory = sqlite3.Row
        return connection

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS material_profiles (
                    material_id TEXT PRIMARY KEY,
                    aliases_json TEXT NOT NULL,
                    properties_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS object_profiles (
                    object_id TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    aliases_json TEXT NOT NULL,
                    material_id TEXT,
                    preferred_strategy TEXT,
                    preferred_grasp_family TEXT,
                    properties_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS grasp_trials (
                    trial_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at REAL NOT NULL,
                    goal_id TEXT,
                    target_label TEXT,
                    strategy TEXT,
                    planner_strategy TEXT,
                    planner_confidence REAL,
                    planner_source TEXT,
                    planner_feature_json TEXT,
                    promote_rollup INTEGER DEFAULT 1,
                    object_profile_id TEXT,
                    material_profile_id TEXT,
                    success INTEGER,
                    params_json TEXT,
                    metrics_json TEXT,
                    note TEXT,
                    status_signature TEXT,
                    phase TEXT,
                    error_code TEXT
                );

                CREATE TABLE IF NOT EXISTS profile_rollups (
                    rollup_key TEXT PRIMARY KEY,
                    object_profile_id TEXT,
                    material_profile_id TEXT,
                    strategy TEXT,
                    stats_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS planner_decisions (
                    decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at REAL NOT NULL,
                    status_signature TEXT,
                    goal_id TEXT,
                    target_label TEXT,
                    grasp_strategy TEXT,
                    planner_strategy TEXT,
                    planner_confidence REAL,
                    retrieval_applied INTEGER,
                    promote_rollup INTEGER,
                    object_profile_id TEXT,
                    material_profile_id TEXT,
                    feature_json TEXT,
                    db_prior_json TEXT,
                    retrieval_json TEXT,
                    final_profile_json TEXT,
                    outcome_json TEXT
                );
                """
            )
            columns = {
                str(row["name"]): row
                for row in connection.execute("PRAGMA table_info(grasp_trials)").fetchall()
            }
            if "planner_strategy" not in columns:
                connection.execute("ALTER TABLE grasp_trials ADD COLUMN planner_strategy TEXT")
            if "planner_confidence" not in columns:
                connection.execute("ALTER TABLE grasp_trials ADD COLUMN planner_confidence REAL")
            if "planner_source" not in columns:
                connection.execute("ALTER TABLE grasp_trials ADD COLUMN planner_source TEXT")
            if "planner_feature_json" not in columns:
                connection.execute("ALTER TABLE grasp_trials ADD COLUMN planner_feature_json TEXT")
            if "promote_rollup" not in columns:
                connection.execute("ALTER TABLE grasp_trials ADD COLUMN promote_rollup INTEGER DEFAULT 1")
            if "status_signature" not in columns:
                connection.execute("ALTER TABLE grasp_trials ADD COLUMN status_signature TEXT")
            if "phase" not in columns:
                connection.execute("ALTER TABLE grasp_trials ADD COLUMN phase TEXT")
            if "error_code" not in columns:
                connection.execute("ALTER TABLE grasp_trials ADD COLUMN error_code TEXT")
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_grasp_trials_status_signature
                ON grasp_trials(status_signature)
                WHERE status_signature IS NOT NULL
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_grasp_trials_rollup_object
                ON grasp_trials(object_profile_id, strategy, promote_rollup)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_grasp_trials_rollup_material
                ON grasp_trials(material_profile_id, strategy, promote_rollup)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_grasp_trials_planner_strategy
                ON grasp_trials(planner_strategy, created_at)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_grasp_trials_success_created
                ON grasp_trials(success, created_at)
                """
            )
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_planner_decisions_status_signature
                ON planner_decisions(status_signature)
                WHERE status_signature IS NOT NULL
                """
            )
            connection.commit()

    def _ensure_seed_data(self) -> None:
        seed_path = self.seed_data_path
        if seed_path is None or not seed_path.is_file():
            return
        raw_seed = seed_path.read_text(encoding="utf-8")
        seed_hash = hashlib.sha256(raw_seed.encode("utf-8")).hexdigest()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM meta WHERE key = 'seed_hash'"
            ).fetchone()
            if row is not None and str(row["value"] or "") == seed_hash:
                return
            parsed = _safe_json_loads(raw_seed)
            materials = parsed.get("materials") if isinstance(parsed.get("materials"), list) else []
            objects = parsed.get("objects") if isinstance(parsed.get("objects"), list) else []
            now_sec = time.time()

            for record in materials:
                if not isinstance(record, dict):
                    continue
                material_id = _normalized_text(record.get("material_id"))
                if not material_id:
                    continue
                aliases = _normalized_aliases(material_id, record.get("aliases"))
                properties = copy.deepcopy(record)
                connection.execute(
                    """
                    INSERT INTO material_profiles(material_id, aliases_json, properties_json, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(material_id) DO UPDATE SET
                        aliases_json=excluded.aliases_json,
                        properties_json=excluded.properties_json,
                        updated_at=excluded.updated_at
                    """,
                    (
                        material_id,
                        json.dumps(aliases, ensure_ascii=False, separators=(",", ":")),
                        json.dumps(properties, ensure_ascii=False, separators=(",", ":")),
                        now_sec,
                    ),
                )

            for record in objects:
                if not isinstance(record, dict):
                    continue
                object_id = _normalized_text(record.get("object_id"))
                label = str(record.get("label") or object_id or "").strip()
                if not object_id or not label:
                    continue
                aliases = _normalized_aliases(object_id, label, record.get("aliases"))
                properties = copy.deepcopy(record)
                connection.execute(
                    """
                    INSERT INTO object_profiles(
                        object_id,
                        label,
                        aliases_json,
                        material_id,
                        preferred_strategy,
                        preferred_grasp_family,
                        properties_json,
                        updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(object_id) DO UPDATE SET
                        label=excluded.label,
                        aliases_json=excluded.aliases_json,
                        material_id=excluded.material_id,
                        preferred_strategy=excluded.preferred_strategy,
                        preferred_grasp_family=excluded.preferred_grasp_family,
                        properties_json=excluded.properties_json,
                        updated_at=excluded.updated_at
                    """,
                    (
                        object_id,
                        label,
                        json.dumps(aliases, ensure_ascii=False, separators=(",", ":")),
                        _normalized_text(record.get("material_id")) or None,
                        str(record.get("preferred_strategy") or "").strip(),
                        str(record.get("preferred_grasp_family") or "").strip(),
                        json.dumps(properties, ensure_ascii=False, separators=(",", ":")),
                        now_sec,
                    ),
                )

            connection.execute(
                """
                INSERT INTO meta(key, value, updated_at)
                VALUES ('seed_hash', ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value=excluded.value,
                    updated_at=excluded.updated_at
                """,
                (seed_hash, now_sec),
            )
            connection.commit()

    def _resolve_context(
        self,
        *,
        target_label: str,
        target_hint: str,
        confidence: float,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        requested_tokens = _normalized_aliases(
            context.get("object_type"),
            context.get("object"),
            target_label,
            target_hint,
        )
        object_record = self._find_best_object_profile(requested_tokens)
        material_record = None
        if object_record is not None:
            material_record = self._find_material_profile(object_record.get("material_id"))
        if material_record is None:
            material_tokens = _normalized_aliases(
                context.get("material"),
                context.get("material_id"),
            )
            material_record = self._find_best_material_profile(material_tokens)

        resolved: dict[str, Any] = {
            "target_label": str(target_label or "").strip(),
            "target_hint": str(target_hint or "").strip(),
            "confidence": float(confidence or 0.0),
            "source_chain": [],
            "query_tokens": requested_tokens,
        }
        for source_name, record in (("material_profile", material_record), ("object_profile", object_record), ("handoff_context", context)):
            if not record:
                continue
            resolved["source_chain"].append(source_name)
            if isinstance(record, dict):
                resolved.update(copy.deepcopy(record))

        if object_record is not None:
            resolved["object_profile_id"] = str(object_record.get("object_id") or "").strip()
            resolved["matched_object_label"] = str(object_record.get("label") or "").strip()
        if material_record is not None:
            resolved["material_id"] = str(material_record.get("material_id") or "").strip()

        if not resolved.get("object_type"):
            resolved["object_type"] = str(
                context.get("object_type")
                or context.get("object")
                or target_label
                or target_hint
                or ""
            ).strip()
        return resolved

    def _find_best_object_profile(self, tokens: list[str]) -> dict[str, Any] | None:
        if not tokens:
            return None
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT object_id, label, material_id, preferred_strategy, preferred_grasp_family, aliases_json, properties_json
                FROM object_profiles
                """
            ).fetchall()
        best_score = -1.0
        best_record: dict[str, Any] | None = None
        for row in rows:
            try:
                alias_list = json.loads(str(row["aliases_json"] or "[]"))
            except Exception:
                alias_list = []
            alias_tokens = _normalized_aliases(alias_list, row["label"], row["object_id"])
            score = self._match_alias_score(tokens, alias_tokens)
            if score <= best_score:
                continue
            properties = _safe_json_loads(str(row["properties_json"] or "{}"))
            properties.update(
                {
                    "object_id": str(row["object_id"] or "").strip(),
                    "label": str(row["label"] or "").strip(),
                    "material_id": str(row["material_id"] or "").strip(),
                    "preferred_strategy": str(row["preferred_strategy"] or "").strip(),
                    "preferred_grasp_family": str(row["preferred_grasp_family"] or "").strip(),
                    "aliases": alias_tokens,
                }
            )
            best_score = score
            best_record = properties
        return best_record if best_score >= 0.6 else None

    def _find_best_material_profile(self, tokens: list[str]) -> dict[str, Any] | None:
        if not tokens:
            return None
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT material_id, aliases_json, properties_json FROM material_profiles"
            ).fetchall()
        best_score = -1.0
        best_record: dict[str, Any] | None = None
        for row in rows:
            try:
                alias_list = json.loads(str(row["aliases_json"] or "[]"))
            except Exception:
                alias_list = []
            alias_tokens = _normalized_aliases(alias_list, row["material_id"])
            score = self._match_alias_score(tokens, alias_tokens)
            if score <= best_score:
                continue
            properties = _safe_json_loads(str(row["properties_json"] or "{}"))
            properties.update(
                {
                    "material_id": str(row["material_id"] or "").strip(),
                    "aliases": alias_tokens,
                }
            )
            best_score = score
            best_record = properties
        return best_record if best_score >= 0.6 else None

    def _find_material_profile(self, material_id: Any) -> dict[str, Any] | None:
        normalized = _normalized_text(material_id)
        if not normalized:
            return None
        with self._connect() as connection:
            row = connection.execute(
                "SELECT material_id, aliases_json, properties_json FROM material_profiles WHERE material_id = ?",
                (normalized,),
            ).fetchone()
        if row is None:
            return None
        properties = _safe_json_loads(str(row["properties_json"] or "{}"))
        try:
            alias_list = json.loads(str(row["aliases_json"] or "[]"))
        except Exception:
            alias_list = []
        properties.update(
            {
                "material_id": str(row["material_id"] or "").strip(),
                "aliases": _normalized_aliases(alias_list, row["material_id"]),
            }
        )
        return properties

    def _match_alias_score(self, query_tokens: list[str], alias_tokens: list[str]) -> float:
        if not query_tokens or not alias_tokens:
            return -1.0
        best = -1.0
        alias_set = set(alias_tokens)
        for token in query_tokens:
            if token in alias_set:
                best = max(best, 1.0)
                continue
            for alias in alias_tokens:
                if token and alias and (token in alias or alias in token):
                    best = max(best, 0.75)
        return best

    def _build_profile_from_context(self, context: dict[str, Any], *, raw_context: str) -> dict[str, Any]:
        fragility = _clamp01(context.get("fragility"), 0.45)
        surface_friction = _clamp01(
            context.get("surface_friction", context.get("friction")), 0.50
        )
        compliance = _clamp01(context.get("compliance"), 0.45)
        slip_risk = _clamp01(context.get("slip_risk"), max(0.0, 0.8 - surface_friction))
        confidence = _clamp01(context.get("confidence"), 0.50)
        mass_g = _positive(context.get("mass_g"), 80.0)

        range_target_force = _midpoint(
            context.get("target_force_min"),
            context.get("target_force_max"),
            1.8,
        )
        preferred_contact_force = _positive(
            context.get("preferred_contact_force_n"),
            range_target_force
            if range_target_force > 0.0
            else (0.8 + min(1.2, mass_g / 250.0)) * (1.15 - 0.45 * fragility) * (1.0 + 0.25 * slip_risk),
        )
        target_force = _positive(context.get("target_force"), preferred_contact_force)

        if fragility >= 0.70:
            profile_id = "fragile_soft"
        elif surface_friction <= 0.35:
            profile_id = "slippery_secure"
        elif compliance >= 0.65:
            profile_id = "compliant_gentle"
        else:
            profile_id = "default_pick"

        if str(context.get("object_profile_id") or "").strip():
            profile_id = str(context.get("object_profile_id") or "").strip()
        elif str(context.get("material_id") or "").strip():
            profile_id = str(context.get("material_id") or "").strip()

        kp = _positive(
            context.get("kp"),
            0.60 + (1.0 - fragility) * 0.80 + slip_risk * 0.30,
        )
        ki = _positive(
            context.get("ki"),
            0.02 + (1.0 - fragility) * 0.04,
        )
        kd = _positive(
            context.get("kd"),
            0.08 + slip_risk * 0.18 + (1.0 - surface_friction) * 0.08,
        )
        contact_threshold = _positive(
            context.get("contact_threshold"),
            max(1.0, min(6.0, target_force * 0.35 + compliance * 0.60)),
        )
        safety_max = _positive(
            context.get("safety_max"),
            max(target_force + 0.60, target_force * (1.70 - 0.40 * fragility + 0.20 * slip_risk)),
        )
        deadband = _positive(
            context.get("deadband"),
            max(0.35, min(1.2, 0.45 + 0.55 * fragility)),
        )
        max_step_per_tick = _positive(
            context.get("max_step_per_tick"),
            max(1.0, min(4.0, 3.4 - 2.0 * fragility + 0.6 * slip_risk)),
        )
        move_time_ms = int(
            round(
                _positive(
                    context.get("move_time_ms"),
                    max(45.0, min(120.0, 60.0 + 45.0 * fragility)),
                )
            )
        )
        poll_period_ms = int(
            round(
                _positive(
                    context.get("poll_period_ms"),
                    max(20.0, min(45.0, 24.0 + 14.0 * fragility)),
                )
            )
        )

        source_chain = context.get("source_chain")
        source = "default_rules"
        if isinstance(source_chain, list) and source_chain:
            source = "+".join(str(item) for item in source_chain if str(item).strip())
        elif str(raw_context or "").strip():
            source = "handoff_context_json"

        return {
            "profile_id": profile_id,
            "target_label": str(
                context.get("target_label")
                or context.get("target_hint")
                or context.get("object_type")
                or ""
            ).strip(),
            "object_type": str(context.get("object_type") or "").strip(),
            "object_profile_id": str(context.get("object_profile_id") or "").strip(),
            "material_id": str(context.get("material_id") or "").strip(),
            "preferred_strategy": str(context.get("preferred_strategy") or "").strip(),
            "preferred_grasp_family": str(context.get("preferred_grasp_family") or "").strip(),
            "source": source,
            "kp": round(kp, 3),
            "ki": round(ki, 3),
            "kd": round(kd, 3),
            "target_force": round(target_force, 2),
            "target_force_range": [
                round(_positive(context.get("target_force_min"), target_force), 2),
                round(_positive(context.get("target_force_max"), target_force), 2),
            ],
            "contact_threshold": round(contact_threshold, 2),
            "safety_max": round(safety_max, 2),
            "deadband": round(deadband, 3),
            "max_step_per_tick": round(max_step_per_tick, 3),
            "move_time_ms": int(move_time_ms),
            "poll_period_ms": int(poll_period_ms),
            "confidence": round(confidence, 2),
            "updated_at": time.time(),
            "raw": context,
        }

    def _build_planner_features(
        self,
        *,
        goal_context: dict[str, Any],
        profile: dict[str, Any],
        resolved_context: dict[str, Any],
    ) -> dict[str, Any]:
        handoff_context = _safe_json_loads(str(goal_context.get("handoff_context_json") or "").strip())
        raw_goal = _safe_json_loads(str(goal_context.get("raw_json") or "").strip())
        query_tokens = _normalized_aliases(
            goal_context.get("target_label"),
            goal_context.get("target_hint"),
            goal_context.get("target_part_query"),
            handoff_context.get("object"),
            handoff_context.get("object_type"),
            raw_goal.get("target_label"),
            raw_goal.get("target_hint"),
            profile.get("target_label"),
            profile.get("object_type"),
            resolved_context.get("matched_object_label"),
            resolved_context.get("query_tokens"),
        )
        material_tokens = _normalized_aliases(
            profile.get("material_id"),
            handoff_context.get("material"),
            handoff_context.get("material_id"),
            raw_goal.get("material"),
            raw_goal.get("material_id"),
        )
        return {
            "target_label": str(goal_context.get("target_label") or profile.get("target_label") or "").strip(),
            "target_hint": str(goal_context.get("target_hint") or "").strip(),
            "target_part_query": str(goal_context.get("target_part_query") or "").strip(),
            "grasp_region": str(goal_context.get("grasp_region") or "").strip(),
            "preferred_grasp_family": str(
                goal_context.get("preferred_grasp_family")
                or profile.get("preferred_grasp_family")
                or ""
            ).strip(),
            "preferred_strategy": str(profile.get("preferred_strategy") or "").strip(),
            "object_profile_id": str(profile.get("object_profile_id") or "").strip(),
            "material_id": str(profile.get("material_id") or "").strip(),
            "confidence": round(float(goal_context.get("confidence", 0.0) or 0.0), 4),
            "query_tokens": query_tokens,
            "material_tokens": material_tokens,
        }

    def _estimate_db_planner_confidence(self, profile: dict[str, Any]) -> float:
        score = 0.35 + 0.25 * _clamp01(profile.get("confidence"), 0.5)
        if str(profile.get("object_profile_id") or "").strip():
            score += 0.22
        elif str(profile.get("material_id") or "").strip():
            score += 0.14
        if bool(profile.get("rollup_applied")):
            success_count = int(profile.get("rollup_success_count", 0) or 0)
            score += min(0.18, 0.06 * float(max(1, success_count)))
        return round(min(0.98, max(0.2, score)), 3)

    def _apply_retrieval_augmented_plan(
        self,
        *,
        goal_context: dict[str, Any],
        db_profile: dict[str, Any],
    ) -> dict[str, Any]:
        retrieval = self._find_retrieval_neighbors(
            query_features=(
                copy.deepcopy(db_profile.get("planner_features"))
                if isinstance(db_profile.get("planner_features"), dict)
                else {}
            ),
            preferred_strategy=str(db_profile.get("preferred_strategy") or "").strip(),
        )
        updated = dict(db_profile)
        updated["planner_strategy"] = PLANNER_RETRIEVAL_AUG
        updated["promote_rollup"] = False
        updated["retrieval"] = retrieval
        if not bool(retrieval.get("applied")):
            updated["planner_source"] = "retrieval_fallback_db"
            updated["planner_confidence"] = round(
                max(
                    0.18,
                    min(
                        float(db_profile.get("planner_confidence", 0.0) or 0.0),
                        float(retrieval.get("confidence", 0.0) or 0.0),
                    ),
                ),
                3,
            )
            updated["updated_at"] = time.time()
            return updated

        aggregated_params = (
            retrieval.get("aggregated_params")
            if isinstance(retrieval.get("aggregated_params"), dict)
            else {}
        )
        neighbor_count = int(retrieval.get("neighbor_count", 0) or 0)
        retrieval_confidence = float(retrieval.get("confidence", 0.0) or 0.0)
        fragility = _clamp01(
            (
                updated.get("raw", {})
                if isinstance(updated.get("raw"), dict)
                else {}
            ).get("fragility"),
            0.45,
        )
        blend_alpha = max(
            0.16,
            min(
                0.62,
                0.18
                + 0.08 * float(max(0, neighbor_count - 1))
                + 0.24 * retrieval_confidence
                - 0.08 * fragility,
            ),
        )
        retrieval["blend_alpha"] = round(blend_alpha, 4)
        for name in PLANNER_NUMERIC_FIELDS:
            if name not in aggregated_params:
                continue
            try:
                current_value = float(updated.get(name))
                retrieval_value = float(aggregated_params.get(name))
            except Exception:
                continue
            blended = (1.0 - blend_alpha) * current_value + blend_alpha * retrieval_value
            if name in {"move_time_ms", "poll_period_ms"}:
                updated[name] = int(round(blended))
            else:
                updated[name] = round(blended, 3)
        force_range = aggregated_params.get("target_force_range")
        if isinstance(force_range, list) and len(force_range) == 2:
            try:
                updated["target_force_range"] = [
                    round(float(force_range[0]), 2),
                    round(float(force_range[1]), 2),
                ]
            except Exception:
                pass
        updated["planner_source"] = "database_plus_retrieval"
        updated["planner_confidence"] = round(
            min(
                0.99,
                max(
                    float(db_profile.get("planner_confidence", 0.0) or 0.0),
                    0.28 + 0.55 * retrieval_confidence,
                ),
            ),
            3,
        )
        updated["source"] = f"{updated.get('source', 'default_rules')}+retrieval"
        updated["updated_at"] = time.time()
        return updated

    def _find_retrieval_neighbors(
        self,
        *,
        query_features: dict[str, Any],
        preferred_strategy: str,
    ) -> dict[str, Any]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    trial_id,
                    created_at,
                    target_label,
                    strategy,
                    planner_strategy,
                    object_profile_id,
                    material_profile_id,
                    params_json,
                    metrics_json,
                    planner_feature_json
                FROM grasp_trials
                WHERE success = 1 AND phase = 'completed'
                ORDER BY created_at DESC
                LIMIT 128
                """
            ).fetchall()

        neighbors: list[dict[str, Any]] = []
        for row in rows:
            candidate_features, candidate_profile, candidate_metrics = self._trial_feature_from_row(row)
            similarity, evidence = self._planner_similarity_score(query_features, candidate_features)
            if similarity < 0.35:
                continue
            weight = round(
                max(
                    0.05,
                    similarity * self._planner_neighbor_quality_bonus(candidate_metrics),
                ),
                4,
            )
            neighbors.append(
                {
                    "trial_id": int(row["trial_id"] or 0),
                    "created_at": float(row["created_at"] or 0.0),
                    "target_label": str(row["target_label"] or "").strip(),
                    "strategy": str(row["strategy"] or "").strip(),
                    "planner_strategy": normalize_planner_strategy(row["planner_strategy"]),
                    "object_profile_id": str(row["object_profile_id"] or "").strip(),
                    "material_id": str(row["material_profile_id"] or "").strip(),
                    "score": round(similarity, 4),
                    "weight": weight,
                    "evidence": evidence,
                    "profile": candidate_profile,
                    "metrics": candidate_metrics,
                }
            )

        neighbors.sort(
            key=lambda item: (
                float(item.get("score", 0.0)),
                float(item.get("weight", 0.0)),
                float(item.get("created_at", 0.0)),
            ),
            reverse=True,
        )
        top_neighbors = neighbors[:6]
        aggregated_params = self._aggregate_neighbor_params(top_neighbors)
        top_score = float(top_neighbors[0].get("score", 0.0) or 0.0) if top_neighbors else 0.0
        same_object_hits = sum(
            1
            for item in top_neighbors
            if bool((item.get("evidence") or {}).get("object_profile_match"))
        )
        same_material_hits = sum(
            1
            for item in top_neighbors
            if bool((item.get("evidence") or {}).get("material_match"))
        )
        applied = bool(aggregated_params) and bool(top_neighbors) and (
            top_score >= 0.62
            or same_object_hits >= 1
            or (same_material_hits >= 2 and top_score >= 0.46)
            or (
                len(top_neighbors) >= 3
                and preferred_strategy
                and sum(1 for item in top_neighbors if str(item.get("strategy") or "").strip() == preferred_strategy) >= 2
                and top_score >= 0.44
            )
        )
        if len(top_neighbors) == 1 and top_score < 0.72:
            applied = False
        confidence = round(
            min(
                0.96,
                0.18 + 0.44 * top_score + 0.07 * float(len(top_neighbors)),
            ),
            3,
        )
        return {
            "applied": applied,
            "fallback_used": not applied,
            "neighbor_count": len(top_neighbors),
            "top_score": round(top_score, 4),
            "confidence": confidence,
            "aggregated_params": aggregated_params,
            "neighbors": [
                {
                    key: value
                    for key, value in item.items()
                    if key not in {"profile", "metrics"}
                }
                for item in top_neighbors
            ],
            "blend_alpha": round(
                max(0.0, min(0.62, 0.18 + 0.08 * float(max(0, len(top_neighbors) - 1)) + 0.24 * confidence)),
                4,
            ),
        }

    def _trial_feature_from_row(
        self,
        row: sqlite3.Row,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        params = _safe_json_loads(str(row["params_json"] or "{}"))
        profile = params.get("profile") if isinstance(params.get("profile"), dict) else {}
        goal = params.get("goal") if isinstance(params.get("goal"), dict) else {}
        stored_features = _safe_json_loads(str(row["planner_feature_json"] or "{}"))
        if stored_features:
            features = stored_features
        else:
            features = self._build_planner_features(
                goal_context=goal,
                profile=profile,
                resolved_context=profile.get("raw") if isinstance(profile.get("raw"), dict) else {},
            )
        metrics = _safe_json_loads(str(row["metrics_json"] or "{}"))
        key_metrics = metrics.get("key_metrics") if isinstance(metrics.get("key_metrics"), dict) else {}
        return features, profile, key_metrics

    def _planner_similarity_score(
        self,
        query_features: dict[str, Any],
        candidate_features: dict[str, Any],
    ) -> tuple[float, dict[str, Any]]:
        score = 0.0
        weight = 0.0
        object_profile_match = False
        material_match = False

        query_object = str(query_features.get("object_profile_id") or "").strip()
        candidate_object = str(candidate_features.get("object_profile_id") or "").strip()
        if query_object or candidate_object:
            weight += 1.0
            object_profile_match = bool(query_object and query_object == candidate_object)
            if object_profile_match:
                score += 1.0

        query_material = str(query_features.get("material_id") or "").strip()
        candidate_material = str(candidate_features.get("material_id") or "").strip()
        if query_material or candidate_material:
            weight += 0.65
            material_match = bool(query_material and query_material == candidate_material)
            if material_match:
                score += 0.65

        query_tokens = set(_normalized_aliases(query_features.get("query_tokens")))
        candidate_tokens = set(_normalized_aliases(candidate_features.get("query_tokens")))
        token_score = 0.0
        if query_tokens or candidate_tokens:
            union = query_tokens | candidate_tokens
            if union:
                token_score = float(len(query_tokens & candidate_tokens)) / float(len(union))
            weight += 0.55
            score += 0.55 * token_score

        for field_name, field_weight in (
            ("preferred_strategy", 0.35),
            ("preferred_grasp_family", 0.25),
            ("grasp_region", 0.15),
            ("target_part_query", 0.10),
        ):
            query_value = str(query_features.get(field_name) or "").strip()
            candidate_value = str(candidate_features.get(field_name) or "").strip()
            if not query_value and not candidate_value:
                continue
            weight += field_weight
            if query_value and query_value == candidate_value:
                score += field_weight

        try:
            confidence_delta = abs(
                float(query_features.get("confidence", 0.0) or 0.0)
                - float(candidate_features.get("confidence", 0.0) or 0.0)
            )
        except Exception:
            confidence_delta = 0.0
        weight += 0.12
        score += 0.12 * max(0.0, 1.0 - min(1.0, confidence_delta))

        normalized_score = 0.0 if weight <= 0.0 else score / weight
        return round(normalized_score, 4), {
            "object_profile_match": object_profile_match,
            "material_match": material_match,
            "token_overlap": len(query_tokens & candidate_tokens),
            "query_token_count": len(query_tokens),
            "candidate_token_count": len(candidate_tokens),
        }

    def _planner_neighbor_quality_bonus(self, key_metrics: dict[str, Any]) -> float:
        verification_score = _clamp01(key_metrics.get("verification_score"), 0.5)
        grasp_quality = _clamp01(key_metrics.get("best_grasp_quality"), 0.5)
        grounding_confidence = _clamp01(key_metrics.get("grounding_confidence"), 0.5)
        retry_count = max(0.0, float(key_metrics.get("retry_count", 0) or 0.0))
        return max(
            0.75,
            0.92
            + 0.14 * verification_score
            + 0.10 * grasp_quality
            + 0.08 * grounding_confidence
            - 0.05 * min(3.0, retry_count),
        )

    def _aggregate_neighbor_params(self, neighbors: list[dict[str, Any]]) -> dict[str, Any]:
        if not neighbors:
            return {}
        weighted_sums: dict[str, float] = {}
        weighted_counts: dict[str, float] = {}
        target_force_values: list[float] = []
        for item in neighbors:
            profile = item.get("profile") if isinstance(item.get("profile"), dict) else {}
            weight = max(0.05, float(item.get("weight", 0.0) or 0.0))
            for name in PLANNER_NUMERIC_FIELDS:
                try:
                    value = float(profile.get(name))
                except Exception:
                    continue
                weighted_sums[name] = weighted_sums.get(name, 0.0) + weight * value
                weighted_counts[name] = weighted_counts.get(name, 0.0) + weight
                if name == "target_force":
                    target_force_values.append(value)
        aggregated: dict[str, Any] = {}
        for name, total in weighted_sums.items():
            denom = weighted_counts.get(name, 0.0)
            if denom <= 0.0:
                continue
            value = total / denom
            aggregated[name] = int(round(value)) if name in {"move_time_ms", "poll_period_ms"} else round(value, 4)
        if target_force_values:
            aggregated["target_force_range"] = [
                round(min(target_force_values), 2),
                round(max(target_force_values), 2),
            ]
        return aggregated

    def _apply_rollup(self, profile: dict[str, Any]) -> dict[str, Any]:
        rollup = self._lookup_best_rollup(profile)
        if not rollup:
            profile["rollup_applied"] = False
            return profile
        stats = rollup.get("stats", {})
        success_count = int(stats.get("success_count", 0) or 0)
        success_rate = float(stats.get("success_rate", 0.0) or 0.0)
        if success_count < self.aggregation_min_success_trials:
            profile["rollup_applied"] = False
            profile["rollup_key"] = str(rollup.get("rollup_key") or "")
            profile["rollup_success_count"] = success_count
            profile["rollup_success_rate"] = round(success_rate, 3)
            return profile
        recommended = stats.get("recommended_params")
        if not isinstance(recommended, dict) or not recommended:
            profile["rollup_applied"] = False
            return profile

        alpha = min(0.85, 0.35 + 0.1 * max(0, success_count - self.aggregation_min_success_trials))
        updated = dict(profile)
        numeric_params = (
            "target_force",
            "contact_threshold",
            "safety_max",
            "kp",
            "ki",
            "kd",
            "deadband",
            "max_step_per_tick",
            "move_time_ms",
            "poll_period_ms",
        )
        for name in numeric_params:
            if name not in recommended:
                continue
            try:
                rollup_value = float(recommended.get(name))
                current_value = float(updated.get(name))
            except Exception:
                continue
            blended = (1.0 - alpha) * current_value + alpha * rollup_value
            if name in {"move_time_ms", "poll_period_ms"}:
                updated[name] = int(round(blended))
            else:
                updated[name] = round(blended, 3)

        updated["source"] = f"{updated.get('source', 'default_rules')}+rollup"
        updated["rollup_applied"] = True
        updated["rollup_key"] = str(rollup.get("rollup_key") or "")
        updated["rollup_success_count"] = success_count
        updated["rollup_success_rate"] = round(success_rate, 3)
        updated["rollup_recommended_params"] = recommended
        return updated

    def _lookup_best_rollup(self, profile: dict[str, Any]) -> dict[str, Any] | None:
        candidates: list[str] = []
        strategy = str(profile.get("preferred_strategy") or "").strip()
        object_profile_id = str(profile.get("object_profile_id") or "").strip()
        material_id = str(profile.get("material_id") or "").strip()
        if object_profile_id:
            candidates.append(self._make_rollup_key("object", object_profile_id, strategy))
        if material_id:
            candidates.append(self._make_rollup_key("material", material_id, strategy))
        if not candidates:
            return None
        with self._connect() as connection:
            for key in candidates:
                row = connection.execute(
                    "SELECT rollup_key, stats_json FROM profile_rollups WHERE rollup_key = ?",
                    (key,),
                ).fetchone()
                if row is None:
                    continue
                stats = _safe_json_loads(str(row["stats_json"] or "{}"))
                return {
                    "rollup_key": str(row["rollup_key"] or ""),
                    "stats": stats,
                }
        return None

    def _write_trial_artifact(
        self,
        *,
        goal_context: dict[str, Any],
        status_context: dict[str, Any],
        profile: dict[str, Any],
        payload: dict[str, Any] | None,
        key_metrics: dict[str, Any],
    ) -> str:
        if self.trial_artifact_dir is None or payload is None:
            return ""
        goal_id = str(goal_context.get("goal_id") or "goal").strip() or "goal"
        phase = str(status_context.get("phase") or "unknown").strip() or "unknown"
        now_sec = time.time()
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now_sec))
        stamp_ms = int(round((now_sec - int(now_sec)) * 1000.0))
        safe_goal = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in goal_id)[:48]
        safe_phase = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in phase)[:24]
        path = self.trial_artifact_dir / f"{stamp}-{stamp_ms:03d}_{safe_goal}_{safe_phase}.json"
        artifact = {
            "schema_version": "grasp_trial_artifact.v1",
            "saved_at": now_sec,
            "goal": copy.deepcopy(goal_context),
            "status": copy.deepcopy(status_context),
            "profile": copy.deepcopy(profile),
            "key_metrics": copy.deepcopy(key_metrics),
            "payload": copy.deepcopy(payload),
        }
        path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)

    def _record_planner_decision(
        self,
        connection: sqlite3.Connection,
        *,
        status_signature: str,
        goal_context: dict[str, Any],
        status_context: dict[str, Any],
        profile: dict[str, Any],
        key_metrics: dict[str, Any],
    ) -> None:
        retrieval = profile.get("retrieval") if isinstance(profile.get("retrieval"), dict) else {}
        db_prior = profile.get("db_prior") if isinstance(profile.get("db_prior"), dict) else {}
        planner_features = (
            profile.get("planner_features")
            if isinstance(profile.get("planner_features"), dict)
            else {}
        )
        outcome_json = {
            "status": copy.deepcopy(status_context),
            "key_metrics": copy.deepcopy(key_metrics),
        }
        connection.execute(
            """
            INSERT OR IGNORE INTO planner_decisions(
                created_at,
                status_signature,
                goal_id,
                target_label,
                grasp_strategy,
                planner_strategy,
                planner_confidence,
                retrieval_applied,
                promote_rollup,
                object_profile_id,
                material_profile_id,
                feature_json,
                db_prior_json,
                retrieval_json,
                final_profile_json,
                outcome_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                time.time(),
                status_signature,
                str(goal_context.get("goal_id") or "").strip(),
                str(
                    goal_context.get("target_label")
                    or status_context.get("target_label")
                    or profile.get("target_label")
                    or ""
                ).strip(),
                str(profile.get("preferred_strategy") or "").strip(),
                normalize_planner_strategy(profile.get("planner_strategy")),
                float(profile.get("planner_confidence", 0.0) or 0.0),
                1 if bool(retrieval.get("applied")) else 0,
                1 if bool(profile.get("promote_rollup", False)) else 0,
                str(profile.get("object_profile_id") or "").strip(),
                str(profile.get("material_id") or "").strip(),
                json.dumps(planner_features, ensure_ascii=False, separators=(",", ":")),
                json.dumps(db_prior, ensure_ascii=False, separators=(",", ":")),
                json.dumps(retrieval, ensure_ascii=False, separators=(",", ":")),
                json.dumps(profile, ensure_ascii=False, separators=(",", ":")),
                json.dumps(outcome_json, ensure_ascii=False, separators=(",", ":")),
            ),
        )

    def _refresh_rollups_for_profile(self, profile: dict[str, Any]) -> None:
        strategy = str(profile.get("preferred_strategy") or "").strip()
        object_profile_id = str(profile.get("object_profile_id") or "").strip()
        material_id = str(profile.get("material_id") or "").strip()
        if object_profile_id:
            self._recompute_rollup(scope="object", scope_id=object_profile_id, strategy=strategy)
        if material_id:
            self._recompute_rollup(scope="material", scope_id=material_id, strategy=strategy)

    def _recompute_rollup(self, *, scope: str, scope_id: str, strategy: str) -> None:
        normalized_scope = str(scope or "").strip()
        normalized_id = str(scope_id or "").strip()
        normalized_strategy = str(strategy or "").strip()
        if not normalized_scope or not normalized_id:
            return
        query = ""
        params: tuple[Any, ...]
        if normalized_scope == "object":
            query = """
                SELECT success, params_json, metrics_json
                FROM grasp_trials
                WHERE object_profile_id = ? AND strategy = ? AND COALESCE(promote_rollup, 1) = 1
            """
            params = (normalized_id, normalized_strategy)
        else:
            query = """
                SELECT success, params_json, metrics_json
                FROM grasp_trials
                WHERE material_profile_id = ? AND strategy = ? AND COALESCE(promote_rollup, 1) = 1
            """
            params = (normalized_id, normalized_strategy)

        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
            if not rows:
                connection.execute(
                    "DELETE FROM profile_rollups WHERE rollup_key = ?",
                    (self._make_rollup_key(normalized_scope, normalized_id, normalized_strategy),),
                )
                connection.commit()
                return

            total_count = len(rows)
            success_rows = [row for row in rows if int(row["success"] or 0) == 1]
            success_count = len(success_rows)
            success_rate = float(success_count) / float(max(1, total_count))
            recommended_params = self._aggregate_params(success_rows)
            metrics_summary = self._aggregate_metrics(success_rows)

            stats = {
                "scope": normalized_scope,
                "scope_id": normalized_id,
                "strategy": normalized_strategy,
                "total_count": total_count,
                "success_count": success_count,
                "success_rate": round(success_rate, 4),
                "recommended_params": recommended_params,
                "metrics_summary": metrics_summary,
                "updated_at": time.time(),
            }
            rollup_artifact_path = self._write_rollup_artifact(
                scope=normalized_scope,
                scope_id=normalized_id,
                strategy=normalized_strategy,
                stats=stats,
            )
            if rollup_artifact_path:
                stats["artifact_path"] = rollup_artifact_path
            connection.execute(
                """
                INSERT INTO profile_rollups(rollup_key, object_profile_id, material_profile_id, strategy, stats_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(rollup_key) DO UPDATE SET
                    object_profile_id=excluded.object_profile_id,
                    material_profile_id=excluded.material_profile_id,
                    strategy=excluded.strategy,
                    stats_json=excluded.stats_json,
                    updated_at=excluded.updated_at
                """,
                (
                    self._make_rollup_key(normalized_scope, normalized_id, normalized_strategy),
                    normalized_id if normalized_scope == "object" else None,
                    normalized_id if normalized_scope == "material" else None,
                    normalized_strategy,
                    json.dumps(stats, ensure_ascii=False, separators=(",", ":")),
                    time.time(),
                ),
            )
            connection.commit()

    def _extract_key_metrics(
        self,
        *,
        status_context: dict[str, Any],
        payload: dict[str, Any] | None,
    ) -> dict[str, Any]:
        payload = payload if isinstance(payload, dict) else {}
        tactile_summary = (
            status_context.get("tactile_summary")
            if isinstance(status_context.get("tactile_summary"), dict)
            else payload.get("tactile_summary")
        )
        tactile_summary = tactile_summary if isinstance(tactile_summary, dict) else {}
        pick_summary = (
            status_context.get("pick_status_summary")
            if isinstance(status_context.get("pick_status_summary"), dict)
            else payload.get("pick_status_summary")
        )
        pick_summary = pick_summary if isinstance(pick_summary, dict) else {}
        status_events = payload.get("status_events") if isinstance(payload.get("status_events"), list) else []
        trial_start_time = float(payload.get("trial_start_time", 0.0) or 0.0)
        trial_end_time = float(payload.get("trial_end_time", 0.0) or 0.0)
        trial_duration_sec = max(0.0, trial_end_time - trial_start_time) if trial_end_time > 0.0 else 0.0
        return {
            "success": bool(status_context.get("success")),
            "phase": str(status_context.get("phase") or "").strip(),
            "error_code": str(status_context.get("error_code") or "").strip(),
            "retry_count": int(status_context.get("retry_count") or 0),
            "progress": round(float(status_context.get("progress", 0.0) or 0.0), 4),
            "verification_score": round(float(status_context.get("verification_score", 0.0) or 0.0), 4),
            "best_grasp_quality": round(float(status_context.get("best_grasp_quality", 0.0) or 0.0), 4),
            "best_affordance_score": round(float(status_context.get("best_affordance_score", 0.0) or 0.0), 4),
            "grounding_confidence": round(float(status_context.get("grounding_confidence", 0.0) or 0.0), 4),
            "trial_duration_sec": round(trial_duration_sec, 4),
            "status_event_count": len(status_events),
            "pick_status_event_count": int(pick_summary.get("event_count", 0) or 0),
            "pick_final_phase": str(pick_summary.get("final_phase") or "").strip(),
            "planning_progress_percent_max": int(pick_summary.get("max_progress_percent", 0) or 0),
            "tactile_sample_count": int(tactile_summary.get("sample_count", 0) or 0),
            "tactile_duration_sec": round(float(tactile_summary.get("duration_sec", 0.0) or 0.0), 4),
            "mean_total_fz": round(float(tactile_summary.get("mean_total_fz", 0.0) or 0.0), 4),
            "std_total_fz": round(float(tactile_summary.get("std_total_fz", 0.0) or 0.0), 4),
            "peak_total_fz": round(float(tactile_summary.get("peak_total_fz", 0.0) or 0.0), 4),
            "p90_total_fz": round(float(tactile_summary.get("p90_total_fz", 0.0) or 0.0), 4),
            "p95_total_fz": round(float(tactile_summary.get("p95_total_fz", 0.0) or 0.0), 4),
            "final_total_fz": round(float(tactile_summary.get("final_total_fz", 0.0) or 0.0), 4),
            "contact_ratio": round(float(tactile_summary.get("contact_ratio", 0.0) or 0.0), 4),
            "first_contact_sec": round(float(tactile_summary.get("first_contact_sec", -1.0) or -1.0), 4),
            "source_connected_ratio": round(float(tactile_summary.get("source_connected_ratio", 0.0) or 0.0), 4),
            "avg_publish_rate_hz": round(float(tactile_summary.get("avg_publish_rate_hz", 0.0) or 0.0), 4),
            "avg_transport_rate_hz": round(float(tactile_summary.get("avg_transport_rate_hz", 0.0) or 0.0), 4),
            "peak_contact_score": round(float(tactile_summary.get("peak_contact_score", 0.0) or 0.0), 4),
        }

    def _write_rollup_artifact(
        self,
        *,
        scope: str,
        scope_id: str,
        strategy: str,
        stats: dict[str, Any],
    ) -> str:
        if self.rollup_artifact_dir is None:
            return ""
        safe_scope = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(scope or "").strip())[:24]
        safe_scope_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(scope_id or "").strip())[:48]
        safe_strategy = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(strategy or "").strip())[:32]
        filename = f"{safe_scope}__{safe_scope_id}__{safe_strategy or 'default'}.json"
        path = self.rollup_artifact_dir / filename
        artifact = {
            "schema_version": "grasp_profile_rollup.v1",
            "saved_at": time.time(),
            "rollup_key": self._make_rollup_key(scope, scope_id, strategy),
            "stats": copy.deepcopy(stats),
        }
        path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)

    def _aggregate_params(self, rows: list[sqlite3.Row]) -> dict[str, Any]:
        numeric_params = (
            "target_force",
            "contact_threshold",
            "safety_max",
            "kp",
            "ki",
            "kd",
            "deadband",
            "max_step_per_tick",
            "move_time_ms",
            "poll_period_ms",
        )
        buckets: dict[str, list[float]] = {name: [] for name in numeric_params}
        for row in rows:
            parsed = _safe_json_loads(str(row["params_json"] or "{}"))
            profile = parsed.get("profile") if isinstance(parsed.get("profile"), dict) else {}
            for name in numeric_params:
                try:
                    buckets[name].append(float(profile.get(name)))
                except Exception:
                    continue
        aggregated: dict[str, Any] = {}
        for name, values in buckets.items():
            if not values:
                continue
            value = statistics.median(values)
            aggregated[name] = int(round(value)) if name in {"move_time_ms", "poll_period_ms"} else round(value, 4)
        return aggregated

    def _aggregate_metrics(self, rows: list[sqlite3.Row]) -> dict[str, Any]:
        buckets: dict[str, list[float]] = {
            "verification_score": [],
            "best_grasp_quality": [],
            "best_affordance_score": [],
            "grounding_confidence": [],
            "retry_count": [],
            "trial_duration_sec": [],
            "planning_progress_percent_max": [],
            "peak_total_fz": [],
            "final_total_fz": [],
            "mean_total_fz": [],
            "std_total_fz": [],
            "contact_ratio": [],
            "first_contact_sec": [],
            "peak_contact_score": [],
        }
        for row in rows:
            parsed = _safe_json_loads(str(row["metrics_json"] or "{}"))
            key_metrics = parsed.get("key_metrics") if isinstance(parsed.get("key_metrics"), dict) else {}
            tactile_summary = parsed.get("tactile_summary") if isinstance(parsed.get("tactile_summary"), dict) else {}
            for key, bucket in buckets.items():
                source = key_metrics if key in key_metrics else tactile_summary
                try:
                    value = float(source.get(key, 0.0) or 0.0)
                except Exception:
                    continue
                if key == "first_contact_sec" and value < 0.0:
                    continue
                bucket.append(value)
        result: dict[str, Any] = {}
        for key, values in buckets.items():
            if not values:
                continue
            result[f"{key}_mean"] = round(statistics.fmean(values), 4)
            result[f"{key}_median"] = round(statistics.median(values), 4)
        return result

    def _make_rollup_key(self, scope: str, scope_id: str, strategy: str) -> str:
        normalized_scope = str(scope or "").strip()
        normalized_id = str(scope_id or "").strip()
        normalized_strategy = str(strategy or "").strip()
        return f"{normalized_scope}::{normalized_id}::{normalized_strategy}"
