import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { BusyAction } from "./appHelpers";
import { formatNumber, formatTimestamp } from "./appHelpers";
import { EmptyState, Panel, StatusPill } from "./appUi";
import type { GripperTuningState, UiState } from "./types";

const JOINT_MIN_DEG = -180;
const JOINT_MAX_DEG = 270;
const AUTO_APPLY_DEBOUNCE_MS = 180;
const AUTO_ALIGN_EPSILON_DEG = 0.75;
const TUNING_ALIGN_EPSILON = 0.01;
const FORCE_TRACE_WINDOW_SEC = 15;
const FORCE_TRACE_MAX_POINTS = 360;
const FORCE_TRACE_WIDTH = 920;
const FORCE_TRACE_HEIGHT = 240;
const FORCE_TRACE_PADDING = { left: 52, right: 18, top: 18, bottom: 34 } as const;
const ROBOT_SECTION_COPY = {
  manual: "Joint jog, home, and basic arm commands.",
  gripper: "Quick grip control, runtime tuning, and calibration actions.",
  strategy: "Strategy selection and current hardware feedback.",
} as const;
const ROBOT_SECTIONS = [
  { id: "manual", label: "Manual", description: "关节手动控制、回零和基础动作。" },
  { id: "gripper", label: "Gripper", description: "夹爪快速控制、运行调参和标定动作。" },
  { id: "strategy", label: "Strategy", description: "策略选择和当前硬件状态回看。" },
] as const;

type RobotSection = (typeof ROBOT_SECTIONS)[number]["id"];
type GripperTuningDraft = Pick<
  GripperTuningState,
  | "servo_id"
  | "tactile_dev_addr"
  | "close_direction"
  | "open_limit_raw"
  | "close_limit_raw"
  | "kp"
  | "ki"
  | "kd"
  | "contact_threshold"
  | "hard_force_limit"
  | "deadband"
  | "max_step_per_tick"
  | "poll_period_ms"
  | "move_time_ms"
>;

type ForceTraceSample = {
  ts: number;
  filtered: number;
  raw: number;
  target: number;
};

function clampAngle(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.max(JOINT_MIN_DEG, Math.min(JOINT_MAX_DEG, value));
}

function normalizeInteger(value: number, fallback: number, minimum = 0): number {
  if (!Number.isFinite(value)) return fallback;
  return Math.max(minimum, Math.round(value));
}

function normalizeFloat(value: number, fallback: number, minimum = 0): number {
  if (!Number.isFinite(value)) return fallback;
  return Math.max(minimum, value);
}

function createTuningDraft(runtime: GripperTuningState): GripperTuningDraft {
  return {
    servo_id: normalizeInteger(runtime.servo_id, 6, 1),
    tactile_dev_addr: normalizeInteger(runtime.tactile_dev_addr, 1, 1),
    close_direction: runtime.close_direction === -1 ? -1 : 1,
    open_limit_raw: normalizeInteger(runtime.open_limit_raw, 900, 0),
    close_limit_raw: normalizeInteger(runtime.close_limit_raw, 3100, 0),
    kp: normalizeFloat(runtime.kp, 0.8),
    ki: normalizeFloat(runtime.ki, 0.08),
    kd: normalizeFloat(runtime.kd, 0.0),
    contact_threshold: normalizeFloat(runtime.contact_threshold, 4.0),
    hard_force_limit: normalizeFloat(runtime.hard_force_limit, 18.0),
    deadband: normalizeFloat(runtime.deadband, 2.0),
    max_step_per_tick: normalizeFloat(runtime.max_step_per_tick, 6.0),
    poll_period_ms: normalizeInteger(runtime.poll_period_ms, 40, 1),
    move_time_ms: normalizeInteger(runtime.move_time_ms, 100, 1),
  };
}

function isClose(a: number, b: number, epsilon = TUNING_ALIGN_EPSILON): boolean {
  return Math.abs((a || 0) - (b || 0)) <= epsilon;
}

function buildGripperRawPresets(openLimitRaw: number, closeLimitRaw: number) {
  const openRaw = normalizeInteger(openLimitRaw, 0, 0);
  const closeRaw = normalizeInteger(closeLimitRaw, 0, 0);
  const midRaw = Math.round((openRaw + closeRaw) / 2);
  return [
    { label: "Open", rawPos: openRaw },
    { label: "Half", rawPos: midRaw },
    { label: "Close", rawPos: closeRaw },
  ] as const;
}

function clampForceValue(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, value);
}

function buildTracePolyline(
  samples: ForceTraceSample[],
  selector: (sample: ForceTraceSample) => number,
  nowSec: number,
  yMax: number,
) {
  if (samples.length === 0 || yMax <= 0) return "";
  const plotWidth = FORCE_TRACE_WIDTH - FORCE_TRACE_PADDING.left - FORCE_TRACE_PADDING.right;
  const plotHeight = FORCE_TRACE_HEIGHT - FORCE_TRACE_PADDING.top - FORCE_TRACE_PADDING.bottom;
  const startSec = nowSec - FORCE_TRACE_WINDOW_SEC;
  return samples.map((sample) => {
    const x = FORCE_TRACE_PADDING.left + ((sample.ts - startSec) / FORCE_TRACE_WINDOW_SEC) * plotWidth;
    const normalized = Math.max(0, Math.min(1, selector(sample) / yMax));
    const y = FORCE_TRACE_PADDING.top + (1 - normalized) * plotHeight;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
}

export function RobotPage(props: {
  state: UiState;
  busyAction: BusyAction;
  onEnableArm: (enabled: boolean) => Promise<void>;
  onApplyJoints: (
    jointIds: number[],
    anglesDeg: number[],
    durationMs: number,
    wait: boolean,
  ) => Promise<void>;
  onApplyGripperTuning: (payload: Record<string, unknown>) => Promise<void>;
  onSaveGripperTuningDefaults: (payload: Record<string, unknown>) => Promise<void>;
  onRunGripperTuningAction: (
    action: string,
    payload?: Record<string, unknown>,
  ) => Promise<void>;
  onReturnHome: () => Promise<void>;
  onSelectStrategy: (strategyId: string) => Promise<void>;
  onSelectPlannerStrategy: (strategyId: string) => Promise<void>;
}) {
  const robot = props.state.robot;
  const armState = props.state.health.arm_state;
  const gripperRuntime = props.state.gripper_tuning!;
  const gripperProfile = props.state.gripper_profile ?? {};
  const [section, setSection] = useState<RobotSection>("manual");
  const jointIds = robot.joint_ids.length > 0 ? robot.joint_ids : [1, 2, 3, 4, 5, 6];
  const currentAngles = useMemo(
    () => jointIds.map((jointId, index) => {
      const jointAngles = armState.joint_angles ?? [];
      const rawValue = jointAngles[jointId - 1] ?? jointAngles[index] ?? 0;
      return clampAngle(Number(rawValue));
    }),
    [armState.joint_angles, jointIds],
  );
  const [draftAngles, setDraftAngles] = useState<number[]>(currentAngles);
  const [durationMs, setDurationMs] = useState(1200);
  const [draftDirty, setDraftDirty] = useState(false);
  const [autoStatus, setAutoStatus] = useState<"idle" | "pending" | "sending">("idle");
  const autoTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const autoPendingCommandRef = useRef<{ jointId: number; angleDeg: number } | null>(null);
  const autoSendingRef = useRef(false);
  const [tuningDraft, setTuningDraft] = useState<GripperTuningDraft>(() => createTuningDraft(gripperRuntime));
  const [tuningDirty, setTuningDirty] = useState(false);
  const [forceTarget, setForceTarget] = useState<number>(normalizeFloat(gripperRuntime.target_force, 8));
  const [rawPosition, setRawPosition] = useState<number>(normalizeInteger(gripperRuntime.commanded_pos_raw, 0, 0));
  const [forceTrace, setForceTrace] = useState<ForceTraceSample[]>([]);
  const lastForceTraceStampRef = useRef<number>(0);

  useEffect(() => {
    if (!draftDirty) setDraftAngles(currentAngles);
  }, [currentAngles, draftDirty]);

  useEffect(() => {
    if (!tuningDirty) setTuningDraft(createTuningDraft(gripperRuntime));
  }, [gripperRuntime, tuningDirty]);

  useEffect(() => {
    setForceTarget(normalizeFloat(gripperRuntime.target_force, 8));
  }, [gripperRuntime.target_force]);

  useEffect(() => {
    setRawPosition(normalizeInteger(gripperRuntime.commanded_pos_raw, 0, 0));
  }, [gripperRuntime.commanded_pos_raw]);

  useEffect(() => {
    if (!gripperRuntime.connection_ready) return;
    const stamp = Number(gripperRuntime.updated_at || 0);
    const effectiveStamp = stamp > 0 ? stamp : (Date.now() / 1000);
    if (effectiveStamp <= lastForceTraceStampRef.current) return;
    lastForceTraceStampRef.current = effectiveStamp;
    setForceTrace((current) => {
      const next = [
        ...current,
        {
          ts: effectiveStamp,
          filtered: clampForceValue(gripperRuntime.filtered_force),
          raw: clampForceValue(gripperRuntime.measured_force_raw),
          target: clampForceValue(gripperRuntime.target_force),
        },
      ];
      const cutoff = effectiveStamp - FORCE_TRACE_WINDOW_SEC;
      return next.filter((sample) => sample.ts >= cutoff).slice(-FORCE_TRACE_MAX_POINTS);
    });
  }, [
    gripperRuntime.connection_ready,
    gripperRuntime.filtered_force,
    gripperRuntime.measured_force_raw,
    gripperRuntime.target_force,
    gripperRuntime.updated_at,
  ]);

  const clearAutoTimer = useCallback(() => {
    if (autoTimerRef.current !== null) {
      clearTimeout(autoTimerRef.current);
      autoTimerRef.current = null;
    }
  }, []);

  const flushAutoCommand = useCallback(async function flushAutoCommandImpl() {
    clearAutoTimer();
    if (!robot.move_ready || autoSendingRef.current) return;
    const nextCommand = autoPendingCommandRef.current;
    if (nextCommand === null) {
      setAutoStatus("idle");
      return;
    }

    autoPendingCommandRef.current = null;
    autoSendingRef.current = true;
    setAutoStatus("sending");

    try {
      await props.onApplyJoints([nextCommand.jointId], [nextCommand.angleDeg], durationMs, false);
    } finally {
      autoSendingRef.current = false;
      if (autoPendingCommandRef.current !== null) void flushAutoCommandImpl();
      else setAutoStatus("idle");
    }
  }, [clearAutoTimer, durationMs, props.onApplyJoints, robot.move_ready]);

  const scheduleAutoCommand = useCallback((
    jointId: number,
    angleDeg: number,
    options?: { immediate?: boolean },
  ) => {
    if (jointId <= 0 || !robot.move_ready) return;
    autoPendingCommandRef.current = { jointId, angleDeg: clampAngle(angleDeg) };
    if (options?.immediate) {
      void flushAutoCommand();
      return;
    }
    clearAutoTimer();
    setAutoStatus("pending");
    autoTimerRef.current = setTimeout(() => {
      autoTimerRef.current = null;
      void flushAutoCommand();
    }, AUTO_APPLY_DEBOUNCE_MS);
  }, [clearAutoTimer, flushAutoCommand, robot.move_ready]);

  useEffect(() => () => {
    clearAutoTimer();
    autoPendingCommandRef.current = null;
  }, [clearAutoTimer]);

  const gripperJointId = robot.gripper_joint_id > 0 ? robot.gripper_joint_id : 6;
  const gripperIndex = Math.max(0, jointIds.findIndex((jointId) => jointId === gripperJointId));
  const gripperDraftAngle = draftAngles[gripperIndex] ?? currentAngles[gripperIndex] ?? 0;
  const gripperTestPresets = useMemo(
    () => buildGripperRawPresets(gripperRuntime.open_limit_raw, gripperRuntime.close_limit_raw),
    [gripperRuntime.close_limit_raw, gripperRuntime.open_limit_raw],
  );
  const strategyOptions = robot.strategy_options.length > 0
    ? robot.strategy_options
    : [{ id: "mainline_pick", label: "Mainline Pick", description: "Current integrated visual-language perception and grasp execution chain." }];
  const selectedStrategy = robot.selected_strategy || strategyOptions[0].id;
  const selectedStrategyMeta = strategyOptions.find((item) => item.id === selectedStrategy) ?? strategyOptions[0];
  const plannerOptions = robot.planner_options.length > 0
    ? robot.planner_options
    : [
        { id: "planner_db_only", label: "DB Only", description: "Rule and rollup planner using the curated object/material database only." },
        { id: "planner_retrieval_aug", label: "Retrieval Augmented", description: "DB prior plus nearest successful grasp trials for parameter refinement." },
      ];
  const selectedPlannerStrategy = robot.planner_strategy || plannerOptions[0].id;
  const selectedPlannerMeta = plannerOptions.find((item) => item.id === selectedPlannerStrategy) ?? plannerOptions[0];
  const retrieval = typeof gripperProfile.retrieval === "object" && gripperProfile.retrieval !== null
    ? gripperProfile.retrieval as Record<string, unknown>
    : {};
  const retrievalApplied = Boolean(gripperProfile.retrieval_applied ?? retrieval.applied);
  const retrievalNeighborCount = Number(gripperProfile.retrieval_neighbor_count ?? retrieval.neighbor_count ?? 0);
  const plannerConfidence = Number(gripperProfile.planner_confidence ?? 0);
  const targetForceRange = Array.isArray(gripperProfile.target_force_range) ? gripperProfile.target_force_range : [];
  const busy = props.busyAction !== null;
  const tuningBusy = props.busyAction === "gripper-tuning";
  const armConnected = Boolean(armState.connected);
  const armError = Boolean(armState.error);
  const armMoving = Boolean(armState.moving);
  const gripperDisabled = busy || !robot.move_ready || gripperIndex < 0;
  const gripperQuickControlDisabled = tuningBusy || !gripperRuntime.connection_ready || !gripperRuntime.supported;
  const returnHomeDisabled = busy || !robot.return_home_ready;
  const draftAligned = draftAngles.length === currentAngles.length
    && draftAngles.every((value, index) => Math.abs(value - (currentAngles[index] ?? 0)) < AUTO_ALIGN_EPSILON_DEG);
  const tuningDraftAligned = (
    tuningDraft.servo_id === normalizeInteger(gripperRuntime.servo_id, 6, 1)
    && tuningDraft.tactile_dev_addr === normalizeInteger(gripperRuntime.tactile_dev_addr, 1, 1)
    && tuningDraft.close_direction === (gripperRuntime.close_direction === -1 ? -1 : 1)
    && tuningDraft.open_limit_raw === normalizeInteger(gripperRuntime.open_limit_raw, 900, 0)
    && tuningDraft.close_limit_raw === normalizeInteger(gripperRuntime.close_limit_raw, 3100, 0)
    && isClose(tuningDraft.kp, gripperRuntime.kp)
    && isClose(tuningDraft.ki, gripperRuntime.ki)
    && isClose(tuningDraft.kd, gripperRuntime.kd)
    && isClose(tuningDraft.contact_threshold, gripperRuntime.contact_threshold)
    && isClose(tuningDraft.hard_force_limit, gripperRuntime.hard_force_limit)
    && isClose(tuningDraft.deadband, gripperRuntime.deadband)
    && isClose(tuningDraft.max_step_per_tick, gripperRuntime.max_step_per_tick)
    && tuningDraft.poll_period_ms === normalizeInteger(gripperRuntime.poll_period_ms, 40, 1)
    && tuningDraft.move_time_ms === normalizeInteger(gripperRuntime.move_time_ms, 100, 1)
  );
  const forceTraceWindow = useMemo(() => {
    const nowSec = forceTrace.length > 0
      ? forceTrace[forceTrace.length - 1].ts
      : Number(gripperRuntime.updated_at || Date.now() / 1000);
    const samples = forceTrace.filter((sample) => sample.ts >= (nowSec - FORCE_TRACE_WINDOW_SEC));
    const observedMax = samples.reduce((maxValue, sample) => Math.max(maxValue, sample.filtered, sample.raw, sample.target), 0);
    const referenceMax = Math.max(
      10,
      clampForceValue(gripperRuntime.hard_force_limit),
      clampForceValue(gripperRuntime.target_force) * 1.5,
      observedMax * 1.15,
    );
    const yMax = Math.ceil(referenceMax);
    const filteredPoints = buildTracePolyline(samples, (sample) => sample.filtered, nowSec, yMax);
    const rawPoints = buildTracePolyline(samples, (sample) => sample.raw, nowSec, yMax);
    const plotWidth = FORCE_TRACE_WIDTH - FORCE_TRACE_PADDING.left - FORCE_TRACE_PADDING.right;
    const plotHeight = FORCE_TRACE_HEIGHT - FORCE_TRACE_PADDING.top - FORCE_TRACE_PADDING.bottom;
    const targetValue = clampForceValue(gripperRuntime.target_force);
    const targetY = FORCE_TRACE_PADDING.top + (1 - Math.max(0, Math.min(1, targetValue / yMax))) * plotHeight;
    const ticks = Array.from({ length: 5 }, (_, index) => {
      const ratio = index / 4;
      return {
        label: ((1 - ratio) * yMax).toFixed(0),
        y: FORCE_TRACE_PADDING.top + ratio * plotHeight,
      };
    });
    const timeTicks = [0, 5, 10, 15].map((value) => ({
      label: `-${FORCE_TRACE_WINDOW_SEC - value}s`,
      x: FORCE_TRACE_PADDING.left + (value / FORCE_TRACE_WINDOW_SEC) * plotWidth,
    }));
    return {
      samples,
      nowSec,
      yMax,
      filteredPoints,
      rawPoints,
      targetY,
      ticks,
      timeTicks,
    };
  }, [
    forceTrace,
    gripperRuntime.hard_force_limit,
    gripperRuntime.target_force,
    gripperRuntime.updated_at,
  ]);

  useEffect(() => {
    if (draftDirty && draftAligned && autoStatus === "idle") setDraftDirty(false);
  }, [autoStatus, draftAligned, draftDirty]);

  useEffect(() => {
    if (tuningDirty && tuningDraftAligned) setTuningDirty(false);
  }, [tuningDirty, tuningDraftAligned]);

  const updateJointAngle = (
    index: number,
    nextValue: number,
    options?: { autoSend?: boolean; immediate?: boolean },
  ) => {
    const clampedValue = clampAngle(nextValue);
    setDraftAngles((current) => {
      const next = [...current];
      next[index] = clampedValue;
      return next;
    });
    setDraftDirty(true);
    if (options?.autoSend) scheduleAutoCommand(jointIds[index] ?? 0, clampedValue, { immediate: options.immediate });
  };

  const syncCurrent = () => {
    clearAutoTimer();
    autoPendingCommandRef.current = null;
    setAutoStatus("idle");
    setDraftAngles(currentAngles);
    setDraftDirty(false);
  };

  const updateTuningField = <K extends keyof GripperTuningDraft>(key: K, value: GripperTuningDraft[K]) => {
    setTuningDraft((current) => ({ ...current, [key]: value }));
    setTuningDirty(true);
  };

  const syncTuningDraft = () => {
    setTuningDraft(createTuningDraft(gripperRuntime));
    setTuningDirty(false);
  };

  const copyProfileSuggestion = () => {
    setTuningDraft((current) => ({
      ...current,
      kp: Number.isFinite(Number(gripperProfile.kp)) ? normalizeFloat(Number(gripperProfile.kp), current.kp) : current.kp,
      kd: Number.isFinite(Number(gripperProfile.kd)) ? normalizeFloat(Number(gripperProfile.kd), current.kd) : current.kd,
      contact_threshold: Number.isFinite(Number(gripperProfile.contact_threshold))
        ? normalizeFloat(Number(gripperProfile.contact_threshold), current.contact_threshold)
        : current.contact_threshold,
      hard_force_limit: Number.isFinite(Number(gripperProfile.safety_max))
        ? normalizeFloat(Number(gripperProfile.safety_max), current.hard_force_limit)
        : current.hard_force_limit,
    }));
    if (Number.isFinite(Number(gripperProfile.target_force))) {
      setForceTarget(normalizeFloat(Number(gripperProfile.target_force), forceTarget));
    }
    setTuningDirty(true);
  };

  const applyTuningDraft = () => {
    void props.onApplyGripperTuning({ ...tuningDraft });
  };

  const renderManualSection = () => (
    <>
      <Panel
        title="Robot Control"
        subtitle="Manual joint control uses the same ROS2 arm pipeline as the real execution stack."
        actions={(
          <div className="button-row compact">
            <button className="ghost-button" disabled={busy || !robot.enable_ready} onClick={() => void props.onEnableArm(true)} type="button">
              {props.busyAction === "arm-enable" ? "Working..." : "Enable"}
            </button>
            <button className="ghost-button" disabled={busy || !robot.enable_ready} onClick={() => void props.onEnableArm(false)} type="button">
              Disable
            </button>
          </div>
        )}
      >
        <div className="status-row">
          <StatusPill tone={robot.control_ready ? "success" : "warn"}>Control {robot.control_ready ? "ready" : "pending"}</StatusPill>
          <StatusPill tone={armConnected ? "success" : "warn"}>Link {armConnected ? "up" : "down"}</StatusPill>
          <StatusPill tone={armMoving ? "info" : "neutral"}>Motion {armMoving ? "moving" : "idle"}</StatusPill>
          <StatusPill tone={armError ? "error" : "success"}>Arm {armError ? "fault" : "healthy"}</StatusPill>
          <StatusPill tone={draftDirty ? "warn" : "neutral"}>Draft {draftDirty ? "modified" : "synced"}</StatusPill>
          <StatusPill tone={autoStatus === "sending" ? "info" : autoStatus === "pending" ? "warn" : "success"}>Auto {autoStatus}</StatusPill>
        </div>

        <div className="metric-grid compact">
          <div className="metric-card"><span className="metric-label">Joint Count</span><strong>{robot.joint_count || jointIds.length}</strong></div>
          <div className="metric-card"><span className="metric-label">Gripper Joint</span><strong>J{gripperJointId}</strong></div>
          <div className="metric-card"><span className="metric-label">Battery</span><strong>{formatNumber(armState.battery_voltage, 2)}</strong></div>
          <div className="metric-card"><span className="metric-label">Updated</span><strong>{formatTimestamp(armState.updated_at)}</strong></div>
        </div>

        {armError && armState.error_message ? (
          <div className="muted-block"><div className="record-row"><span>Driver Error</span><strong>{armState.error_message}</strong></div></div>
        ) : null}

        <div className="button-row">
          <button className="ghost-button" disabled={busy} onClick={syncCurrent} type="button">Sync Current</button>
          <button className="ghost-button" disabled={returnHomeDisabled} onClick={() => void props.onReturnHome()} type="button">
            {props.busyAction === "return-home" ? "Returning..." : "Return Home"}
          </button>
        </div>

        <div className="robot-settings-grid">
          <label className="field-group">
            <span className="field-label">Duration (ms)</span>
            <input
              className="text-input"
              min={250}
              step={50}
              type="number"
              value={durationMs}
              onChange={(event) => setDurationMs(Math.max(250, Number(event.currentTarget.value) || 250))}
            />
          </label>
        </div>

        <div className="inline-note">Sliders auto-send through <code>/control/arm/move_joints</code> and flush the latest value on release.</div>
      </Panel>

      <Panel title="Manual Joints" subtitle="Use the slider for quick shaping and the numeric box for precise trim. Values are degrees.">
        {jointIds.length === 0 ? (
          <EmptyState title="No joints configured" message="Waiting for robot metadata from the backend." />
        ) : (
          <div className="joint-editor">
            {jointIds.map((jointId, index) => (
              <div key={jointId} className="joint-row">
                <div className="joint-labels">
                  <strong>Joint {jointId}</strong>
                  <span className="inline-note">Current {formatNumber(currentAngles[index], 2)} deg</span>
                </div>
                <input
                  className="joint-slider"
                  max={JOINT_MAX_DEG}
                  min={JOINT_MIN_DEG}
                  step={0.5}
                  type="range"
                  value={draftAngles[index] ?? 0}
                  onChange={(event) => updateJointAngle(index, Number(event.currentTarget.value), { autoSend: true })}
                  onMouseUp={() => void flushAutoCommand()}
                  onTouchEnd={() => void flushAutoCommand()}
                />
                <input
                  className="text-input joint-number"
                  max={JOINT_MAX_DEG}
                  min={JOINT_MIN_DEG}
                  step={0.5}
                  type="number"
                  value={draftAngles[index] ?? 0}
                  onBlur={(event) => updateJointAngle(index, Number(event.currentTarget.value), { autoSend: true, immediate: true })}
                  onChange={(event) => updateJointAngle(index, Number(event.currentTarget.value))}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") event.currentTarget.blur();
                  }}
                />
              </div>
            ))}
          </div>
        )}
      </Panel>
    </>
  );

  const renderGripperSection = () => (
    <>
      <Panel title="Gripper Quick Control" subtitle="Quick raw-position tests stay separate from low-level tuning.">
        <div className="status-row">
          <StatusPill tone={gripperQuickControlDisabled ? "warn" : "success"}>Channel {gripperQuickControlDisabled ? "pending" : "ready"}</StatusPill>
          <StatusPill tone={armConnected ? "success" : "warn"}>Servo {armConnected ? "reachable" : "offline"}</StatusPill>
          <StatusPill tone={gripperRuntime.runtime_ready ? "success" : gripperRuntime.connection_ready ? "warn" : "neutral"}>
            Runtime {gripperRuntime.runtime_ready ? "ready" : gripperRuntime.connection_ready ? "degraded" : "down"}
          </StatusPill>
        </div>

        <div className="metric-grid compact">
          <div className="metric-card"><span className="metric-label">Draft Angle</span><strong>{formatNumber(gripperDraftAngle, 2)}</strong></div>
          <div className="metric-card"><span className="metric-label">Current Angle</span><strong>{formatNumber(currentAngles[gripperIndex], 2)}</strong></div>
          <div className="metric-card">
            <span className="metric-label">Raw Limits</span>
            <strong>{`${formatNumber(gripperRuntime.open_limit_raw, 0)} -> ${formatNumber(gripperRuntime.close_limit_raw, 0)}`}</strong>
          </div>
          <div className="metric-card"><span className="metric-label">Runtime Mode</span><strong>{gripperRuntime.mode || "--"}</strong></div>
          <div className="metric-card"><span className="metric-label">Filtered Force</span><strong>{formatNumber(gripperRuntime.filtered_force, 2)}</strong></div>
          <div className="metric-card"><span className="metric-label">Commanded Raw</span><strong>{formatNumber(gripperRuntime.commanded_pos_raw, 0)}</strong></div>
        </div>

        <div className="button-row">
          {gripperTestPresets.map((preset) => (
            <button
              key={preset.label}
              className="ghost-button"
              disabled={gripperQuickControlDisabled}
              onClick={() => void props.onRunGripperTuningAction("set_position", {
                raw_pos: preset.rawPos,
                move_time_ms: gripperRuntime.move_time_ms || tuningDraft.move_time_ms,
              })}
              type="button"
            >
              {preset.label}
            </button>
          ))}
        </div>

        <div className="inline-note">These test buttons send raw STM32 gripper positions based on the current open/close limits. They do not use the arm joint angle mapping.</div>
      </Panel>

      <Panel
        title="Gripper Tuning"
        subtitle="These fields map to the STM32 G* command set. Force values are tactile raw counts, not Newtons. PID gains stay visible because hybrid hold uses them for post-contact trim."
        actions={(
          <div className="button-row compact">
            <button className="ghost-button" disabled={tuningBusy} onClick={() => void props.onRunGripperTuningAction("refresh")} type="button">Refresh</button>
            <button className="ghost-button" disabled={tuningBusy || !gripperProfile.profile_id} onClick={copyProfileSuggestion} type="button">Copy Profile Suggestion</button>
            <button className="ghost-button" disabled={tuningBusy} onClick={applyTuningDraft} type="button">{tuningBusy ? "Applying..." : "Apply Runtime"}</button>
            <button className="ghost-button" disabled={tuningBusy} onClick={() => void props.onSaveGripperTuningDefaults({ ...tuningDraft })} type="button">Save Default</button>
          </div>
        )}
      >
        <div className="status-row">
          <StatusPill tone={gripperRuntime.enabled ? "info" : "neutral"}>Endpoint {gripperRuntime.enabled ? "configured" : "empty"}</StatusPill>
          <StatusPill tone={gripperRuntime.connection_ready ? "success" : "warn"}>Bridge {gripperRuntime.connection_ready ? "up" : "down"}</StatusPill>
          <StatusPill tone={gripperRuntime.supported ? "success" : "warn"}>Firmware {gripperRuntime.supported ? "G* ready" : "missing G*"}</StatusPill>
          <StatusPill tone={gripperRuntime.contact_active ? "success" : "neutral"}>Contact {gripperRuntime.contact_active ? "on" : "off"}</StatusPill>
          <StatusPill tone={gripperRuntime.source_connected ? "success" : "warn"}>Tactile {gripperRuntime.source_connected ? "live" : "stale"}</StatusPill>
          <StatusPill tone={tuningDirty ? "warn" : "neutral"}>Draft {tuningDirty ? "modified" : "synced"}</StatusPill>
        </div>

        <div className="metric-grid compact">
          <div className="metric-card"><span className="metric-label">Servo ID</span><strong>{gripperRuntime.servo_id}</strong></div>
          <div className="metric-card"><span className="metric-label">Tactile Addr</span><strong>{gripperRuntime.tactile_dev_addr}</strong></div>
          <div className="metric-card"><span className="metric-label">Target Force</span><strong>{formatNumber(gripperRuntime.target_force, 2)}</strong></div>
          <div className="metric-card"><span className="metric-label">Kp / Ki / Kd</span><strong>{`${formatNumber(gripperRuntime.kp, 2)} / ${formatNumber(gripperRuntime.ki, 2)} / ${formatNumber(gripperRuntime.kd, 2)}`}</strong></div>
          <div className="metric-card"><span className="metric-label">Limits</span><strong>{`${formatNumber(gripperRuntime.open_limit_raw, 0)} -> ${formatNumber(gripperRuntime.close_limit_raw, 0)}`}</strong></div>
          <div className="metric-card"><span className="metric-label">Poll / Move</span><strong>{`${formatNumber(gripperRuntime.poll_period_ms, 0)} / ${formatNumber(gripperRuntime.move_time_ms, 0)} ms`}</strong></div>
        </div>

        {gripperProfile.profile_id ? (
          <div className="muted-block"><div className="record-row"><span>Profile</span><strong>{`${gripperProfile.profile_id} (${gripperProfile.target_label || gripperProfile.object_type || "--"})`}</strong></div></div>
        ) : null}

        {!gripperRuntime.supported ? (
          <div className="inline-note">The bridge is reachable, but the board is not returning stable <code>GSTAT</code> replies yet. That usually means the STM32 has not been flashed with the G* firmware build.</div>
        ) : null}

        {gripperRuntime.last_error ? (
          <div className="muted-block"><div className="record-row"><span>Last Error</span><strong>{gripperRuntime.last_error}</strong></div></div>
        ) : null}

        <div className="robot-form-grid">
          <label className="field-group"><span className="field-label">Servo ID</span><input className="text-input" min={1} step={1} type="number" value={tuningDraft.servo_id} onChange={(event) => updateTuningField("servo_id", normalizeInteger(Number(event.currentTarget.value), tuningDraft.servo_id, 1))} /></label>
          <label className="field-group"><span className="field-label">Tactile Addr</span><input className="text-input" min={1} step={1} type="number" value={tuningDraft.tactile_dev_addr} onChange={(event) => updateTuningField("tactile_dev_addr", normalizeInteger(Number(event.currentTarget.value), tuningDraft.tactile_dev_addr, 1))} /></label>
          <label className="field-group">
            <span className="field-label">Close Direction</span>
            <select className="select-input" value={tuningDraft.close_direction} onChange={(event) => updateTuningField("close_direction", Number(event.currentTarget.value) === -1 ? -1 : 1)}>
              <option value={1}>1 (raw increase closes)</option>
              <option value={-1}>-1 (raw decrease closes)</option>
            </select>
          </label>
          <label className="field-group"><span className="field-label">Open Limit Raw</span><input className="text-input" min={0} step={1} type="number" value={tuningDraft.open_limit_raw} onChange={(event) => updateTuningField("open_limit_raw", normalizeInteger(Number(event.currentTarget.value), tuningDraft.open_limit_raw, 0))} /></label>
          <label className="field-group"><span className="field-label">Close Limit Raw</span><input className="text-input" min={0} step={1} type="number" value={tuningDraft.close_limit_raw} onChange={(event) => updateTuningField("close_limit_raw", normalizeInteger(Number(event.currentTarget.value), tuningDraft.close_limit_raw, 0))} /></label>
          <label className="field-group"><span className="field-label">Contact Threshold</span><input className="text-input" min={0} step={0.1} type="number" value={tuningDraft.contact_threshold} onChange={(event) => updateTuningField("contact_threshold", normalizeFloat(Number(event.currentTarget.value), tuningDraft.contact_threshold))} /></label>
          <label className="field-group"><span className="field-label">Kp</span><input className="text-input" min={0} step={0.05} type="number" value={tuningDraft.kp} onChange={(event) => updateTuningField("kp", normalizeFloat(Number(event.currentTarget.value), tuningDraft.kp))} /></label>
          <label className="field-group"><span className="field-label">Ki</span><input className="text-input" min={0} step={0.01} type="number" value={tuningDraft.ki} onChange={(event) => updateTuningField("ki", normalizeFloat(Number(event.currentTarget.value), tuningDraft.ki))} /></label>
          <label className="field-group"><span className="field-label">Kd</span><input className="text-input" min={0} step={0.01} type="number" value={tuningDraft.kd} onChange={(event) => updateTuningField("kd", normalizeFloat(Number(event.currentTarget.value), tuningDraft.kd))} /></label>
          <label className="field-group"><span className="field-label">Deadband</span><input className="text-input" min={0} step={0.1} type="number" value={tuningDraft.deadband} onChange={(event) => updateTuningField("deadband", normalizeFloat(Number(event.currentTarget.value), tuningDraft.deadband))} /></label>
          <label className="field-group"><span className="field-label">Max Step</span><input className="text-input" min={0} step={0.5} type="number" value={tuningDraft.max_step_per_tick} onChange={(event) => updateTuningField("max_step_per_tick", normalizeFloat(Number(event.currentTarget.value), tuningDraft.max_step_per_tick))} /></label>
          <label className="field-group"><span className="field-label">Hard Force Limit</span><input className="text-input" min={0} step={0.5} type="number" value={tuningDraft.hard_force_limit} onChange={(event) => updateTuningField("hard_force_limit", normalizeFloat(Number(event.currentTarget.value), tuningDraft.hard_force_limit))} /></label>
          <label className="field-group"><span className="field-label">Poll Period (ms)</span><input className="text-input" min={1} step={1} type="number" value={tuningDraft.poll_period_ms} onChange={(event) => updateTuningField("poll_period_ms", normalizeInteger(Number(event.currentTarget.value), tuningDraft.poll_period_ms, 1))} /></label>
          <label className="field-group"><span className="field-label">Move Time (ms)</span><input className="text-input" min={1} step={1} type="number" value={tuningDraft.move_time_ms} onChange={(event) => updateTuningField("move_time_ms", normalizeInteger(Number(event.currentTarget.value), tuningDraft.move_time_ms, 1))} /></label>
        </div>

        <div className="button-row"><button className="ghost-button" disabled={tuningBusy} onClick={syncTuningDraft} type="button">Sync Runtime</button></div>
      </Panel>

      <Panel title="Gripper Actions" subtitle="Runtime actions stay separate from parameter edits. Target force is consumed by Hybrid Hold after contact is detected and settled.">
        <div className="metric-grid compact">
          <div className="metric-card"><span className="metric-label">Filtered Force</span><strong>{formatNumber(gripperRuntime.filtered_force, 2)}</strong></div>
          <div className="metric-card"><span className="metric-label">Raw Force</span><strong>{formatNumber(gripperRuntime.measured_force_raw, 2)}</strong></div>
          <div className="metric-card"><span className="metric-label">Tare Force</span><strong>{formatNumber(gripperRuntime.tare_force, 2)}</strong></div>
          <div className="metric-card"><span className="metric-label">Status</span><strong>{gripperRuntime.status_text || "--"}</strong></div>
          <div className="metric-card"><span className="metric-label">Updated</span><strong>{formatTimestamp(gripperRuntime.updated_at)}</strong></div>
          <div className="metric-card"><span className="metric-label">Endpoint</span><strong>{gripperRuntime.endpoint || "--"}</strong></div>
        </div>

        {gripperRuntime.last_response ? (
          <div className="muted-block"><div className="record-row"><span>Last Reply</span><strong>{gripperRuntime.last_response}</strong></div></div>
        ) : null}

        <div className="robot-form-grid">
          <label className="field-group"><span className="field-label">Target Force</span><input className="text-input" min={0} step={0.5} type="number" value={forceTarget} onChange={(event) => setForceTarget(normalizeFloat(Number(event.currentTarget.value), forceTarget))} /></label>
          <label className="field-group"><span className="field-label">Raw Position</span><input className="text-input" min={0} step={1} type="number" value={rawPosition} onChange={(event) => setRawPosition(normalizeInteger(Number(event.currentTarget.value), rawPosition, 0))} /></label>
        </div>

        <div className="button-row">
          <button className="ghost-button" disabled={tuningBusy} onClick={() => void props.onRunGripperTuningAction("tare")} type="button">Tare</button>
          <button className="ghost-button" disabled={tuningBusy} onClick={() => void props.onRunGripperTuningAction("sync")} type="button">Sync Runtime</button>
          <button className="ghost-button" disabled={tuningBusy} onClick={() => void props.onRunGripperTuningAction("hold")} type="button">Contact Hold Test</button>
          <button className="ghost-button" disabled={tuningBusy} onClick={() => void props.onRunGripperTuningAction("stop")} type="button">Stop</button>
          <button className="ghost-button" disabled={tuningBusy} onClick={() => void props.onRunGripperTuningAction("hybrid_hold", { target_force: forceTarget })} type="button">Start Hybrid Hold</button>
          <button className="ghost-button" disabled={tuningBusy} onClick={() => void props.onRunGripperTuningAction("set_position", { raw_pos: rawPosition, move_time_ms: tuningDraft.move_time_ms })} type="button">Set Raw Position</button>
        </div>

        <div className="inline-note">Tune direction and limits first. Use Contact Hold Test to verify first contact, then switch to Hybrid Hold for real grasping.</div>
      </Panel>

      <Panel
        title="Force Trace"
        subtitle="Filtered force is plotted over the last 15 seconds. The dashed reference line is the current target force."
        actions={(
          <div className="button-row compact">
            <button
              className="ghost-button"
              type="button"
              onClick={() => {
                lastForceTraceStampRef.current = Number(gripperRuntime.updated_at || 0);
                setForceTrace([]);
              }}
            >
              Clear Trace
            </button>
          </div>
        )}
      >
        <div className="metric-grid compact force-trace-metrics">
          <div className="metric-card"><span className="metric-label">Current Force</span><strong>{formatNumber(gripperRuntime.filtered_force, 2)}</strong></div>
          <div className="metric-card"><span className="metric-label">Target Force</span><strong>{formatNumber(gripperRuntime.target_force, 2)}</strong></div>
          <div className="metric-card"><span className="metric-label">Trace Window</span><strong>{FORCE_TRACE_WINDOW_SEC}s</strong></div>
          <div className="metric-card"><span className="metric-label">Mode / Status</span><strong>{`${gripperRuntime.mode || "--"} / ${gripperRuntime.status_text || "--"}`}</strong></div>
        </div>

        <div className="force-trace-legend">
          <span><i className="force-trace-swatch filtered" />Filtered Force</span>
          <span><i className="force-trace-swatch raw" />Raw Force</span>
          <span><i className="force-trace-swatch target" />Target Force</span>
        </div>

        {forceTraceWindow.samples.length === 0 ? (
          <div className="empty-state">
            <div className="empty-message">No force samples yet. Start Contact Hold Test or Hybrid Hold and the trace will populate automatically.</div>
          </div>
        ) : (
          <div className="force-trace-shell">
            <svg className="force-trace-svg" viewBox={`0 0 ${FORCE_TRACE_WIDTH} ${FORCE_TRACE_HEIGHT}`} role="img" aria-label="Gripper force trace">
              {forceTraceWindow.ticks.map((tick) => (
                <g key={`y-${tick.label}`}>
                  <line
                    className="force-trace-gridline"
                    x1={FORCE_TRACE_PADDING.left}
                    x2={FORCE_TRACE_WIDTH - FORCE_TRACE_PADDING.right}
                    y1={tick.y}
                    y2={tick.y}
                  />
                  <text className="force-trace-axis-label" x={FORCE_TRACE_PADDING.left - 10} y={tick.y + 4} textAnchor="end">{tick.label}</text>
                </g>
              ))}

              {forceTraceWindow.timeTicks.map((tick) => (
                <g key={`x-${tick.label}`}>
                  <line
                    className="force-trace-gridline vertical"
                    x1={tick.x}
                    x2={tick.x}
                    y1={FORCE_TRACE_PADDING.top}
                    y2={FORCE_TRACE_HEIGHT - FORCE_TRACE_PADDING.bottom}
                  />
                  <text className="force-trace-axis-label" x={tick.x} y={FORCE_TRACE_HEIGHT - 10} textAnchor="middle">{tick.label}</text>
                </g>
              ))}

              <line
                className="force-trace-target-line"
                x1={FORCE_TRACE_PADDING.left}
                x2={FORCE_TRACE_WIDTH - FORCE_TRACE_PADDING.right}
                y1={forceTraceWindow.targetY}
                y2={forceTraceWindow.targetY}
              />

              {forceTraceWindow.rawPoints ? (
                <polyline className="force-trace-line raw" points={forceTraceWindow.rawPoints} />
              ) : null}
              {forceTraceWindow.filteredPoints ? (
                <polyline className="force-trace-line filtered" points={forceTraceWindow.filteredPoints} />
              ) : null}
            </svg>
          </div>
        )}

        <div className="inline-note">Y-axis uses tactile raw force counts. The chart is intended for live tuning and demo playback, not calibrated Newtons.</div>
      </Panel>
    </>
  );

  const renderStrategySection = () => (
    <>
      <Panel title="Strategy" subtitle="The current page keeps strategy selection in its own workspace so it does not crowd manual control and tuning.">
        <label className="field-group">
          <span className="field-label">Execution Strategy</span>
          <select className="select-input" disabled={busy || strategyOptions.length <= 1} value={selectedStrategy} onChange={(event) => void props.onSelectStrategy(event.currentTarget.value)}>
            {strategyOptions.map((item) => (
              <option key={item.id} value={item.id}>{item.label}</option>
            ))}
          </select>
        </label>
        <div className="muted-block">
          <div className="record-row"><span>Selected</span><strong>{selectedStrategyMeta.label}</strong></div>
          <div className="record-row"><span>Notes</span><strong>{selectedStrategyMeta.description || "No description yet."}</strong></div>
        </div>

        <label className="field-group">
          <span className="field-label">Gripper Planner</span>
          <select className="select-input" disabled={busy || plannerOptions.length <= 1} value={selectedPlannerStrategy} onChange={(event) => void props.onSelectPlannerStrategy(event.currentTarget.value)}>
            {plannerOptions.map((item) => (
              <option key={item.id} value={item.id}>{item.label}</option>
            ))}
          </select>
        </label>
        <div className="muted-block">
          <div className="record-row"><span>Selected Planner</span><strong>{selectedPlannerMeta.label}</strong></div>
          <div className="record-row"><span>Planner Notes</span><strong>{selectedPlannerMeta.description || "No description yet."}</strong></div>
        </div>
      </Panel>

      <Panel title="Planner Output" subtitle="Latest gripper plan published from the ROS2 planner node. This is the effective profile that the runtime will consume.">
        <div className="record-grid">
          <div className="record-row"><span>Profile</span><strong>{String(gripperProfile.profile_id || "--")}</strong></div>
          <div className="record-row"><span>Planner</span><strong>{String(gripperProfile.planner_strategy || selectedPlannerStrategy || "--")}</strong></div>
          <div className="record-row"><span>Planner Source</span><strong>{String(gripperProfile.planner_source || "--")}</strong></div>
          <div className="record-row"><span>Planner Confidence</span><strong>{formatNumber(plannerConfidence, 3)}</strong></div>
          <div className="record-row"><span>Retrieval Applied</span><strong>{retrievalApplied ? "true" : "false"}</strong></div>
          <div className="record-row"><span>Neighbor Count</span><strong>{Number.isFinite(retrievalNeighborCount) ? retrievalNeighborCount : 0}</strong></div>
          <div className="record-row"><span>Promote Rollup</span><strong>{gripperProfile.promote_rollup ? "true" : "false"}</strong></div>
          <div className="record-row"><span>Grasp Strategy</span><strong>{String(gripperProfile.preferred_strategy || "--")}</strong></div>
          <div className="record-row"><span>Grasp Family</span><strong>{String(gripperProfile.preferred_grasp_family || "--")}</strong></div>
          <div className="record-row"><span>Target Force</span><strong>{formatNumber(Number(gripperProfile.target_force || 0), 2)}</strong></div>
          <div className="record-row"><span>Force Range</span><strong>{targetForceRange.length === 2 ? `${formatNumber(Number(targetForceRange[0] || 0), 2)} - ${formatNumber(Number(targetForceRange[1] || 0), 2)}` : "--"}</strong></div>
          <div className="record-row"><span>Rollup Applied</span><strong>{gripperProfile.rollup_applied ? "true" : "false"}</strong></div>
          <div className="record-row"><span>Rollup Success</span><strong>{`${Number(gripperProfile.rollup_success_count || 0)} @ ${formatNumber(Number(gripperProfile.rollup_success_rate || 0), 3)}`}</strong></div>
        </div>
      </Panel>

      <Panel title="Arm State" subtitle="Latest feedback from /arm/state stays visible here for comparison against manual commands and tuning actions.">
        <div className="record-grid">
          <div className="record-row"><span>Connected</span><strong>{armConnected ? "true" : "false"}</strong></div>
          <div className="record-row"><span>Moving</span><strong>{armMoving ? "true" : "false"}</strong></div>
          <div className="record-row"><span>Error</span><strong>{armError ? "true" : "false"}</strong></div>
          <div className="record-row"><span>Joint Angles</span><strong>{(armState.joint_angles ?? []).map((value) => formatNumber(value, 2)).join(", ") || "--"}</strong></div>
          <div className="record-row"><span>Joint Positions</span><strong>{(armState.joint_positions ?? []).map((value) => formatNumber(value, 0)).join(", ") || "--"}</strong></div>
          <div className="record-row"><span>Updated</span><strong>{formatTimestamp(armState.updated_at)}</strong></div>
        </div>
      </Panel>
    </>
  );

  return (
    <div className="page-grid robot-shell">
      <Panel
        title="Robot Workspace"
        subtitle="Robot stays as a top-level tab, but now uses an internal directory so the page does not keep flattening more panels into one screen."
        actions={(
          <div className="button-row compact">
            <button className="ghost-button" disabled={busy || !robot.enable_ready} onClick={() => void props.onEnableArm(true)} type="button">Enable</button>
            <button className="ghost-button" disabled={returnHomeDisabled} onClick={() => void props.onReturnHome()} type="button">Return Home</button>
          </div>
        )}
      >
        <div className="status-row">
          <StatusPill tone={robot.control_ready ? "success" : "warn"}>Arm Control {robot.control_ready ? "ready" : "pending"}</StatusPill>
          <StatusPill tone={armConnected ? "success" : "warn"}>Arm Link {armConnected ? "up" : "down"}</StatusPill>
          <StatusPill tone={gripperRuntime.runtime_ready ? "success" : gripperRuntime.connection_ready ? "warn" : "neutral"}>
            Gripper Runtime {gripperRuntime.runtime_ready ? "ready" : gripperRuntime.connection_ready ? "degraded" : "down"}
          </StatusPill>
          <StatusPill tone={selectedStrategy ? "info" : "neutral"}>Strategy {selectedStrategy || "--"}</StatusPill>
          <StatusPill tone={selectedPlannerStrategy ? "info" : "neutral"}>Planner {selectedPlannerStrategy || "--"}</StatusPill>
        </div>

        <div className="metric-grid">
          <div className="metric-card"><span className="metric-label">Current Section</span><strong>{ROBOT_SECTIONS.find((item) => item.id === section)?.label}</strong></div>
          <div className="metric-card"><span className="metric-label">Joint Count</span><strong>{robot.joint_count || jointIds.length}</strong></div>
          <div className="metric-card"><span className="metric-label">Gripper Joint</span><strong>J{gripperJointId}</strong></div>
          <div className="metric-card"><span className="metric-label">Arm Updated</span><strong>{formatTimestamp(armState.updated_at)}</strong></div>
        </div>
      </Panel>

      <div className="robot-workspace">
        <Panel title="Sections" subtitle="Switch between Manual, Gripper, and Strategy without loading a new top-level route." className="robot-sidebar-panel">
          <div className="robot-section-nav">
            {ROBOT_SECTIONS.map((item) => (
              <button key={item.id} className={section === item.id ? "robot-section-button active" : "robot-section-button"} onClick={() => setSection(item.id)} type="button">
                <strong>{item.label}</strong>
                <span>{ROBOT_SECTION_COPY[item.id]}</span>
              </button>
            ))}
          </div>
        </Panel>

        <div className="robot-detail-column">
          {section === "manual" ? renderManualSection() : null}
          {section === "gripper" ? renderGripperSection() : null}
          {section === "strategy" ? renderStrategySection() : null}
        </div>
      </div>
    </div>
  );
}
