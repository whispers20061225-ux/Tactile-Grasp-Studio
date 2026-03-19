import { useMemo, useState } from "react";
import type { OverlayBox, StreamMap } from "./appHelpers";
import { findCandidate, formatNumber, STREAM_OPTIONS } from "./appHelpers";
import { EmptyState, OverlayStage, Panel, StatusPill } from "./appUi";
import type { UiState } from "./types";

type VisionPageProps = {
  state: UiState;
  streams: StreamMap;
  onChooseLabel: (label: string) => void;
};

export function VisionPage(props: VisionPageProps) {
  const [activeStream, setActiveStream] = useState<keyof StreamMap>("detection_overlay");
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const hoveredCandidate = useMemo(
    () => findCandidate(props.state.vision.debug_candidates, hoveredIndex),
    [hoveredIndex, props.state.vision.debug_candidates],
  );

  const overlayBoxes = useMemo<OverlayBox[]>(() => {
    const boxes: OverlayBox[] = [];
    const selected = props.state.vision.selected_candidate;
    if (selected?.bbox_xyxy && selected.bbox_xyxy.length === 4) {
      boxes.push({ bbox: selected.bbox_xyxy, label: `Selected: ${selected.label}`, tone: "selected" });
    } else if (props.state.vision.detection.bbox?.xyxy) {
      boxes.push({
        bbox: props.state.vision.detection.bbox.xyxy,
        label: `Detection: ${props.state.vision.detection.target_label || "target"}`,
        tone: "selected",
      });
    }
    if (hoveredCandidate?.bbox_xyxy && hoveredCandidate.bbox_xyxy.length === 4) {
      const hoveredKey = hoveredCandidate.bbox_xyxy.join("-");
      const existingIndex = boxes.findIndex((item) => item.bbox.join("-") === hoveredKey);
      const hoveredBox: OverlayBox = { bbox: hoveredCandidate.bbox_xyxy, label: `Hover: ${hoveredCandidate.label}`, tone: "hovered" };
      if (existingIndex >= 0) boxes[existingIndex] = hoveredBox;
      else boxes.push(hoveredBox);
    }
    return boxes;
  }, [hoveredCandidate, props.state.vision.detection, props.state.vision.selected_candidate]);

  const stageSrc = activeStream === "detection_overlay" ? props.streams.rgb : props.streams[activeStream];
  const stageBoxes = overlayBoxes;

  return (
    <div className="page-grid vision-grid">
      <Panel
        title="Vision Monitor"
        subtitle="Detection overlay now draws candidate boxes on top of the live RGB stream. Hovering a candidate highlights its bounding box."
        className="panel-tall"
        actions={
          <div className="toggle-row">
            {STREAM_OPTIONS.map((item) => (
              <button
                key={item.key}
                type="button"
                data-testid={`vision-toggle-${item.key}`}
                aria-pressed={activeStream === item.key}
                className={activeStream === item.key ? "toggle-chip active" : "toggle-chip"}
                onClick={() => setActiveStream(item.key)}
              >
                {item.label}
              </button>
            ))}
          </div>
        }
      >
        <OverlayStage
          src={stageSrc}
          alt={STREAM_OPTIONS.find((item) => item.key === activeStream)?.label ?? "Vision Stream"}
          imageWidth={props.state.vision.image_width}
          imageHeight={props.state.vision.image_height}
          boxes={stageBoxes}
          testId="vision-stage"
        />

        <div className="metric-grid">
          <div className="metric-card"><span className="metric-label">Accepted</span><strong>{props.state.vision.detection.accepted ? "yes" : "no"}</strong></div>
          <div className="metric-card"><span className="metric-label">Target Label</span><strong>{props.state.vision.detection.target_label || "--"}</strong></div>
          <div className="metric-card"><span className="metric-label">Confidence</span><strong>{formatNumber(props.state.vision.detection.confidence, 4)}</strong></div>
          <div className="metric-card"><span className="metric-label">Image Size</span><strong>{props.state.vision.image_width > 0 && props.state.vision.image_height > 0 ? `${props.state.vision.image_width} x ${props.state.vision.image_height}` : "--"}</strong></div>
        </div>

        <div className="summary-block">
          <div className="field-label">Candidate Summary</div>
          <div className="summary-text">{props.state.vision.candidate_summary || "Waiting for detection_debug top-k data."}</div>
        </div>
      </Panel>

      <Panel title="Top-K Candidates" subtitle="Clicking a candidate only stages a label-level override. There is no instance pinning in v1.">
        {props.state.vision.debug_candidates.length === 0 ? (
          <EmptyState title="No candidates yet" message="Waiting for detection_debug top-k candidates and bounding boxes." />
        ) : (
          <div className="candidate-list">
            {props.state.vision.debug_candidates.map((candidate) => {
              const isHovered = hoveredIndex === candidate.index;
              const isSelected = props.state.vision.selected_candidate?.index === candidate.index;
              return (
                <button
                  key={candidate.index}
                  type="button"
                  data-testid={`vision-candidate-${candidate.index}`}
                  className={["candidate-item", isHovered ? "hovered" : "", isSelected ? "selected" : ""].filter(Boolean).join(" ")}
                  onMouseEnter={() => setHoveredIndex(candidate.index)}
                  onMouseLeave={() => setHoveredIndex((current) => (current === candidate.index ? null : current))}
                  onClick={() => props.onChooseLabel(candidate.label)}
                >
                  <div className="candidate-header">
                    <strong>{candidate.label}</strong>
                    <StatusPill tone={candidate.status === "selectable" ? "success" : "warn"}>{candidate.status}</StatusPill>
                  </div>
                  <div className="candidate-meta">
                    <span>index {candidate.index}</span>
                    <span>conf {formatNumber(candidate.confidence, 4)}</span>
                    <span>score {formatNumber(candidate.score, 4)}</span>
                    <span>bonus {formatNumber(candidate.semantic_bonus, 3)}</span>
                  </div>
                  <div className="candidate-meta">
                    <span>bbox {candidate.bbox_xyxy ? candidate.bbox_xyxy.join(", ") : "--"}</span>
                    <span>mask {candidate.mask_pixels ?? 0}</span>
                  </div>
                </button>
              );
            })}
          </div>
        )}
      </Panel>
    </div>
  );
}
