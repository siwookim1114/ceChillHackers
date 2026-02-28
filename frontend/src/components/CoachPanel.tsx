import { useEffect, useRef, useState } from "react";
import type { Intervention, StuckSignals } from "../types";

type CoachPanelProps = {
  signals: StuckSignals;
  intervention: Intervention | null;
  onRequestHint: () => void;
};

function levelFromScore(score: number): number {
  if (score >= 70) return 3;
  if (score >= 45) return 2;
  if (score >= 25) return 1;
  return 0;
}

function scoreColor(score: number): string {
  if (score >= 70) return "#c5350a";
  if (score >= 45) return "var(--accent)";
  if (score >= 25) return "#d4a017";
  return "var(--success)";
}

export function CoachPanel({ signals, intervention, onRequestHint }: CoachPanelProps) {
  const currentLevel = intervention?.level ?? levelFromScore(signals.stuck_score);
  const isStuck = currentLevel > 0;
  const [flash, setFlash] = useState(false);
  const prevKeyRef = useRef<string | null>(null);

  useEffect(() => {
    const key = intervention?.created_at ?? null;
    if (key && key !== prevKeyRef.current) {
      prevKeyRef.current = key;
      setFlash(true);
      const t = setTimeout(() => setFlash(false), 1800);
      return () => clearTimeout(t);
    }
  }, [intervention]);

  return (
    <aside className="coach-panel">
      <div className="panel-title-row">
        <h3>Coach</h3>
        <span className={`badge level-${currentLevel}`}>Level {currentLevel}</span>
      </div>

      <div className="status-row">
        <span className={`dot ${isStuck ? "warn" : "ok"}`} />
        <p>{isStuck ? "Stuck detected" : "Flowing well"}</p>
      </div>

      <div className="stuck-section">
        <div className="stuck-label-row">
          <span>Stuck Score</span>
          <strong style={{ color: scoreColor(signals.stuck_score) }}>
            {signals.stuck_score}
          </strong>
        </div>
        <div className="progress-track">
          <div
            className="progress-fill"
            style={{
              width: `${signals.stuck_score}%`,
              background: scoreColor(signals.stuck_score),
            }}
          />
        </div>
      </div>

      <div className="signal-grid">
        <div className="signal-item">
          <small>Idle</small>
          <strong>{Math.floor(signals.idle_ms / 1000)}s</strong>
        </div>
        <div className="signal-item">
          <small>Erases</small>
          <strong>{signals.erase_count_delta}</strong>
        </div>
        <div className="signal-item">
          <small>Retries</small>
          <strong>{signals.repeated_error_count}</strong>
        </div>
        <div className="signal-item">
          <small>Hint Lvl</small>
          <strong>{currentLevel === 0 ? "—" : currentLevel}</strong>
        </div>
      </div>

      <div className={`hint-box${flash ? " flash" : ""}`}>
        {intervention ? (
          <>
            <div className="hint-header">
              <h4>Coach says</h4>
              <span className={`badge level-${intervention.level}`}>Hint {intervention.level}</span>
            </div>
            <p className="hint-reason">{intervention.reason}</p>
            <p className="hint-message">{intervention.tutor_message}</p>
          </>
        ) : (
          <p className="hint-empty">Keep writing your work to receive adaptive hints.</p>
        )}
      </div>

      <button className="btn-teal" onClick={onRequestHint} type="button">
        Ask for a Hint
      </button>
    </aside>
  );
}
