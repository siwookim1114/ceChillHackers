import type { Intervention, StuckSignals } from "../types";

type CoachPanelProps = {
  signals: StuckSignals;
  intervention: Intervention | null;
};

function levelFromScore(score: number): number {
  if (score >= 70) {
    return 3;
  }
  if (score >= 45) {
    return 2;
  }
  if (score >= 25) {
    return 1;
  }
  return 0;
}

export function CoachPanel({ signals, intervention }: CoachPanelProps) {
  const currentLevel = intervention?.level ?? levelFromScore(signals.stuck_score);
  const isStuck = currentLevel > 0;

  return (
    <aside className="panel coach-panel">
      <div className="panel-title-row">
        <h3>Coach Panel</h3>
        <span className={`badge level-${currentLevel}`}>Hint Level {currentLevel}</span>
      </div>

      <div className="status-row">
        <span className={`dot ${isStuck ? "warn" : "ok"}`} />
        <p>{isStuck ? "Stuck detected" : "Flowing"}</p>
      </div>

      <p className="muted">
        You paused for {Math.floor(signals.idle_ms / 1000)}s and retried the same step{" "}
        {signals.repeated_error_count} times.
      </p>

      <div className="metric-grid">
        <div>
          <small>Stuck Score</small>
          <strong>{signals.stuck_score}</strong>
        </div>
        <div>
          <small>Erase Delta</small>
          <strong>{signals.erase_count_delta}</strong>
        </div>
      </div>

      <div className="hint-box">
        <h4>Latest Intervention</h4>
        <p>{intervention?.reason ?? "No intervention yet."}</p>
        <p>{intervention?.tutor_message ?? "Keep writing your reasoning to receive adaptive hints."}</p>
      </div>
    </aside>
  );
}
