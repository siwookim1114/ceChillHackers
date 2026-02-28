import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { getSummary } from "../api";
import { AppShell } from "../components/AppShell";
import type { Summary } from "../types";

function formatSeconds(value: number | null): string {
  if (value === null) {
    return "—";
  }
  const minutes = Math.floor(value / 60);
  const seconds = value % 60;
  return minutes > 0 ? `${minutes}m ${seconds}s` : `${seconds}s`;
}

function tlDotClass(type: string): string {
  if (type === "attempt_start") {
    return "tl-dot start";
  }
  if (type === "intervention") {
    return "tl-dot intervention";
  }
  if (type === "solved") {
    return "tl-dot solved";
  }
  return "tl-dot start";
}

function tlIcon(type: string): string {
  if (type === "attempt_start") {
    return "▶";
  }
  if (type === "intervention") {
    return "💡";
  }
  if (type === "solved") {
    return "✓";
  }
  return "•";
}

export function ResultPage() {
  const { attemptId } = useParams();
  const navigate = useNavigate();
  const [summary, setSummary] = useState<Summary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!attemptId) {
      setError("Missing attempt id");
      setLoading(false);
      return;
    }
    getSummary(attemptId)
      .then(setSummary)
      .catch((err: Error) => setError(err.message))
      .finally(() => setLoading(false));
  }, [attemptId]);

  const didSolve =
    summary?.metrics.time_to_solve_sec !== null && summary?.metrics.time_to_solve_sec !== undefined;
  const wasReallyStuck = (summary?.metrics.max_stuck ?? 0) >= 60;

  return (
    <AppShell title="Session Complete" subtitle="Stuck → Intervention → Resolution timeline">
      {loading && (
        <section className="panel-card">
          <div className="loading-state">
            <div className="spinner" />
            <span>Loading summary…</span>
          </div>
        </section>
      )}

      {error && (
        <section className="panel-card">
          <p className="error">{error}</p>
        </section>
      )}

      {summary && (
        <section className="panel-card result-page-inner">
          {didSolve && wasReallyStuck && (
            <div className="insight-banner hard">
              <span className="insight-icon">🔥</span>
              <span>
                Stuck score hit <strong>{summary.metrics.max_stuck}/100</strong> and you still solved it.
                Great productive struggle.
              </span>
            </div>
          )}
          {didSolve && !wasReallyStuck && (
            <div className="insight-banner">
              <span className="insight-icon">✅</span>
              <span>
                Solved in <strong>{formatSeconds(summary.metrics.time_to_solve_sec)}</strong> with stable
                momentum.
              </span>
            </div>
          )}

          <div className="metrics-row">
            <div className="metric-card">
              <small>Time to Solve</small>
              <strong>{formatSeconds(summary.metrics.time_to_solve_sec)}</strong>
              <span className="metric-sub">total duration</span>
            </div>
            <div className="metric-card">
              <small>Max Stuck</small>
              <strong style={{ color: summary.metrics.max_stuck >= 60 ? "var(--accent)" : "inherit" }}>
                {summary.metrics.max_stuck}
              </strong>
              <span className="metric-sub">/ 100 score</span>
            </div>
            <div className="metric-card">
              <small>Hint Level</small>
              <strong style={{ color: summary.metrics.hint_max_level > 0 ? "var(--accent-2)" : "inherit" }}>
                {summary.metrics.hint_max_level === 0 ? "—" : `L${summary.metrics.hint_max_level}`}
              </strong>
              <span className="metric-sub">max reached</span>
            </div>
            <div className="metric-card">
              <small>Erases</small>
              <strong>{summary.metrics.erase_count}</strong>
              <span className="metric-sub">strokes erased</span>
            </div>
          </div>

          <div className="timeline-section">
            <h3>Session Timeline</h3>
            <ul className="timeline">
              {summary.timeline.map((entry) => (
                <li key={`${entry.type}-${entry.at}`}>
                  <span className={tlDotClass(entry.type)}>{tlIcon(entry.type)}</span>
                  <div className="tl-body">
                    <strong>{entry.label}</strong>
                    <span>{new Date(entry.at).toLocaleTimeString()}</span>
                  </div>
                </li>
              ))}
            </ul>
          </div>

          <div className="result-actions">
            <button className="btn-primary" onClick={() => navigate("/home")} type="button">
              Practice Again →
            </button>
            <button className="btn-muted" onClick={() => navigate("/")} type="button">
              Home
            </button>
          </div>
        </section>
      )}
    </AppShell>
  );
}
