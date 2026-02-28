import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { getSummary } from "../api";
import type { Summary } from "../types";

function formatSeconds(value: number | null): string {
  if (value === null) {
    return "-";
  }
  const minutes = Math.floor(value / 60);
  const seconds = value % 60;
  return `${minutes}m ${seconds}s`;
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

  return (
    <main className="page">
      <section className="card">
        <h2>Result Summary</h2>
        <p className="muted">Stuck -> Intervention -> Resolution timeline</p>

        {loading && <p>Loading summary...</p>}
        {error && <p className="error">{error}</p>}

        {summary && (
          <>
            <div className="metric-grid result-grid">
              <article>
                <small>time-to-solve</small>
                <strong>{formatSeconds(summary.metrics.time_to_solve_sec)}</strong>
              </article>
              <article>
                <small>max stuck</small>
                <strong>{summary.metrics.max_stuck}</strong>
              </article>
              <article>
                <small>hint max level</small>
                <strong>{summary.metrics.hint_max_level}</strong>
              </article>
              <article>
                <small>erase count</small>
                <strong>{summary.metrics.erase_count}</strong>
              </article>
            </div>

            <h3>Timeline</h3>
            <ul className="timeline">
              {summary.timeline.map((entry) => (
                <li key={`${entry.type}-${entry.at}`}>
                  <strong>{entry.label}</strong>
                  <span>{new Date(entry.at).toLocaleString()}</span>
                </li>
              ))}
            </ul>
          </>
        )}

        <div className="action-row">
          <button className="btn-primary" onClick={() => navigate("/home")}>
            Start Another Practice
          </button>
        </div>
      </section>
    </main>
  );
}
