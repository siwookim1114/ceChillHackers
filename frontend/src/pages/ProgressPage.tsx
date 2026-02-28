import { useEffect, useState } from "react";
import { getDailyProgress } from "../api";
import { getAccessToken } from "../auth";
import { AppShell } from "../components/AppShell";
import { getDailyProgressSnapshot } from "../utils/dailyProgress";

type ProgressSnapshot = {
  solved: number;
  created: number;
  coached: number;
  target: number;
};

const TREND = [26, 38, 44, 52, 49, 66, 78];

export function ProgressPage() {
  const [snapshot, setSnapshot] = useState<ProgressSnapshot>({
    solved: 0,
    created: 0,
    coached: 0,
    target: 2
  });

  useEffect(() => {
    if (!getAccessToken()) {
      const local = getDailyProgressSnapshot();
      setSnapshot({
        solved: local.solvedSessions,
        created: local.createdCourses,
        coached: local.coachedSessions,
        target: 2
      });
      return;
    }

    getDailyProgress()
      .then((progress) =>
        setSnapshot({
          solved: progress.solved_sessions,
          created: progress.created_courses,
          coached: progress.coached_sessions,
          target: progress.daily_target_sessions
        })
      )
      .catch(() => {
        const local = getDailyProgressSnapshot();
        setSnapshot({
          solved: local.solvedSessions,
          created: local.createdCourses,
          coached: local.coachedSessions,
          target: 2
        });
      });
  }, []);

  const completion = Math.round((Math.min(snapshot.solved, snapshot.target) / Math.max(1, snapshot.target)) * 100);

  return (
    <AppShell title="Progress" subtitle="Track performance trends and coaching usage.">
      <div className="progress-page-grid">
        <section className="panel-card progress-ring-card reveal reveal-1">
          <h3>Daily Completion</h3>
          <div className="progress-ring-wrap">
            <div className="progress-ring" style={{ ["--progress" as string]: `${completion}%` }}>
              <strong>{completion}%</strong>
              <span>done</span>
            </div>
          </div>
          <p>
            {snapshot.solved}/{snapshot.target} sessions completed today. Keep consistency for better retention.
          </p>
        </section>

        <section className="panel-card trend-card reveal reveal-2">
          <h3>7-Day Momentum</h3>
          <div className="trend-bars">
            {TREND.map((value, idx) => (
              <span key={`trend-${idx}`} style={{ height: `${value}%` }} />
            ))}
          </div>
          <div className="trend-meta">
            <span>Mon</span>
            <span>Tue</span>
            <span>Wed</span>
            <span>Thu</span>
            <span>Fri</span>
            <span>Sat</span>
            <span>Sun</span>
          </div>
        </section>

        <section className="panel-card progress-kpis reveal reveal-3">
          <article>
            <small>Solved Sessions</small>
            <strong>{snapshot.solved}</strong>
          </article>
          <article>
            <small>Coached Sessions</small>
            <strong>{snapshot.coached}</strong>
          </article>
          <article>
            <small>Courses Created</small>
            <strong>{snapshot.created}</strong>
          </article>
        </section>
      </div>
    </AppShell>
  );
}
