import { useEffect, useState } from "react";
import { getDailyProgress } from "../api";
import { getAccessToken } from "../auth";
import { AppShell } from "../components/AppShell";
import { getDailyProgressSnapshot } from "../utils/dailyProgress";

type PlannerSnapshot = {
  solved: number;
  created: number;
  coached: number;
  target: number;
};

export function PlannerPage() {
  const [snapshot, setSnapshot] = useState<PlannerSnapshot>({
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

  const agenda = [
    {
      time: "09:00",
      title: "Warm-up practice",
      desc: "Solve 1 foundational problem to start your momentum."
    },
    {
      time: "14:00",
      title: "Deep focus session",
      desc: "Run one full attempt with AI hints kept at minimal level."
    },
    {
      time: "20:00",
      title: "Review and reflection",
      desc: "Open summary timeline and note one recurring mistake."
    }
  ];

  return (
    <AppShell title="Study Planner" subtitle="Plan smart blocks and keep consistency every day.">
      <div className="planner-grid">
        <section className="panel-card planner-summary-card reveal reveal-1">
          <h3>Today&apos;s Targets</h3>
          <div className="planner-chip-row">
            <span>Sessions: {snapshot.solved}/{snapshot.target}</span>
            <span>Courses: {snapshot.created}/1</span>
            <span>Coached: {snapshot.coached}/1</span>
          </div>
          <p>
            Keep your work blocks short and deliberate. High quality repetition beats long unfocused sessions.
          </p>
        </section>

        <section className="panel-card planner-agenda reveal reveal-2">
          <h3>Daily Agenda</h3>
          <ul className="agenda-list">
            {agenda.map((item) => (
              <li key={item.time}>
                <span>{item.time}</span>
                <div>
                  <strong>{item.title}</strong>
                  <p>{item.desc}</p>
                </div>
              </li>
            ))}
          </ul>
        </section>
      </div>
    </AppShell>
  );
}
