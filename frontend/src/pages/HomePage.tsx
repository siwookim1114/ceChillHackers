import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ApiError, getDailyProgress, getMe, listProblems } from "../api";
import {
  clearAuthSession,
  getAccessToken,
  getAuthUser,
  saveAuthSession,
} from "../auth";
import { AppShell } from "../components/AppShell";
import type { AuthUser } from "../types";
import { getDailyProgressSnapshot } from "../utils/dailyProgress";

type DashboardProgressState = {
  solvedSessions: number;
  createdCourses: number;
  coachedSessions: number;
  dailyTargetSessions: number;
  currentCourseTopic: string | null;
};

const DEFAULT_PROGRESS_STATE: DashboardProgressState = {
  solvedSessions: 0,
  createdCourses: 0,
  coachedSessions: 0,
  dailyTargetSessions: 2,
  currentCourseTopic: null,
};

function readLocalProgress(): DashboardProgressState {
  const snapshot = getDailyProgressSnapshot();
  return {
    solvedSessions: snapshot.solvedSessions,
    createdCourses: snapshot.createdCourses,
    coachedSessions: snapshot.coachedSessions,
    dailyTargetSessions: 2,
    currentCourseTopic: localStorage.getItem("current_course_topic"),
  };
}

function toProgressState(payload: {
  solved_sessions: number;
  created_courses: number;
  coached_sessions: number;
  daily_target_sessions: number;
  current_course_topic: string | null;
}): DashboardProgressState {
  return {
    solvedSessions: payload.solved_sessions,
    createdCourses: payload.created_courses,
    coachedSessions: payload.coached_sessions,
    dailyTargetSessions: payload.daily_target_sessions,
    currentCourseTopic: payload.current_course_topic,
  };
}

export function HomePage() {
  const navigate = useNavigate();
  const [user, setUser] = useState<AuthUser | null>(() => getAuthUser());
  const [authLoading, setAuthLoading] = useState(true);
  const [problemCount, setProblemCount] = useState(0);
  const [dailyProgress, setDailyProgress] = useState<DashboardProgressState>(
    DEFAULT_PROGRESS_STATE,
  );

  const level = localStorage.getItem("preferred_level") ?? "Beginner";
  const style = localStorage.getItem("preferred_style") ?? "Socratic";

  useEffect(() => {
    const token = getAccessToken();
    if (!token) {
      setAuthLoading(false);
      return;
    }

    getMe()
      .then((me) => {
        saveAuthSession(token, me);
        setUser(me);
      })
      .catch((err) => {
        // Only log out on 401 (token invalid/expired). Server errors (5xx) should
        // not clear a valid session — the user would be stuck in a login loop.
        if (err instanceof ApiError && err.status === 401) {
          clearAuthSession();
          navigate("/login", { replace: true });
        }
      })
      .finally(() => setAuthLoading(false));
  }, [navigate]);

  useEffect(() => {
    listProblems()
      .then((items) => setProblemCount(items.length))
      .catch(() => setProblemCount(0));
  }, []);

  useEffect(() => {
    if (authLoading) {
      return;
    }
    if (getAccessToken() && user) {
      getDailyProgress()
        .then((progress) => setDailyProgress(toProgressState(progress)))
        .catch(() => setDailyProgress(readLocalProgress()));
      return;
    }
    setDailyProgress(readLocalProgress());
  }, [authLoading, user]);

  const solvedSessions = dailyProgress.solvedSessions;
  const createdCourses = dailyProgress.createdCourses;
  const coachedSessions = dailyProgress.coachedSessions;
  const currentTrack = dailyProgress.currentCourseTopic ?? `${level} Track`;
  const dailyTarget = Math.max(1, dailyProgress.dailyTargetSessions || 2);
  const trackGoal = Math.max(3, Math.min(6, problemCount || 3));
  const trackProgress = Math.min(trackGoal, solvedSessions);
  const trackPercent = Math.round((trackProgress / trackGoal) * 100);
  const missionDoneCount =
    Number(solvedSessions >= dailyTarget) +
    Number(createdCourses >= 1) +
    Number(coachedSessions >= 1);
  const missionPercent = Math.round((missionDoneCount / 3) * 100);

  const weeklyBars = [35, 62, 48, 71, 58, 84, Math.max(22, trackPercent)];

  if (authLoading) {
    return (
      <AppShell title="Dashboard" subtitle="Checking your session...">
        <section
          className="panel-card session-skeleton"
          aria-label="Loading dashboard"
        >
          <div className="skeleton-line skeleton-line-short" />
          <div className="skeleton-line skeleton-line-medium" />
          <div className="skeleton-pill-row">
            <span className="skeleton-pill" />
            <span className="skeleton-pill" />
          </div>
          <div className="skeleton-grid-4">
            <span className="skeleton-block" />
            <span className="skeleton-block" />
            <span className="skeleton-block" />
            <span className="skeleton-block" />
          </div>
        </section>
      </AppShell>
    );
  }

  return (
    <AppShell
      title={
        <span className="welcome-title">
          <span>Welcome back, {user?.display_name ?? "Learner"}</span>
          <span className="welcome-wave" aria-hidden="true">
            👋
          </span>
        </span>
      }
      subtitle="Today is a great day to keep your streak alive."
    >
      <div className="dashboard-clean">
        <section className="dashboard-hero reveal reveal-1">
          <div className="dashboard-hero-copy">
            <p className="overline">Today&apos;s Focus</p>
            <h3>{currentTrack}</h3>
            <p>
              You are at <strong>{trackPercent}%</strong> of today&apos;s track.
              Keep the flow with one focused session.
            </p>
            <div className="hero-cta-row">
              <button
                className="btn-primary"
                onClick={() => navigate("/practice")}
                type="button"
              >
                Open Practice Studio
              </button>
              <button
                className="btn-muted"
                onClick={() => navigate("/planner")}
                type="button"
              >
                View Study Planner
              </button>
            </div>
          </div>
          <div className="dashboard-hero-art" aria-hidden="true">
            <div className="hero-orb hero-orb-a" />
            <div className="hero-orb hero-orb-b" />
            <div className="hero-orb hero-orb-c" />
            <div className="hero-float-card card-1">
              <strong>{Math.max(1, dailyTarget - solvedSessions)}</strong>
              <span>sessions left today</span>
            </div>
            <div className="hero-float-card card-2">
              <strong>{missionPercent}%</strong>
              <span>daily mission done</span>
            </div>
          </div>
        </section>

        <section className="dashboard-stat-grid reveal reveal-2">
          <article className="panel-card dashboard-stat-card">
            <small>Practice Solved</small>
            <strong>{solvedSessions}</strong>
            <p>Completed today</p>
          </article>
          <article className="panel-card dashboard-stat-card">
            <small>Courses Created</small>
            <strong>{createdCourses}</strong>
            <p>Custom tracks launched</p>
          </article>
          <article className="panel-card dashboard-stat-card">
            <small>Coach Sessions</small>
            <strong>{coachedSessions}</strong>
            <p>AI-guided attempts</p>
          </article>
          <article className="panel-card dashboard-stat-card">
            <small>Available Problems</small>
            <strong>{problemCount}</strong>
            <p>Ready right now</p>
          </article>
        </section>

        <section className="dashboard-main-grid reveal reveal-3">
          <article className="panel-card journey-card">
            <div className="journey-head">
              <h4>Track Progress</h4>
              <span>{trackPercent}%</span>
            </div>
            <div className="progress-bar-track">
              <div
                className="progress-bar-fill"
                style={{ width: `${trackPercent}%` }}
              />
            </div>
            <p className="journey-copy">
              {trackProgress}/{trackGoal} sessions completed on today&apos;s
              route.
            </p>
            <div className="mini-graph">
              {weeklyBars.map((value, index) => (
                <span
                  className="mini-bar"
                  key={`bar-${index}`}
                  style={{ height: `${Math.max(16, value)}%` }}
                />
              ))}
            </div>
          </article>

          <article className="panel-card missions-card">
            <div className="daily-task-head">
              <h3>Daily Missions</h3>
              <span className="daily-task-percent">{missionPercent}% done</span>
            </div>
            <ul className="daily-task-list">
              <li className={solvedSessions >= dailyTarget ? "done" : ""}>
                <span className="task-check" aria-hidden="true" />
                <div>
                  <strong>Finish {dailyTarget} practice sessions</strong>
                  <small>
                    {solvedSessions}/{dailyTarget}
                  </small>
                </div>
              </li>
              <li className={createdCourses >= 1 ? "done" : ""}>
                <span className="task-check" aria-hidden="true" />
                <div>
                  <strong>Create 1 custom course</strong>
                  <small>{createdCourses}/1</small>
                </div>
              </li>
              <li className={coachedSessions >= 1 ? "done" : ""}>
                <span className="task-check" aria-hidden="true" />
                <div>
                  <strong>Use AI coach at least once</strong>
                  <small>{coachedSessions}/1</small>
                </div>
              </li>
            </ul>
          </article>
        </section>

        <section className="dashboard-links-grid reveal reveal-4">
          <button
            className="panel-card dashboard-link-card"
            onClick={() => navigate("/practice")}
            type="button"
          >
            <h4>Practice Studio</h4>
            <p>
              Start problem solving, submit answers, and get hint intervention.
            </p>
          </button>
          <button
            className="panel-card dashboard-link-card"
            onClick={() => navigate("/progress")}
            type="button"
          >
            <h4>Progress Analytics</h4>
            <p>Check momentum, stuck score trend, and hint-level patterns.</p>
          </button>
          <button
            className="panel-card dashboard-link-card"
            onClick={() => navigate("/library")}
            type="button"
          >
            <h4>Learning Library</h4>
            <p>Collect notes, unit guides, and resources by subject.</p>
          </button>
        </section>
      </div>
    </AppShell>
  );
}
