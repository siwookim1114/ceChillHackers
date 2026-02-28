import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { createAttempt, getDailyProgress, getMe, listProblems, postDailyProgressEvent } from "../api";
import { clearAuthSession, getAccessToken, getAuthUser, saveAuthSession } from "../auth";
import { AppShell } from "../components/AppShell";
import type { AuthUser, Problem } from "../types";
import { getDailyProgressSnapshot, markCourseCreated } from "../utils/dailyProgress";

const TOPIC_SUGGESTIONS = [
  "Differentiation Basics",
  "Quadratic Equations",
  "Financial Literacy",
  "Creative Writing",
  "Public Speaking"
];

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
  currentCourseTopic: null
};

function readLocalProgress(): DashboardProgressState {
  const snapshot = getDailyProgressSnapshot();
  return {
    solvedSessions: snapshot.solvedSessions,
    createdCourses: snapshot.createdCourses,
    coachedSessions: snapshot.coachedSessions,
    dailyTargetSessions: 2,
    currentCourseTopic: localStorage.getItem("current_course_topic")
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
    currentCourseTopic: payload.current_course_topic
  };
}

export function HomePage() {
  const navigate = useNavigate();
  const [user, setUser] = useState<AuthUser | null>(() => getAuthUser());
  const [problems, setProblems] = useState<Problem[]>([]);
  const [loading, setLoading] = useState(true);
  const [authLoading, setAuthLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [launchingId, setLaunchingId] = useState<string | null>(null);
  const [creatingCourse, setCreatingCourse] = useState(false);
  const [courseTopic, setCourseTopic] = useState("Custom Course");
  const [coursePrompt, setCoursePrompt] = useState("");
  const [courseAnswer, setCourseAnswer] = useState("");
  const [dailyProgress, setDailyProgress] = useState<DashboardProgressState>(
    DEFAULT_PROGRESS_STATE
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
      .catch(() => {
        clearAuthSession();
        navigate("/login", { replace: true });
      })
      .finally(() => setAuthLoading(false));
  }, [navigate]);

  useEffect(() => {
    listProblems()
      .then(setProblems)
      .catch((err: Error) => setError(err.message))
      .finally(() => setLoading(false));
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

  useEffect(() => {
    const refreshDailyProgress = () => {
      if (getAccessToken() && user) {
        getDailyProgress()
          .then((progress) => setDailyProgress(toProgressState(progress)))
          .catch(() => setDailyProgress(readLocalProgress()));
      } else {
        setDailyProgress(readLocalProgress());
      }
    };

    const onVisibilityChange = () => {
      if (document.visibilityState === "visible") {
        refreshDailyProgress();
      }
    };

    window.addEventListener("focus", refreshDailyProgress);
    document.addEventListener("visibilitychange", onVisibilityChange);

    return () => {
      window.removeEventListener("focus", refreshDailyProgress);
      document.removeEventListener("visibilitychange", onVisibilityChange);
    };
  }, [user]);

  const getActorId = () => {
    if (user) {
      return `user_${user.id}`;
    }
    const existing = localStorage.getItem("guest_id");
    if (existing) {
      return existing;
    }
    const next = `guest_${Math.random().toString(36).slice(2, 10)}`;
    localStorage.setItem("guest_id", next);
    return next;
  };

  const startPractice = async (problemId: string) => {
    setLaunchingId(problemId);
    setError(null);
    try {
      const attempt = await createAttempt({ guest_id: getActorId(), problem_id: problemId });
      const selectedProblem = problems.find((problem) => problem.id === problemId);
      if (selectedProblem) {
        if (getAccessToken() && user) {
          postDailyProgressEvent({
            event_type: "set_current_topic",
            topic: selectedProblem.unit
          })
            .then((progress) => setDailyProgress(toProgressState(progress)))
            .catch(() => {
              // Keep practice flow uninterrupted when progress sync fails.
            });
        } else {
          localStorage.setItem("current_course_topic", selectedProblem.unit);
        }
      }
      navigate(`/solve/${attempt.attempt_id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start attempt");
    } finally {
      setLaunchingId(null);
    }
  };

  const createCustomCourse = async () => {
    if (!coursePrompt.trim()) {
      setError("Please enter a course prompt first.");
      return;
    }
    if (!courseAnswer.trim()) {
      setError("Please add an expected answer key for demo solving.");
      return;
    }

    setCreatingCourse(true);
    setError(null);
    try {
      const attempt = await createAttempt({
        guest_id: getActorId(),
        problem_text: coursePrompt.trim(),
        answer_key: courseAnswer.trim(),
        unit: courseTopic.trim() || "Custom Course"
      });
      const createdTopic = courseTopic.trim() || "Custom Course";
      if (getAccessToken() && user) {
        try {
          const progress = await postDailyProgressEvent({
            event_type: "course_created",
            attempt_id: attempt.attempt_id,
            topic: createdTopic
          });
          setDailyProgress(toProgressState(progress));
        } catch {
          // Keep attempt creation successful even if progress sync fails.
        }
      } else {
        markCourseCreated(attempt.attempt_id);
        setDailyProgress(readLocalProgress());
        localStorage.setItem("current_course_topic", createdTopic);
      }
      navigate(`/solve/${attempt.attempt_id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create course");
    } finally {
      setCreatingCourse(false);
    }
  };

  const applySuggestion = (topic: string) => {
    setCourseTopic(topic);
    setCoursePrompt(`Teach me ${topic}. Give me one guided practice problem.`);
  };

  const profileState = user ? "Synced account" : "Guest mode";
  const problemCountText = loading ? "..." : `${problems.length}`;
  const hintPolicyLabel = style === "Socratic" ? "Question-first hints" : `${style} hints`;
  const solvedSessions = dailyProgress.solvedSessions;
  const createdCourses = dailyProgress.createdCourses;
  const coachedSessions = dailyProgress.coachedSessions;
  const currentTrack =
    dailyProgress.currentCourseTopic ??
    localStorage.getItem("current_course_topic") ??
    `${level} Track`;
  const dailySessionTarget = Math.max(1, dailyProgress.dailyTargetSessions || 2);
  const trackGoal = Math.max(3, Math.min(6, problems.length || 3));
  const trackProgress = Math.min(trackGoal, solvedSessions);
  const trackPercent = Math.round((trackProgress / trackGoal) * 100);

  const dailyTasks = [
    {
      id: "sessions",
      label: `Finish ${dailySessionTarget} practice sessions`,
      status: `${solvedSessions}/${dailySessionTarget}`,
      done: solvedSessions >= dailySessionTarget
    },
    {
      id: "course",
      label: "Create 1 custom course",
      status: `${createdCourses}/1`,
      done: createdCourses >= 1
    },
    {
      id: "coach",
      label: "Use AI coaching at least once",
      status: `${coachedSessions}/1`,
      done: coachedSessions >= 1
    }
  ];
  const completedTaskCount = dailyTasks.filter((task) => task.done).length;
  const dailyCompletionPercent = Math.round((completedTaskCount / dailyTasks.length) * 100);

  if (authLoading) {
    return (
      <AppShell title="Learning Hub" subtitle="Checking your session...">
        <section className="panel-card">
          <div className="loading-state compact">
            <div className="spinner" />
            <span>Verifying session…</span>
          </div>
        </section>
      </AppShell>
    );
  }

  return (
    <AppShell
      title="Learning Hub"
      subtitle={
        user
          ? `Welcome back, ${user.display_name}. Start quick practice or create a new course.`
          : "Start quick practice or create a brand-new course from your own topic."
      }
    >
      <div className="home-dashboard">
        <section className="promo-banner reveal reveal-1">
          <div>
            <p className="overline">Adaptive Session</p>
            <h3>Build a custom course, then solve with real-time AI coaching</h3>
            <p>
              Level: <strong>{level}</strong> · Style: <strong>{style}</strong>
            </p>
          </div>
          <button
            className="btn-primary"
            onClick={() => document.getElementById("create-course")?.scrollIntoView({ behavior: "smooth" })}
            type="button"
          >
            Create New Course
          </button>
        </section>

        <section className="home-kpi-grid reveal reveal-2">
          <article className="home-kpi-card">
            <small>Profile</small>
            <strong>{profileState}</strong>
            <p>{user ? `Welcome, ${user.display_name}.` : "You can switch to account mode anytime."}</p>
          </article>
          <article className="home-kpi-card">
            <small>Available problems</small>
            <strong>{problemCountText}</strong>
            <p>Ready to launch with event tracking.</p>
          </article>
          <article className="home-kpi-card">
            <small>Hint strategy</small>
            <strong>{hintPolicyLabel}</strong>
            <p>Smallest helpful intervention first.</p>
          </article>
        </section>

        {error && <p className="error">{error}</p>}

        <section className="progress-task-grid reveal reveal-3" id="results">
          <article className="panel-card progress-overview-card">
            <div className="progress-overview-head">
              <div>
                <p className="overline">In Progress Course</p>
                <h3>{currentTrack}</h3>
              </div>
              <span className="progress-percent">{trackPercent}%</span>
            </div>
            <p className="progress-helper">
              Keep momentum daily. Complete your target set before starting new units.
            </p>
            <div className="progress-bar-track">
              <div className="progress-bar-fill" style={{ width: `${trackPercent}%` }} />
            </div>
            <div className="progress-meta-row">
              <span>
                {trackProgress}/{trackGoal} sessions completed today
              </span>
              <span>{coachedSessions} coached sessions</span>
            </div>
          </article>

          <article className="panel-card daily-task-card">
            <div className="daily-task-head">
              <h3>Daily Tasks</h3>
              <span className="daily-task-percent">{dailyCompletionPercent}% done</span>
            </div>
            <ul className="daily-task-list">
              {dailyTasks.map((task) => (
                <li key={task.id} className={task.done ? "done" : ""}>
                  <span className="task-check" aria-hidden="true" />
                  <div>
                    <strong>{task.label}</strong>
                    <small>{task.status}</small>
                  </div>
                </li>
              ))}
            </ul>
          </article>
        </section>

        <section className="home-grid reveal reveal-4">
          <div className="panel-card problem-catalog" id="practice">
            <div className="home-header">
              <h3>My Practice Subjects</h3>
              <span className="user-pill">
                <strong>{level}</strong>
                <span>|</span>
                <strong>{style}</strong>
              </span>
            </div>

            {loading && (
              <div className="loading-state compact">
                <div className="spinner" />
                <span>Loading problems…</span>
              </div>
            )}

            <div className="problem-grid">
              {problems.map((problem) => (
                <article className="problem-card" key={problem.id}>
                  <div className="problem-card-top">
                    <span className="unit-tag">{problem.unit}</span>
                  </div>
                  <h4>{problem.title}</h4>
                  <p className="problem-prompt">{problem.prompt}</p>
                  <button
                    className="btn-primary"
                    disabled={launchingId === problem.id}
                    onClick={() => startPractice(problem.id)}
                    type="button"
                  >
                    {launchingId === problem.id ? "Starting..." : "Start Practice"}
                  </button>
                </article>
              ))}
            </div>
          </div>

          <aside className="panel-card course-builder" id="create-course">
            <div className="builder-header">
              <h3>Create New Course</h3>
              <p>Use your own topic and immediately start a guided attempt.</p>
            </div>

            <label>
              Course Topic
              <input
                value={courseTopic}
                onChange={(event) => setCourseTopic(event.target.value)}
                placeholder="Example: Intro to derivatives"
              />
            </label>

            <label>
              Starter Prompt
              <textarea
                value={coursePrompt}
                onChange={(event) => setCoursePrompt(event.target.value)}
                placeholder="Example: Explain derivatives conceptually and ask one practice question."
                rows={5}
              />
            </label>

            <label>
              Expected Answer Key (for demo)
              <input
                value={courseAnswer}
                onChange={(event) => setCourseAnswer(event.target.value)}
                placeholder="Example: 6x+2"
              />
            </label>

            <div className="topic-chips">
              {TOPIC_SUGGESTIONS.map((topic) => (
                <button
                  className="chip-btn"
                  key={topic}
                  onClick={() => applySuggestion(topic)}
                  type="button"
                >
                  {topic}
                </button>
              ))}
            </div>

            <button className="btn-teal" onClick={createCustomCourse} disabled={creatingCourse} type="button">
              {creatingCourse ? "Creating..." : "Create Course & Start"}
            </button>
          </aside>
        </section>

        <section className="panel-card mini-cards reveal reveal-5">
          <article>
            <h4>Study Plan</h4>
            <p>Track sessions by topic, stuck score trends, and hint level usage.</p>
          </article>
          <article>
            <h4>Coach Mode</h4>
            <p>Hints are progressive: question first, concept summary next, fallback mini-task last.</p>
          </article>
          <article>
            <h4>Session Replay</h4>
            <p>Every attempt stores events and timeline for fast demo storytelling.</p>
          </article>
        </section>
      </div>
    </AppShell>
  );
}
