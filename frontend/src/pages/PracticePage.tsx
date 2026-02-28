import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { createAttempt, getMe, listProblems, postDailyProgressEvent } from "../api";
import { clearAuthSession, getAccessToken, getAuthUser, saveAuthSession } from "../auth";
import { AppShell } from "../components/AppShell";
import type { AuthUser, Problem } from "../types";

export function PracticePage() {
  const navigate = useNavigate();
  const [user, setUser] = useState<AuthUser | null>(() => getAuthUser());
  const [problems, setProblems] = useState<Problem[]>([]);
  const [loading, setLoading] = useState(true);
  const [authLoading, setAuthLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [launchingId, setLaunchingId] = useState<string | null>(null);

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
          }).catch(() => {
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

  if (authLoading) {
    return (
      <AppShell title="Practice Studio" subtitle="Checking your session...">
        <section className="panel-card session-skeleton" aria-label="Loading practice workspace">
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
    <AppShell title="Practice Studio" subtitle="Choose a problem and start a coached solving session.">
      <div className="practice-page-wrap">
        <section className="practice-head-banner reveal reveal-1">
          <div>
            <p className="overline">Live Workspace</p>
            <h3>Train with focused problem sets</h3>
            <p>
              Level: <strong>{level}</strong> | Style: <strong>{style}</strong>
            </p>
          </div>
        </section>

        {error && <p className="error">{error}</p>}

        <section className="panel-card problem-catalog reveal reveal-2">
          <div className="home-header">
            <h3>Problem Catalog</h3>
            <span className="user-pill">
              <strong>{problems.length}</strong>
              <span>ready</span>
            </span>
          </div>

          {loading && (
            <div className="catalog-skeleton" aria-label="Loading problems">
              <div className="skeleton-line skeleton-line-medium" />
              <div className="skeleton-grid-3">
                <span className="skeleton-block" />
                <span className="skeleton-block" />
                <span className="skeleton-block" />
              </div>
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
        </section>
      </div>
    </AppShell>
  );
}
