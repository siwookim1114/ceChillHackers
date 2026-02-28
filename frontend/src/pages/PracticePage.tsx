import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { createAttempt, getMe, listProblems, postDailyProgressEvent } from "../api";
import { clearAuthSession, getAccessToken, getAuthUser, saveAuthSession } from "../auth";
import { AppShell } from "../components/AppShell";
import type { AuthUser, Problem } from "../types";
import { markCourseCreated } from "../utils/dailyProgress";

const TOPIC_SUGGESTIONS = [
  "Differentiation Basics",
  "Quadratic Equations",
  "Financial Literacy",
  "Creative Writing",
  "Public Speaking"
];

export function PracticePage() {
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
        await postDailyProgressEvent({
          event_type: "course_created",
          attempt_id: attempt.attempt_id,
          topic: createdTopic
        }).catch(() => {
          // Keep attempt creation successful even if progress sync fails.
        });
      } else {
        markCourseCreated(attempt.attempt_id);
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

  if (authLoading) {
    return (
      <AppShell title="Practice Studio" subtitle="Checking your session...">
        <section className="panel-card">
          <div className="loading-state compact">
            <div className="spinner" />
            <span>Verifying session...</span>
          </div>
        </section>
      </AppShell>
    );
  }

  return (
    <AppShell
      title="Practice Studio"
      subtitle="Choose a problem or create a custom course with your own prompt."
    >
      <div className="practice-page-wrap">
        <section className="practice-head-banner reveal reveal-1">
          <div>
            <p className="overline">Live Workspace</p>
            <h3>Build and solve with real-time coaching</h3>
            <p>
              Level: <strong>{level}</strong> | Style: <strong>{style}</strong>
            </p>
          </div>
          <button
            className="btn-primary"
            onClick={() => document.getElementById("create-course")?.scrollIntoView({ behavior: "smooth" })}
            type="button"
          >
            Jump to Course Builder
          </button>
        </section>

        {error && <p className="error">{error}</p>}

        <section className="home-grid reveal reveal-2">
          <div className="panel-card problem-catalog">
            <div className="home-header">
              <h3>Problem Catalog</h3>
              <span className="user-pill">
                <strong>{problems.length}</strong>
                <span>ready</span>
              </span>
            </div>

            {loading && (
              <div className="loading-state compact">
                <div className="spinner" />
                <span>Loading problems...</span>
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
                <button className="chip-btn" key={topic} onClick={() => applySuggestion(topic)} type="button">
                  {topic}
                </button>
              ))}
            </div>

            <button className="btn-teal" onClick={createCustomCourse} disabled={creatingCourse} type="button">
              {creatingCourse ? "Creating..." : "Create Course & Start"}
            </button>
          </aside>
        </section>
      </div>
    </AppShell>
  );
}
