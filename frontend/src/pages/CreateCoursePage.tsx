import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { createAttempt, getMe, postDailyProgressEvent } from "../api";
import { clearAuthSession, getAccessToken, getAuthUser, saveAuthSession } from "../auth";
import { AppShell } from "../components/AppShell";
import type { AuthUser } from "../types";
import { markCourseCreated } from "../utils/dailyProgress";

const TOPIC_SUGGESTIONS = [
  "Differentiation Basics",
  "Quadratic Equations",
  "Financial Literacy",
  "Creative Writing",
  "Public Speaking"
];

export function CreateCoursePage() {
  const navigate = useNavigate();
  const [user, setUser] = useState<AuthUser | null>(() => getAuthUser());
  const [authLoading, setAuthLoading] = useState(true);
  const [creatingCourse, setCreatingCourse] = useState(false);
  const [courseTopic, setCourseTopic] = useState("Custom Course");
  const [coursePrompt, setCoursePrompt] = useState("");
  const [courseAnswer, setCourseAnswer] = useState("");
  const [error, setError] = useState<string | null>(null);

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
      <AppShell title="Create New Course" subtitle="Checking your session...">
        <section className="panel-card session-skeleton" aria-label="Loading create course workspace">
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
      title="Create New Course"
      subtitle="Build your own guided topic and launch an AI-coached attempt instantly."
    >
      <div className="create-course-layout">
        <section className="practice-head-banner reveal reveal-1">
          <div>
            <p className="overline">Custom Builder</p>
            <h3>Design a course in one minute</h3>
            <p>Set your own topic, prompt, and expected answer key for your demo flow.</p>
          </div>
        </section>

        {error && <p className="error">{error}</p>}

        <section className="panel-card course-builder reveal reveal-2" id="create-course">
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
        </section>
      </div>
    </AppShell>
  );
}
