import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { createAttempt, listProblems } from "../api";
import { AppShell } from "../components/AppShell";
import type { Problem } from "../types";

const TOPIC_SUGGESTIONS = [
  "Differentiation Basics",
  "Quadratic Equations",
  "Financial Literacy",
  "Creative Writing",
  "Public Speaking"
];

export function HomePage() {
  const navigate = useNavigate();
  const [problems, setProblems] = useState<Problem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [launchingId, setLaunchingId] = useState<string | null>(null);
  const [creatingCourse, setCreatingCourse] = useState(false);
  const [courseTopic, setCourseTopic] = useState("Custom Course");
  const [coursePrompt, setCoursePrompt] = useState("");
  const [courseAnswer, setCourseAnswer] = useState("");

  const level = localStorage.getItem("preferred_level") ?? "Beginner";
  const style = localStorage.getItem("preferred_style") ?? "Socratic";
  const guestId = localStorage.getItem("guest_id") ?? "guest_unknown";

  useEffect(() => {
    listProblems()
      .then(setProblems)
      .catch((err: Error) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  const startPractice = async (problemId: string) => {
    setLaunchingId(problemId);
    setError(null);
    try {
      const attempt = await createAttempt({ guest_id: guestId, problem_id: problemId });
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
        guest_id: guestId,
        problem_text: coursePrompt.trim(),
        answer_key: courseAnswer.trim(),
        unit: courseTopic.trim() || "Custom Course"
      });
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

  return (
    <AppShell
      title="Learning Hub"
      subtitle="Start quick practice or create a brand-new course from your own topic."
    >
      <div className="home-dashboard">
        <section className="promo-banner">
          <div>
            <p className="overline">Adaptive Session</p>
            <h3>Build a custom course, then solve with live AI coaching</h3>
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

        {error && <p className="error">{error}</p>}

        <section className="home-grid">
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
                    {launchingId === problem.id ? "Starting…" : "Start Practice →"}
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
              {creatingCourse ? "Creating…" : "Create Course & Start →"}
            </button>
          </aside>
        </section>

        <section className="panel-card mini-cards" id="results">
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
