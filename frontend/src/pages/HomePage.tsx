import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { createAttempt, listProblems } from "../api";
import type { Problem } from "../types";

export function HomePage() {
  const navigate = useNavigate();
  const [problems, setProblems] = useState<Problem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [launchingId, setLaunchingId] = useState<string | null>(null);

  const level = localStorage.getItem("preferred_level") ?? "Not set";
  const style = localStorage.getItem("preferred_style") ?? "Not set";
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
      const attempt = await createAttempt({
        guest_id: guestId,
        problem_id: problemId
      });
      navigate(`/solve/${attempt.attempt_id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start attempt");
    } finally {
      setLaunchingId(null);
    }
  };

  return (
    <main className="page">
      <section className="card">
        <h2>Start Practice</h2>
        <p className="muted">
          Level: <strong>{level}</strong> | Style: <strong>{style}</strong>
        </p>

        {loading && <p>Loading problems...</p>}
        {error && <p className="error">{error}</p>}

        <div className="problem-list">
          {problems.map((problem) => (
            <article className="problem-card" key={problem.id}>
              <small>{problem.unit}</small>
              <h3>{problem.title}</h3>
              <p>{problem.prompt}</p>
              <button
                className="btn-primary"
                disabled={launchingId === problem.id}
                onClick={() => startPractice(problem.id)}
              >
                {launchingId === problem.id ? "Starting..." : "Start Practice"}
              </button>
            </article>
          ))}
        </div>
      </section>
    </main>
  );
}
