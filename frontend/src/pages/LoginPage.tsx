import { FormEvent, useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { login } from "../api";
import { getAccessToken, saveAuthSession } from "../auth";

export function LoginPage() {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (getAccessToken()) {
      navigate("/home", { replace: true });
    }
  }, [navigate]);

  const onSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setSubmitting(true);
    setError(null);

    try {
      const result = await login({
        email: email.trim(),
        password
      });
      saveAuthSession(result.access_token, result.user);
      navigate("/home", { replace: true });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to log in");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <main className="page page-auth">
      <section className="card auth-card">
        <div className="auth-header">
          <p className="overline">Account Access</p>
          <h2>Log In</h2>
          <p className="muted">Continue your sessions, hints, and progress timeline.</p>
        </div>

        <form className="auth-form" onSubmit={onSubmit}>
          <label>
            Email
            <input
              autoComplete="email"
              onChange={(event) => setEmail(event.target.value)}
              placeholder="you@example.com"
              required
              type="email"
              value={email}
            />
          </label>

          <label>
            Password
            <input
              autoComplete="current-password"
              onChange={(event) => setPassword(event.target.value)}
              placeholder="Enter your password"
              required
              type="password"
              value={password}
            />
          </label>

          {error && <p className="error">{error}</p>}

          <button className="btn-primary auth-submit" disabled={submitting} type="submit">
            {submitting ? "Logging in..." : "Log In"}
          </button>
        </form>

        <p className="auth-footnote">
          Don&apos;t have an account? <Link to="/signup">Create one</Link>
        </p>
      </section>
    </main>
  );
}
