import { FormEvent, useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { signup } from "../api";
import { getAccessToken, saveAuthSession } from "../auth";
import { EyeToggleIcon } from "../components/EyeToggleIcon";
import type { LearningPace, LearningStyle } from "../types";

const STYLE_OPTIONS: Array<{ value: LearningStyle; label: string }> = [
  { value: "question", label: "Question-first" },
  { value: "explanation", label: "Explanation-first" },
  { value: "problem_solving", label: "Problem-solving" }
];

const PACE_OPTIONS: Array<{ value: LearningPace; label: string }> = [
  { value: "slow", label: "Slow" },
  { value: "normal", label: "Normal" },
  { value: "fast", label: "Fast" }
];

export function SignupPage() {
  const navigate = useNavigate();
  const [displayName, setDisplayName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [learningStyle, setLearningStyle] = useState<LearningStyle>("question");
  const [learningPace, setLearningPace] = useState<LearningPace>("normal");
  const [targetGoal, setTargetGoal] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (getAccessToken()) {
      navigate("/home", { replace: true });
    }
  }, [navigate]);

  const canSubmit = useMemo(
    () => Boolean(email.trim() && password && confirmPassword && !submitting),
    [confirmPassword, email, password, submitting]
  );

  const onSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError(null);

    if (password.length < 8) {
      setError("Password must be at least 8 characters.");
      return;
    }
    if (password !== confirmPassword) {
      setError("Password and confirm password do not match.");
      return;
    }

    setSubmitting(true);
    try {
      const result = await signup({
        email: email.trim(),
        password,
        display_name: displayName.trim(),
        role: "student",
        learning_style: learningStyle,
        learning_pace: learningPace,
        target_goal: targetGoal.trim() || undefined
      });
      saveAuthSession(result.access_token, result.user);
      navigate("/home", { replace: true });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to sign up");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <main className="auth-plain-root">
      <header className="auth-plain-topbar">
        <div className="auth-plain-container">
          <button className="entry-brand" onClick={() => navigate("/")} type="button">
            <span className="entry-brand-mark">TC</span>
            <strong>TutorCoach</strong>
          </button>

          <nav className="entry-nav-links">
            <a href="/practice">Practice</a>
            <a href="/practice#create-course">Create Course</a>
            <a href="/progress">Session Results</a>
            <a href="/home">AI Coach</a>
          </nav>

          <div className="entry-top-actions">
            <button className="btn-entry-ghost" type="button">
              English, USD
            </button>
            <button className="btn-entry-outline" onClick={() => navigate("/login")} type="button">
              Log In
            </button>
          </div>
        </div>
      </header>

      <section className="auth-plain-body">
        <div className="auth-plain-panel auth-signup-panel">
          <h1>Create account</h1>
          <p className="auth-plain-signup-links">
            Already have an account? <Link to="/login">Log in</Link>
          </p>

          <form className="auth-plain-form" onSubmit={onSubmit}>
            <label>
              Display Name
              <input
                autoComplete="name"
                onChange={(event) => setDisplayName(event.target.value)}
                placeholder="Your name"
                type="text"
                value={displayName}
              />
            </label>

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

            <div className="auth-grid-two">
              <label>
                Password
                <div className="auth-password-row">
                  <input
                    autoComplete="new-password"
                    onChange={(event) => setPassword(event.target.value)}
                    placeholder="At least 8 characters"
                    required
                    type={showPassword ? "text" : "password"}
                    value={password}
                  />
                  <button
                    aria-label={showPassword ? "Hide password" : "Show password"}
                    className="auth-password-toggle"
                    onClick={() => setShowPassword((prev) => !prev)}
                    type="button"
                  >
                    <EyeToggleIcon visible={showPassword} />
                  </button>
                </div>
              </label>
              <label>
                Confirm Password
                <div className="auth-password-row">
                  <input
                    autoComplete="new-password"
                    onChange={(event) => setConfirmPassword(event.target.value)}
                    placeholder="Re-enter password"
                    required
                    type={showConfirmPassword ? "text" : "password"}
                    value={confirmPassword}
                  />
                  <button
                    aria-label={showConfirmPassword ? "Hide password" : "Show password"}
                    className="auth-password-toggle"
                    onClick={() => setShowConfirmPassword((prev) => !prev)}
                    type="button"
                  >
                    <EyeToggleIcon visible={showConfirmPassword} />
                  </button>
                </div>
              </label>
            </div>

            <div className="auth-grid-two">
              <label>
                Learning Style
                <select
                  onChange={(event) => setLearningStyle(event.target.value as LearningStyle)}
                  value={learningStyle}
                >
                  {STYLE_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>

              <label>
                Learning Pace
                <select
                  onChange={(event) => setLearningPace(event.target.value as LearningPace)}
                  value={learningPace}
                >
                  {PACE_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>
            </div>

            <label>
              Target Goal (optional)
              <input
                onChange={(event) => setTargetGoal(event.target.value)}
                placeholder="Example: AP Calculus exam in May"
                type="text"
                value={targetGoal}
              />
            </label>

            <p className="muted auth-note">
              Account type: learner. You can change learning preferences anytime.
            </p>

            {error && <p className="error">{error}</p>}

            <button className="auth-submit-plain" disabled={!canSubmit} type="submit">
              {submitting ? "Creating account..." : "Create account"}
            </button>
          </form>

          <p className="auth-plain-terms">
            By creating an account, you agree to TutorCoach <Link to="/signup">Terms</Link> and{" "}
            <Link to="/signup">Privacy Policy</Link>.
          </p>
        </div>
      </section>
    </main>
  );
}
