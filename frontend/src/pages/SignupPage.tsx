import { FormEvent, useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { signup } from "../api";
import { getAccessToken, saveAuthSession } from "../auth";
import type { LearningPace, LearningStyle, UserRole } from "../types";

const ROLE_OPTIONS: Array<{ value: UserRole; label: string }> = [
  { value: "student", label: "Student" },
  { value: "teacher", label: "Teacher" },
  { value: "parent", label: "Parent" }
];

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
  const [role, setRole] = useState<UserRole>("student");
  const [learningStyle, setLearningStyle] = useState<LearningStyle>("question");
  const [learningPace, setLearningPace] = useState<LearningPace>("normal");
  const [targetGoal, setTargetGoal] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (getAccessToken()) {
      navigate("/home", { replace: true });
    }
  }, [navigate]);

  const disableStudentFields = role !== "student";
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
        role,
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
    <main className="page page-auth">
      <section className="card auth-card auth-card-wide">
        <div className="auth-header">
          <p className="overline">Get Started</p>
          <h2>Create Account</h2>
          <p className="muted">Set your profile once and use adaptive coaching across sessions.</p>
        </div>

        <form className="auth-form" onSubmit={onSubmit}>
          <div className="auth-grid-two">
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
              Role
              <select onChange={(event) => setRole(event.target.value as UserRole)} value={role}>
                {ROLE_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
          </div>

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
              <input
                autoComplete="new-password"
                onChange={(event) => setPassword(event.target.value)}
                placeholder="At least 8 characters"
                required
                type="password"
                value={password}
              />
            </label>
            <label>
              Confirm Password
              <input
                autoComplete="new-password"
                onChange={(event) => setConfirmPassword(event.target.value)}
                placeholder="Re-enter password"
                required
                type="password"
                value={confirmPassword}
              />
            </label>
          </div>

          <div className="auth-grid-two">
            <label>
              Learning Style
              <select
                disabled={disableStudentFields}
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
                disabled={disableStudentFields}
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
              placeholder="Example: IB Math AA exam in May"
              type="text"
              value={targetGoal}
            />
          </label>

          {disableStudentFields && (
            <p className="muted auth-note">
              Learning style and pace are primarily used for student accounts.
            </p>
          )}

          {error && <p className="error">{error}</p>}

          <button className="btn-primary auth-submit" disabled={!canSubmit} type="submit">
            {submitting ? "Creating account..." : "Create Account"}
          </button>
        </form>

        <p className="auth-footnote">
          Already have an account? <Link to="/login">Log in</Link>
        </p>
      </section>
    </main>
  );
}
