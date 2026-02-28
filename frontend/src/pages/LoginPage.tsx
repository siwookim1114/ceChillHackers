import { FormEvent, useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { getGoogleAuthStartUrl, getMeWithToken, login } from "../api";
import { getAccessToken, saveAuthSession } from "../auth";
import { EyeToggleIcon } from "../components/EyeToggleIcon";

const QUICK_ACCESS_OPTIONS = [
  { icon: "G", label: "Continue with Google" },
  { icon: "F", label: "Continue with Facebook" },
  { icon: "A", label: "Continue with Apple" }
];

function SocialIcon({ type }: { type: "G" | "F" | "A" }) {
  if (type === "G") {
    return (
      <svg aria-hidden="true" className="social-icon" viewBox="0 0 24 24">
        <path
          d="M22 12.2c0-.7-.1-1.3-.2-2H12v3.8h5.6c-.2 1.2-.9 2.3-2 3l3.2 2.5c1.9-1.7 3.2-4.2 3.2-7.3Z"
          fill="#4285F4"
        />
        <path
          d="M12 22c2.9 0 5.3-1 7.1-2.6L16 16.9c-1 .7-2.3 1.2-4 1.2-3 0-5.5-2-6.4-4.8l-3.3 2.6A10 10 0 0 0 12 22Z"
          fill="#34A853"
        />
        <path
          d="M5.6 13.3A6 6 0 0 1 5.2 12c0-.5.1-.9.3-1.3L2.2 8A10 10 0 0 0 2 12c0 1.5.4 2.9 1.1 4.1l2.5-2.8Z"
          fill="#FBBC05"
        />
        <path
          d="M12 5.8c1.6 0 3 .6 4.1 1.7l3-3C17.3 2.8 14.9 2 12 2A10 10 0 0 0 3 8l3.3 2.7c.8-2.8 3.4-4.9 5.7-4.9Z"
          fill="#EA4335"
        />
      </svg>
    );
  }

  if (type === "F") {
    return (
      <svg aria-hidden="true" className="social-icon" viewBox="0 0 24 24">
        <circle cx="12" cy="12" fill="#1877F2" r="11" />
        <path
          d="M13.9 8.4h2V5.1h-2.4c-2.9 0-4.1 1.9-4.1 4.1v2.1H7.5v3.2h1.9v4.5h3.4v-4.5h2.4l.4-3.2h-2.8V9.8c0-.8.4-1.4 1.1-1.4Z"
          fill="#fff"
        />
      </svg>
    );
  }

  return (
    <svg aria-hidden="true" className="social-icon" viewBox="0 0 24 24">
      <path
        d="M16.7 13.4c0-2 1.6-3 1.7-3.1-1-1.4-2.7-1.6-3.2-1.6-1.3-.1-2.6.8-3.3.8-.8 0-1.9-.8-3.1-.7-1.6 0-3 .9-3.9 2.2-1.6 2.4-.4 5.9 1.1 8 .7 1 1.6 2 2.8 2 .9 0 1.3-.6 2.5-.6s1.5.6 2.6.6c1.2 0 1.9-1 2.6-2 .8-1.1 1.1-2.2 1.1-2.2 0 0-2-.8-2-3.4Zm-2.4-6c.6-.8 1-1.8.8-2.9-.9.1-2 .6-2.6 1.4-.6.7-1 1.8-.8 2.8 1 .1 2-.4 2.6-1.3Z"
        fill="#111827"
      />
    </svg>
  );
}

export function LoginPage() {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [rememberMe, setRememberMe] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const hash = window.location.hash.startsWith("#")
      ? window.location.hash.slice(1)
      : window.location.hash;
    const hashParams = new URLSearchParams(hash);
    const oauthToken = hashParams.get("access_token");
    const oauthError = hashParams.get("oauth_error");

    if (oauthError) {
      setError("Google login failed. Please try again.");
      window.history.replaceState({}, document.title, window.location.pathname + window.location.search);
      return;
    }

    if (oauthToken) {
      setSubmitting(true);
      getMeWithToken(oauthToken)
        .then((user) => {
          saveAuthSession(oauthToken, user);
          window.history.replaceState({}, document.title, window.location.pathname + window.location.search);
          navigate("/home", { replace: true });
        })
        .catch(() => {
          setError("Google login verification failed. Please try email login.");
          window.history.replaceState({}, document.title, window.location.pathname + window.location.search);
        })
        .finally(() => setSubmitting(false));
      return;
    }

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

  const startAsGuest = () => {
    const id = localStorage.getItem("guest_id") ?? `guest_${Math.random().toString(36).slice(2, 10)}`;
    localStorage.setItem("guest_id", id);
    navigate("/onboarding");
  };

  const handleQuickAccess = (label: string) => {
    setError(null);
    if (label === "Continue with Google") {
      const returnTo = `${window.location.origin}/login`;
      window.location.href = getGoogleAuthStartUrl(returnTo);
      return;
    }
    setError("This provider will be enabled soon. Please use Google or email login.");
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
            <a href="/create-course">Create Course</a>
            <a href="/progress">Session Results</a>
            <a href="/home">AI Coach</a>
          </nav>

          <div className="entry-top-actions">
            <button className="btn-entry-ghost" type="button">
              English, USD
            </button>
            <button className="btn-entry-outline" onClick={() => navigate("/signup")} type="button">
              Sign Up
            </button>
          </div>
        </div>
      </header>

      <section className="auth-plain-body">
        <div className="auth-plain-panel">
          <h1>Log in</h1>
          <p className="auth-plain-signup-links">
            New here? <Link to="/signup">Create your learner account</Link>
          </p>

          <div className="auth-social-list">
            {QUICK_ACCESS_OPTIONS.map((option) => (
              <button
                className="auth-social-btn"
                key={option.label}
                onClick={() => handleQuickAccess(option.label)}
                type="button"
              >
                <span className="social-icon-wrap">
                  <SocialIcon type={option.icon as "G" | "F" | "A"} />
                </span>
                {option.label}
              </button>
            ))}
          </div>

          <button className="auth-corporate-link" type="button">
            Team workspace SSO will be available in the next version
          </button>

          <button className="auth-corporate-link" onClick={startAsGuest} type="button">
            Start as guest demo
          </button>

          <div className="auth-divider">
            <span />
            <strong>or</strong>
            <span />
          </div>

          <form className="auth-plain-form" onSubmit={onSubmit}>
            <label>
              Email
              <input
                autoComplete="email"
                onChange={(event) => setEmail(event.target.value)}
                placeholder="Your email"
                required
                type="email"
                value={email}
              />
            </label>

            <label>
              Password
              <div className="auth-password-row">
                <input
                  autoComplete="current-password"
                  onChange={(event) => setPassword(event.target.value)}
                  placeholder="Your password"
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

            <button className="auth-inline-link" type="button">
              Forgot your password?
            </button>

            <label className="auth-remember-row">
              <input
                checked={rememberMe}
                onChange={(event) => setRememberMe(event.target.checked)}
                type="checkbox"
              />
              <span>Remember me</span>
            </label>

            {error && <p className="error">{error}</p>}

            <button className="auth-submit-plain" disabled={submitting} type="submit">
              {submitting ? "Logging in..." : "Log in"}
            </button>
          </form>

          <p className="auth-plain-terms">
            By logging in, you agree to TutorCoach <Link to="/signup">Terms</Link> and{" "}
            <Link to="/signup">Privacy Policy</Link>.
          </p>
        </div>
      </section>
    </main>
  );
}
