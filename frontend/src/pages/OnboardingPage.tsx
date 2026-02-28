import { useState } from "react";
import { useNavigate } from "react-router-dom";

const LEVELS: Array<{ value: string; icon: string; label: string; desc: string }> = [
  { value: "Beginner", icon: "🌱", label: "Beginner", desc: "New to this topic" },
  { value: "Intermediate", icon: "🔥", label: "Intermediate", desc: "Some experience" },
  { value: "Advanced", icon: "⚡", label: "Advanced", desc: "Comfortable with concepts" },
];

const STYLES: Array<{ value: string; icon: string; label: string; desc: string }> = [
  { value: "Socratic", icon: "❓", label: "Socratic", desc: "Guide me with questions" },
  { value: "Step-by-step", icon: "📍", label: "Step-by-step", desc: "Break it down for me" },
  { value: "Concept-first", icon: "💡", label: "Concept-first", desc: "Explain the big idea" },
];

export function OnboardingPage() {
  const navigate = useNavigate();
  const [level, setLevel] = useState(localStorage.getItem("preferred_level") ?? LEVELS[0].value);
  const [style, setStyle] = useState(localStorage.getItem("preferred_style") ?? STYLES[0].value);

  const continueToHome = () => {
    localStorage.setItem("preferred_level", level);
    localStorage.setItem("preferred_style", style);
    navigate("/home");
  };

  return (
    <main className="page">
      <section className="card onboarding-card">
        <div className="onboarding-header">
          <p className="overline">TutorCoach</p>
          <h2>Quick Setup</h2>
          <p className="muted">
            Hints will be tuned to your level and coaching style.
          </p>
        </div>

        <div className="option-group">
          <span>Your Level</span>
          <div className="option-cards">
            {LEVELS.map((item) => (
              <button
                key={item.value}
                className={`option-card${level === item.value ? " selected" : ""}`}
                onClick={() => setLevel(item.value)}
                type="button"
              >
                <span className="opt-icon">{item.icon}</span>
                <span className="opt-label">{item.label}</span>
                <span className="opt-desc">{item.desc}</span>
              </button>
            ))}
          </div>
        </div>

        <div className="option-group">
          <span>Coaching Style</span>
          <div className="option-cards">
            {STYLES.map((item) => (
              <button
                key={item.value}
                className={`option-card${style === item.value ? " selected" : ""}`}
                onClick={() => setStyle(item.value)}
                type="button"
              >
                <span className="opt-icon">{item.icon}</span>
                <span className="opt-label">{item.label}</span>
                <span className="opt-desc">{item.desc}</span>
              </button>
            ))}
          </div>
        </div>

        <button className="btn-primary" onClick={continueToHome} type="button">
          Let&apos;s Practice →
        </button>
      </section>
    </main>
  );
}
