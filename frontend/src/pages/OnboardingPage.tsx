import { useState } from "react";
import { useNavigate } from "react-router-dom";

const LEVELS = ["Beginner", "Intermediate", "Advanced"];
const STYLES = ["Socratic", "Step-by-step", "Concept-first"];

export function OnboardingPage() {
  const navigate = useNavigate();
  const [level, setLevel] = useState(localStorage.getItem("preferred_level") ?? LEVELS[0]);
  const [style, setStyle] = useState(localStorage.getItem("preferred_style") ?? STYLES[0]);

  const continueToHome = () => {
    localStorage.setItem("preferred_level", level);
    localStorage.setItem("preferred_style", style);
    navigate("/home");
  };

  return (
    <main className="page">
      <section className="card">
        <h2>Onboarding</h2>
        <p className="muted">Set your baseline so hints are tuned to your level and coaching style.</p>

        <label>
          Learning Level
          <select value={level} onChange={(event) => setLevel(event.target.value)}>
            {LEVELS.map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>
        </label>

        <label>
          Coaching Style
          <select value={style} onChange={(event) => setStyle(event.target.value)}>
            {STYLES.map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>
        </label>

        <button className="btn-primary" onClick={continueToHome}>
          Continue
        </button>
      </section>
    </main>
  );
}
