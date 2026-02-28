import { useNavigate } from "react-router-dom";

function makeGuestId() {
  return `guest_${Math.random().toString(36).slice(2, 10)}`;
}

export function EntryPage() {
  const navigate = useNavigate();

  const startAsGuest = () => {
    const id = localStorage.getItem("guest_id") ?? makeGuestId();
    localStorage.setItem("guest_id", id);
    navigate("/onboarding");
  };

  return (
    <main className="page page-entry">
      <section className="hero">
        <div className="hero-badge">
          <span className="live-dot" />
          Live Adaptive Coaching
        </div>

        <h1>Learn Better With Smart Intervention</h1>

        <p className="hero-description">
          Real-time coach that detects when you&apos;re stuck from your solving behavior
          and steps in with the <em>smallest</em> helpful hint — never the answer.
        </p>

        <div className="hero-features">
          <span className="feature-chip">🔍 Stuck Detection</span>
          <span className="feature-chip">💡 Hint Level 1–3</span>
          <span className="feature-chip">📊 Session Summary</span>
          <span className="feature-chip">⚡ Real-time Signals</span>
        </div>

        <div className="hero-cta">
          <button className="btn-primary" onClick={startAsGuest} type="button">
            Start as Guest →
          </button>
          <span className="hint-text">No account needed · Free forever</span>
        </div>
      </section>
    </main>
  );
}
