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
        <p className="overline">RevisionDojo</p>
        <h1>Productive Struggle Coach</h1>
        <p>
          Real-time learning coach that detects stuck moments from your solving behavior and steps in with
          the smallest helpful hint.
        </p>
        <button className="btn-primary" onClick={startAsGuest}>
          Start as Guest
        </button>
      </section>
    </main>
  );
}
