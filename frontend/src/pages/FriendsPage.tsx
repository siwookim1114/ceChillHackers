import { AppShell } from "../components/AppShell";

const FRIENDS = [
  { name: "Mina", streak: 12, focus: "Differentiation" },
  { name: "Leo", streak: 7, focus: "Quadratic Equations" },
  { name: "Kai", streak: 18, focus: "Public Speaking" }
];

export function FriendsPage() {
  return (
    <AppShell title="Friends" subtitle="Check peer streaks and keep each other accountable.">
      <section className="utility-grid">
        {FRIENDS.map((friend) => (
          <article className="panel-card utility-card" key={friend.name}>
            <h4>{friend.name}</h4>
            <p>Current focus: {friend.focus}</p>
            <p>Streak: {friend.streak} days</p>
            <button className="btn-muted" type="button">
              Send Nudge
            </button>
          </article>
        ))}
      </section>
    </AppShell>
  );
}
