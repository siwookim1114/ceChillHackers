import { AppShell } from "../components/AppShell";
import { getAuthUser } from "../auth";

export function SettingsPage() {
  const user = getAuthUser();
  const level = localStorage.getItem("preferred_level") ?? "Beginner";
  const style = localStorage.getItem("preferred_style") ?? "Socratic";

  return (
    <AppShell title="Settings" subtitle="Manage profile, study preferences, and account behavior.">
      <section className="settings-grid reveal reveal-1">
        <article className="panel-card settings-card">
          <h4>Profile</h4>
          <p>Name: {user?.display_name ?? "Guest User"}</p>
          <p>Email: {user?.email ?? "Not signed in"}</p>
          <p>Role: {user?.role ?? "guest"}</p>
        </article>

        <article className="panel-card settings-card">
          <h4>Learning Preferences</h4>
          <p>Level: {level}</p>
          <p>Coaching style: {style}</p>
          <p>Pace: {user?.learning_pace ?? "normal"}</p>
        </article>

        <article className="panel-card settings-card">
          <h4>Notifications</h4>
          <p>Daily reminder: Enabled</p>
          <p>Progress report: Enabled</p>
          <p>Community updates: Weekly</p>
        </article>
      </section>
    </AppShell>
  );
}
