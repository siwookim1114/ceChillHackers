import { AppShell } from "../components/AppShell";

export function SupportPage() {
  return (
    <AppShell title="Customer Support" subtitle="Need help? Reach out and we will respond quickly.">
      <section className="utility-grid single">
        <article className="panel-card utility-card">
          <h4>Support Channels</h4>
          <ul className="utility-list plain">
            <li>Email: support@tutorcoach.app</li>
            <li>Discord: #help-desk</li>
            <li>Response time: under 24 hours</li>
          </ul>
          <button
            className="btn-primary"
            onClick={() => {
              window.location.href = "mailto:support@tutorcoach.app?subject=TutorCoach%20Support";
            }}
            type="button"
          >
            Contact Support
          </button>
        </article>
      </section>
    </AppShell>
  );
}
