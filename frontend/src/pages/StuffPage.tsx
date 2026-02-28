import { AppShell } from "../components/AppShell";

export function StuffPage() {
  return (
    <AppShell title="My Stuff" subtitle="Personal notes, uploads, and custom learning assets.">
      <section className="utility-grid">
        <article className="panel-card utility-card">
          <h4>Notes</h4>
          <p>Keep key mistakes and mini-insights after each session.</p>
          <button className="btn-muted" type="button">
            Open Notes
          </button>
        </article>

        <article className="panel-card utility-card">
          <h4>Uploads</h4>
          <p>Store worksheets, reference docs, and class materials.</p>
          <button className="btn-muted" type="button">
            Manage Files
          </button>
        </article>
      </section>
    </AppShell>
  );
}
