import { AppShell } from "../components/AppShell";

const SAVED_ITEMS = [
  "Derivative checklist and quick rules",
  "Common factoring mistakes",
  "Socratic prompts for guided solving"
];

export function SavedPage() {
  return (
    <AppShell title="Saved" subtitle="Your bookmarked materials and reusable snippets.">
      <section className="utility-grid single">
        <article className="panel-card utility-card">
          <h4>Saved Resources</h4>
          <ul className="utility-list">
            {SAVED_ITEMS.map((item) => (
              <li key={item}>
                <span>{item}</span>
                <button className="btn-muted" type="button">
                  Open
                </button>
              </li>
            ))}
          </ul>
        </article>
      </section>
    </AppShell>
  );
}
