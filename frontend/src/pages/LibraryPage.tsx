import { AppShell } from "../components/AppShell";

const RESOURCES = [
  {
    title: "Derivatives Quick Guide",
    tag: "Math",
    desc: "Core rules, mistakes, and visual intuition in one concise sheet."
  },
  {
    title: "Quadratic Strategies",
    tag: "Math",
    desc: "Factorization, formula, and graph-based approach with examples."
  },
  {
    title: "Public Speaking Basics",
    tag: "Communication",
    desc: "Structure your talk and control pacing under pressure."
  },
  {
    title: "Financial Literacy Starter",
    tag: "Life Skills",
    desc: "Budgeting, compounding, and practical decision frameworks."
  }
];

export function LibraryPage() {
  return (
    <AppShell title="Library" subtitle="Organized resources for faster revision and deeper understanding.">
      <section className="library-grid reveal reveal-1">
        {RESOURCES.map((item) => (
          <article className="panel-card library-card" key={item.title}>
            <span>{item.tag}</span>
            <h4>{item.title}</h4>
            <p>{item.desc}</p>
            <button className="btn-muted" type="button">
              Open Resource
            </button>
          </article>
        ))}
      </section>
    </AppShell>
  );
}
