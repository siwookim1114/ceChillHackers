import { AppShell } from "../components/AppShell";

const POSTS = [
  {
    author: "Mina",
    title: "How I reduced erasing in algebra",
    body: "I started writing smaller checkpoints before final answer submission. It reduced repeated mistakes."
  },
  {
    author: "Leo",
    title: "Hint Level 1 worked better for me",
    body: "Question-first hints forced me to think. I solved faster than jumping to full explanation."
  },
  {
    author: "Kai",
    title: "My daily 20-minute rhythm",
    body: "One practice, one reflection note, one retry. Easy to keep and surprisingly effective."
  }
];

export function CommunityPage() {
  return (
    <AppShell title="Community" subtitle="Share learning tactics, compare routines, and improve together.">
      <section className="community-list reveal reveal-1">
        {POSTS.map((post) => (
          <article className="panel-card community-card" key={post.title}>
            <small>@{post.author}</small>
            <h4>{post.title}</h4>
            <p>{post.body}</p>
            <div className="community-actions">
              <button className="btn-muted" type="button">
                Reply
              </button>
              <button className="btn-muted" type="button">
                Save
              </button>
            </div>
          </article>
        ))}
      </section>
    </AppShell>
  );
}
