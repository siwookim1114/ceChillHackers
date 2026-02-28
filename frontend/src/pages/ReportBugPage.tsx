import { FormEvent, useState } from "react";
import { AppShell } from "../components/AppShell";

export function ReportBugPage() {
  const [title, setTitle] = useState("");
  const [details, setDetails] = useState("");
  const [submitted, setSubmitted] = useState(false);

  const onSubmit = (event: FormEvent) => {
    event.preventDefault();
    setSubmitted(true);
  };

  return (
    <AppShell title="Report a Bug" subtitle="Tell us what broke and we will prioritize the fix.">
      <section className="utility-grid single">
        <article className="panel-card utility-card">
          <form className="utility-form" onSubmit={onSubmit}>
            <label>
              Bug Title
              <input
                onChange={(event) => setTitle(event.target.value)}
                placeholder="Example: Profile dropdown closes too early"
                required
                value={title}
              />
            </label>

            <label>
              Details
              <textarea
                onChange={(event) => setDetails(event.target.value)}
                placeholder="Steps to reproduce and expected behavior..."
                required
                rows={6}
                value={details}
              />
            </label>

            <button className="btn-primary" type="submit">
              Submit Bug Report
            </button>
          </form>

          {submitted && (
            <p className="muted">Thanks. We logged your report and will investigate it.</p>
          )}
        </article>
      </section>
    </AppShell>
  );
}
