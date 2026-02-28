import { AppShell } from "../components/AppShell";

const PLANS = [
  {
    name: "Starter",
    price: "$0",
    perks: ["Basic practice", "Daily mission tracking", "Community access"]
  },
  {
    name: "Pro",
    price: "$12/mo",
    perks: ["Advanced analytics", "Priority support", "Expanded course generation"]
  },
  {
    name: "Team",
    price: "$39/mo",
    perks: ["Shared workspace", "Instructor dashboard", "Team progress reports"]
  }
];

export function UpgradePage() {
  return (
    <AppShell title="Upgrade" subtitle="Choose the plan that fits your learning pace and workflow.">
      <section className="utility-grid">
        {PLANS.map((plan) => (
          <article className="panel-card utility-card" key={plan.name}>
            <h4>{plan.name}</h4>
            <strong className="utility-price">{plan.price}</strong>
            <ul className="utility-list plain">
              {plan.perks.map((perk) => (
                <li key={perk}>{perk}</li>
              ))}
            </ul>
            <button className="btn-primary" type="button">
              Choose {plan.name}
            </button>
          </article>
        ))}
      </section>
    </AppShell>
  );
}
