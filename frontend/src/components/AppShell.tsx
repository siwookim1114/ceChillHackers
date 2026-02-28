import type { ReactNode } from "react";
import { NavLink, useNavigate } from "react-router-dom";
import { clearAuthSession, getAccessToken, getAuthUser } from "../auth";

type AppShellProps = {
  title: string;
  subtitle: string;
  children: ReactNode;
};

const NAV_ITEMS = [
  { label: "Home", to: "/home" },
  { label: "Practice", to: "/home#practice" },
  { label: "Create Course", to: "/home#create-course" },
  { label: "Results", to: "/home#results" }
];

export function AppShell({ title, subtitle, children }: AppShellProps) {
  const navigate = useNavigate();
  const level = localStorage.getItem("preferred_level") ?? "Beginner";
  const authUser = getAuthUser();
  const isAuthenticated = Boolean(getAccessToken() && authUser);

  const handleExit = () => {
    if (isAuthenticated) {
      clearAuthSession();
      navigate("/login");
      return;
    }
    navigate("/");
  };

  return (
    <div className="dashboard-shell">
      <aside className="app-sidebar">
        <div className="brand-block">
          <div className="brand-mark">TC</div>
          <div>
            <h1>TutorCoach</h1>
            <p>Adaptive Learning</p>
          </div>
        </div>

        <div className="user-status">
          <span>{isAuthenticated ? authUser?.display_name : level}</span>
          <small>{isAuthenticated ? authUser?.role : "Guest"}</small>
        </div>

        <nav className="side-nav">
          {NAV_ITEMS.map((item) => (
            <NavLink key={item.label} to={item.to} className="side-link">
              {item.label}
            </NavLink>
          ))}
        </nav>

        <button className="btn-muted side-cta" onClick={handleExit} type="button">
          {isAuthenticated ? "Log Out" : "Exit Session"}
        </button>
      </aside>

      <section className="shell-main">
        <header className="shell-topbar">
          <label className="search-wrap">
            <span>Search</span>
            <input placeholder="Search topic or concept..." />
          </label>
          <div className="topbar-badges">
            <span className="pill good">Live Coaching</span>
            <span className="pill warm">Productive Struggle</span>
          </div>
        </header>

        <main className="shell-content">
          <div className="content-header">
            <h2>{title}</h2>
            <p>{subtitle}</p>
          </div>
          {children}
        </main>
      </section>
    </div>
  );
}
