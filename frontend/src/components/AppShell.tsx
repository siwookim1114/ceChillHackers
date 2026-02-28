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
  const displayName = isAuthenticated ? authUser?.display_name ?? "Student" : "Guest";
  const roleLabel = isAuthenticated ? authUser?.role ?? "student" : level;
  const avatarInitial = (displayName?.trim().charAt(0) || "G").toUpperCase();

  const handleExit = () => {
    if (isAuthenticated) {
      clearAuthSession();
      navigate("/", { replace: true });
      return;
    }
    navigate("/", { replace: true });
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

        <div className="sidebar-profile">
          <div className="sidebar-avatar">{avatarInitial}</div>
          <div className="sidebar-profile-copy">
            <strong>{displayName}</strong>
            <small>{roleLabel}</small>
          </div>
        </div>

        <div className="user-status">
          <span>Current level</span>
          <strong>{level}</strong>
        </div>

        <nav className="side-nav">
          {NAV_ITEMS.map((item) => (
            <NavLink key={item.label} to={item.to} className="side-link">
              <span className="side-link-dot" />
              {item.label}
            </NavLink>
          ))}
        </nav>

        <div className="sidebar-note">
          <p>Today&apos;s focus</p>
          <strong>Productive Struggle</strong>
          <span>Small hints. Strong understanding.</span>
        </div>

        <button className="btn-muted side-cta" onClick={handleExit} type="button">
          {isAuthenticated ? "Log Out" : "Exit Session"}
        </button>
      </aside>

      <section className="shell-main">
        <header className="shell-topbar">
          <div className="topbar-left">
            <label className="search-wrap">
              <span>Search concepts</span>
              <input placeholder="Try: derivatives, factoring, equation setup, common mistakes..." />
            </label>
          </div>
          <div className="topbar-actions">
            <span className="pill good">Live Coaching</span>
            <span className="pill warm">Productive Struggle</span>
            <button className="btn-muted topbar-cta" onClick={() => navigate("/home#create-course")} type="button">
              Create Course
            </button>
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
