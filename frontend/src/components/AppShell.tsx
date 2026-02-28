import { useEffect, useRef, useState, type ReactNode } from "react";
import { NavLink, useNavigate } from "react-router-dom";
import { clearAuthSession, getAccessToken, getAuthUser } from "../auth";
import { AppIcon, type AppIconName } from "./AppIcon";

type AppShellProps = {
  title: string;
  subtitle: string;
  children: ReactNode;
};

const NAV_ITEMS = [
  { label: "Dashboard", to: "/home", icon: "nav-dashboard" as AppIconName },
  { label: "Practice Studio", to: "/practice", icon: "nav-practice" as AppIconName },
  { label: "Study Planner", to: "/planner", icon: "nav-planner" as AppIconName },
  { label: "Progress", to: "/progress", icon: "nav-progress" as AppIconName },
  { label: "Library", to: "/library", icon: "nav-library" as AppIconName },
  { label: "Community", to: "/community", icon: "nav-community" as AppIconName },
  { label: "Settings", to: "/settings", icon: "nav-settings" as AppIconName }
];

export function AppShell({ title, subtitle, children }: AppShellProps) {
  const navigate = useNavigate();
  const [hideTopbar, setHideTopbar] = useState(false);
  const lastScrollYRef = useRef(0);
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

  useEffect(() => {
    const onScroll = () => {
      const currentY = window.scrollY;
      const previousY = lastScrollYRef.current;

      if (currentY < 24) {
        setHideTopbar(false);
      } else if (currentY > previousY + 8) {
        setHideTopbar(true);
      } else if (currentY < previousY - 6) {
        setHideTopbar(false);
      }

      lastScrollYRef.current = currentY;
    };

    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

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

        <div className="sidebar-section-label">Workspace</div>
        <nav className="side-nav">
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.label}
              to={item.to}
              className={({ isActive }) => `side-link ${isActive ? "active" : ""}`}
            >
              <span className="side-link-icon">
                <AppIcon className="side-link-icon-svg" name={item.icon} />
              </span>
              <span>{item.label}</span>
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
        <header className={`shell-topbar ${hideTopbar ? "topbar-hidden" : ""}`}>
          <div className="topbar-left">
            <label className="search-wrap minimalist">
              <input placeholder="Search concepts, notes, mistakes..." />
            </label>
          </div>
          <div className="topbar-actions">
            <button className="btn-muted topbar-cta" onClick={() => navigate("/practice#create-course")} type="button">
              Create Course
            </button>
            <button
              aria-label="Open profile"
              className="topbar-profile-btn"
              onClick={() => navigate("/home")}
              type="button"
            >
              {avatarInitial}
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
