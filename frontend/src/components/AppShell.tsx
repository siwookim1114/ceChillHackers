import { useEffect, useRef, useState, type ReactNode } from "react";
import { NavLink, useNavigate } from "react-router-dom";
import { clearAuthSession, getAccessToken, getAuthUser } from "../auth";
import { AppIcon, type AppIconName } from "./AppIcon";

type AppShellProps = {
  title: ReactNode;
  subtitle: ReactNode;
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
  const [profileMenuOpen, setProfileMenuOpen] = useState(false);
  const [avatarLoadFailed, setAvatarLoadFailed] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(
    () => localStorage.getItem("theme_mode") === "dark"
  );
  const lastScrollYRef = useRef(0);
  const closeMenuTimerRef = useRef<number | null>(null);
  const profileMenuRef = useRef<HTMLDivElement | null>(null);
  const level = localStorage.getItem("preferred_level") ?? "Beginner";
  const authUser = getAuthUser();
  const isAuthenticated = Boolean(getAccessToken() && authUser);
  const displayName = isAuthenticated ? authUser?.display_name ?? "Student" : "Guest";
  const avatarInitial = (displayName?.trim().charAt(0) || "G").toUpperCase();
  const avatarSrc = "/profile_basic.png";
  const joinedText = isAuthenticated ? "Joined 165 days ago" : "Guest mode";

  const handleExit = () => {
    setProfileMenuOpen(false);
    if (isAuthenticated) {
      clearAuthSession();
      navigate("/", { replace: true });
      return;
    }
    navigate("/", { replace: true });
  };

  const goTo = (path: string) => {
    setProfileMenuOpen(false);
    navigate(path);
  };

  const clearCloseTimer = () => {
    if (closeMenuTimerRef.current !== null) {
      window.clearTimeout(closeMenuTimerRef.current);
      closeMenuTimerRef.current = null;
    }
  };

  const openProfileMenu = () => {
    clearCloseTimer();
    setProfileMenuOpen(true);
  };

  const scheduleCloseProfileMenu = () => {
    clearCloseTimer();
    closeMenuTimerRef.current = window.setTimeout(() => {
      setProfileMenuOpen(false);
      closeMenuTimerRef.current = null;
    }, 180);
  };

  useEffect(() => {
    const onScroll = () => {
      const currentY = window.scrollY;
      const previousY = lastScrollYRef.current;
      const delta = currentY - previousY;

      if (currentY <= 4) {
        setHideTopbar(false);
      } else if (delta > 1) {
        setHideTopbar(true);
      } else if (delta < -1) {
        setHideTopbar(false);
      }

      lastScrollYRef.current = currentY;
    };

    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", isDarkMode ? "dark" : "light");
    localStorage.setItem("theme_mode", isDarkMode ? "dark" : "light");
  }, [isDarkMode]);

  useEffect(() => {
    if (!profileMenuOpen) {
      return;
    }
    const onPointerDown = (event: MouseEvent) => {
      if (
        profileMenuRef.current &&
        !profileMenuRef.current.contains(event.target as Node)
      ) {
        setProfileMenuOpen(false);
      }
    };
    document.addEventListener("mousedown", onPointerDown);
    return () => document.removeEventListener("mousedown", onPointerDown);
  }, [profileMenuOpen]);

  useEffect(
    () => () => {
      clearCloseTimer();
    },
    []
  );

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
            <div
              className="topbar-profile-wrap"
              onMouseEnter={openProfileMenu}
              onMouseLeave={scheduleCloseProfileMenu}
              ref={profileMenuRef}
            >
              <button
                aria-expanded={profileMenuOpen}
                aria-haspopup="menu"
                aria-label="Open profile menu"
                className="topbar-profile-btn"
                onClick={() => {
                  clearCloseTimer();
                  setProfileMenuOpen((prev) => !prev);
                }}
                type="button"
              >
                {avatarLoadFailed ? (
                  avatarInitial
                ) : (
                  <img
                    alt=""
                    className="topbar-profile-image"
                    onError={() => setAvatarLoadFailed(true)}
                    src={avatarSrc}
                  />
                )}
              </button>

              <div
                className={`profile-dropdown ${profileMenuOpen ? "open" : ""}`}
                onMouseEnter={openProfileMenu}
                onMouseLeave={scheduleCloseProfileMenu}
                role="menu"
              >
                <div className="profile-dropdown-head">
                  <div className="profile-dropdown-avatar">
                    {avatarLoadFailed ? (
                      avatarInitial
                    ) : (
                      <img
                        alt={`${displayName} profile`}
                        className="profile-dropdown-avatar-image"
                        onError={() => setAvatarLoadFailed(true)}
                        src={avatarSrc}
                      />
                    )}
                  </div>
                  <div className="profile-dropdown-copy">
                    <strong>{displayName}</strong>
                    <small>{joinedText}</small>
                  </div>
                  <button
                    aria-label="Edit profile"
                    className="profile-edit-btn"
                    onClick={() => {
                      goTo("/settings");
                    }}
                    type="button"
                  >
                    <AppIcon className="profile-item-icon" name="menu-edit" />
                  </button>
                </div>

                <div className="profile-dropdown-section">
                  <button className="profile-item-btn" onClick={() => goTo("/friends")} type="button">
                    <AppIcon className="profile-item-icon" name="menu-friends" />
                    <span>Friends</span>
                  </button>
                  <button className="profile-item-btn" onClick={() => goTo("/saved")} type="button">
                    <AppIcon className="profile-item-icon" name="menu-saved" />
                    <span>Saved</span>
                  </button>
                  <button className="profile-item-btn" onClick={() => goTo("/stuff")} type="button">
                    <AppIcon className="profile-item-icon" name="menu-folder" />
                    <span>My Stuff</span>
                  </button>
                  <button className="profile-item-btn" onClick={() => goTo("/settings")} type="button">
                    <AppIcon className="profile-item-icon" name="nav-settings" />
                    <span>Settings</span>
                  </button>
                  <button className="profile-item-btn accent" onClick={() => goTo("/upgrade")} type="button">
                    <AppIcon className="profile-item-icon" name="nav-progress" />
                    <span>Upgrade</span>
                  </button>
                </div>

                <div className="profile-dropdown-section">
                  <button className="profile-item-btn" onClick={() => goTo("/support")} type="button">
                    <AppIcon className="profile-item-icon" name="menu-help" />
                    <span>Customer Support</span>
                  </button>
                  <button
                    className="profile-item-btn danger"
                    onClick={() => goTo("/report-bug")}
                    type="button"
                  >
                    <AppIcon className="profile-item-icon" name="menu-bug" />
                    <span>Report a Bug</span>
                  </button>
                  <button
                    className="profile-item-btn"
                    onClick={() => {
                      setProfileMenuOpen(false);
                      setIsDarkMode((prev) => !prev);
                    }}
                    type="button"
                  >
                    <AppIcon className="profile-item-icon" name="menu-theme" />
                    <span>{isDarkMode ? "Switch to Light" : "Switch to Dark"}</span>
                  </button>
                </div>

                <div className="profile-dropdown-section">
                  <button className="profile-item-btn" onClick={handleExit} type="button">
                    <AppIcon className="profile-item-icon" name="menu-signout" />
                    <span>{isAuthenticated ? "Sign Out" : "Exit Session"}</span>
                  </button>
                </div>
              </div>
            </div>
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
