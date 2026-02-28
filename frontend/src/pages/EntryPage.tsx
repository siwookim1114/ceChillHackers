import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { getAccessToken } from "../auth";
import { AppIcon, type AppIconName } from "../components/AppIcon";

function makeGuestId() {
  return `guest_${Math.random().toString(36).slice(2, 10)}`;
}

const STATS = [
  { value: "100,000+", label: "Experienced tutors" },
  { value: "300,000+", label: "5-star tutor reviews" },
  { value: "120+", label: "Subjects taught" },
  { value: "180+", label: "Tutor nationalities" },
  { value: "4.8 / 5", label: "on the App Store" }
];

const TOP_SUBJECTS: Array<{ icon: AppIconName; title: string }> = [
  { icon: "subject-english", title: "English tutors" },
  { icon: "subject-spanish", title: "Spanish tutors" },
  { icon: "subject-french", title: "French tutors" }
];

function SoundToggleIcon({ muted }: { muted: boolean }) {
  if (muted) {
    return (
      <svg aria-hidden="true" className="sound-toggle-icon" fill="none" viewBox="0 0 24 24">
        <path
          d="M4 10h4l5-4v12l-5-4H4z"
          stroke="currentColor"
          strokeLinejoin="round"
          strokeWidth="1.8"
        />
        <path
          d="M16 9l5 5M21 9l-5 5"
          stroke="currentColor"
          strokeLinecap="round"
          strokeWidth="1.8"
        />
      </svg>
    );
  }
  return (
    <svg aria-hidden="true" className="sound-toggle-icon" fill="none" viewBox="0 0 24 24">
      <path
        d="M4 10h4l5-4v12l-5-4H4z"
        stroke="currentColor"
        strokeLinejoin="round"
        strokeWidth="1.8"
      />
      <path
        d="M16 9a4 4 0 0 1 0 6M18.5 7a7 7 0 0 1 0 10"
        stroke="currentColor"
        strokeLinecap="round"
        strokeWidth="1.8"
      />
    </svg>
  );
}

export function EntryPage() {
  const navigate = useNavigate();
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [isMuted, setIsMuted] = useState(true);

  useEffect(() => {
    if (getAccessToken()) {
      navigate("/home");
    }
  }, [navigate]);

  const startAsGuest = () => {
    const id = localStorage.getItem("guest_id") ?? makeGuestId();
    localStorage.setItem("guest_id", id);
    navigate("/onboarding");
  };

  const toggleMute = () => {
    const video = videoRef.current;
    if (!video) {
      return;
    }
    const nextMuted = !isMuted;
    video.muted = nextMuted;
    setIsMuted(nextMuted);
    if (!nextMuted) {
      void video.play().catch(() => {
        video.muted = true;
        setIsMuted(true);
      });
    }
  };

  return (
    <main className="entry-root">
      <section className="entry-hero-wrap">
        <div className="entry-container">
          <header className="entry-topbar">
            <button className="entry-brand" onClick={() => navigate("/")} type="button">
              <span className="entry-brand-mark">TC</span>
              <strong>TutorCoach</strong>
            </button>

            <nav className="entry-nav-links">
              <a href="#subjects">Practice Topics</a>
              <a href="#metrics">Learning Metrics</a>
              <a href="#subjects">Create Course</a>
              <a href="#metrics">Progress Proof</a>
            </nav>

            <div className="entry-top-actions">
              <button className="btn-entry-ghost" type="button">
                English, USD
              </button>
              <button className="btn-entry-outline" onClick={() => navigate("/login")} type="button">
                Log In
              </button>
            </div>
          </header>

          <div className="entry-hero-grid">
            <section className="entry-copy">
              <h1>
                Learn faster
                <br />
                with your best
                <br />
                AI tutor.
              </h1>

              <p>
                Real-time coaching that guides each step, detects stuck moments, and helps you
                break through without giving away the answer.
              </p>

              <div className="entry-cta-row">
                <button className="btn-entry-primary" onClick={() => navigate("/signup")} type="button">
                  Find your tutor →
                </button>
                <button className="btn-entry-link" onClick={startAsGuest} type="button">
                  Start as guest
                </button>
              </div>
            </section>

            <section className="entry-media">
              <div className="media-stack">
                <div className="media-layer media-layer-1" />
                <div className="media-layer media-layer-2" />
                <article className="media-main">
                  <video
                    autoPlay
                    className="hero-avatar-video"
                    loop
                    muted={isMuted}
                    playsInline
                    preload="metadata"
                    ref={videoRef}
                  >
                    <source src="/intro.mp4" type="video/mp4" />
                    Your browser does not support the video tag.
                  </video>
                  <button
                    aria-label={isMuted ? "Unmute intro video" : "Mute intro video"}
                    className="video-sound-toggle"
                    onClick={toggleMute}
                    type="button"
                  >
                    <SoundToggleIcon muted={isMuted} />
                  </button>
                  <div className="media-inset media-inset-badge">
                    <span className="live-dot-mini" />
                    <div>
                      <strong>Live AI Coach</strong>
                      <small>Real-time guidance</small>
                    </div>
                  </div>
                </article>
              </div>
            </section>
          </div>
        </div>
      </section>

      <section className="entry-metrics" id="metrics">
        <div className="entry-container entry-metrics-grid">
          {STATS.map((item) => (
            <article key={item.label}>
              <h3>{item.value}</h3>
              <p>{item.label}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="entry-subjects" id="subjects">
        <div className="entry-container entry-subject-grid">
          {TOP_SUBJECTS.map((subject) => (
            <article className="entry-subject-card" key={subject.title}>
              <span className="entry-subject-icon">
                <AppIcon name={subject.icon} />
              </span>
              <h4>{subject.title}</h4>
              <button onClick={() => navigate("/signup")} type="button">
                →
              </button>
            </article>
          ))}
        </div>
      </section>
    </main>
  );
}
