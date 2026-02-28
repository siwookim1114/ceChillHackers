import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { getAccessToken } from "../auth";
import { AppIcon, type AppIconName } from "../components/AppIcon";

function makeGuestId() {
  return `guest_${Math.random().toString(36).slice(2, 10)}`;
}

const WORKFLOW_STEPS = [
  {
    step: "01",
    title: "Create a course in 30 seconds",
    description: "Type what you want to learn and generate a guided practice track instantly."
  },
  {
    step: "02",
    title: "Solve with live coaching",
    description: "The coach detects hesitation, repeated mistakes, and gives just-enough hints."
  },
  {
    step: "03",
    title: "Review your stuck timeline",
    description: "See where you struggled, what hint level appeared, and how you recovered."
  }
];

const FEATURE_CARDS: Array<{ icon: AppIconName; title: string; description: string }> = [
  {
    icon: "nav-practice",
    title: "Practice Studio",
    description: "Handwrite or type your solution while real-time signals track your solving behavior."
  },
  {
    icon: "nav-planner",
    title: "Study Planner",
    description: "Set daily missions and keep momentum with a clean, goal-driven learning routine."
  },
  {
    icon: "nav-progress",
    title: "Progress Analytics",
    description: "Understand your streak, stuck score trend, and intervention impact at a glance."
  }
];

const COURSE_TEMPLATES: Array<{ icon: AppIconName; title: string; description: string; tag: string }> = [
  {
    icon: "subject-english",
    title: "SAT Reading Sprint",
    description: "Inference drills, elimination strategy, and fast feedback on reasoning steps.",
    tag: "Exam Prep"
  },
  {
    icon: "subject-spanish",
    title: "Algebra Foundations",
    description: "Linear equations, factoring, and error-focused hinting for durable mastery.",
    tag: "Math"
  },
  {
    icon: "subject-french",
    title: "Public Speaking Builder",
    description: "Structure, delivery, and confidence exercises with bite-sized daily missions.",
    tag: "Communication"
  }
];

const TESTIMONIALS = [
  {
    quote: "The coach does not just tell me the answer. It helps me think, and that changed my study habits.",
    name: "Mina K.",
    role: "High school learner"
  },
  {
    quote: "I can see exactly where I got stuck and why. The timeline makes revision way more effective.",
    name: "Daniel R.",
    role: "College freshman"
  },
  {
    quote: "Creating custom courses from my own topics is crazy fast. Perfect for hackathon demos too.",
    name: "Noah P.",
    role: "Student builder"
  }
];

const FAQS = [
  {
    q: "Does it give the final answer immediately?",
    a: "No. The coach starts with minimal guidance and escalates hint levels only when needed."
  },
  {
    q: "Can I make my own course topic?",
    a: "Yes. Open Create New Course from the sidebar to generate a custom guided problem instantly."
  },
  {
    q: "Can I use the same account on another laptop?",
    a: "Yes. Progress is stored in the shared database so your dashboard stays consistent across devices."
  }
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
              <a href="#workflow">How It Works</a>
              <a href="#features">Features</a>
              <a href="#courses">Course Templates</a>
              <a href="#faq">FAQ</a>
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
                  Start Learning →
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

      <section className="entry-workflow" id="workflow">
        <div className="entry-container">
          <div className="entry-section-head">
            <p className="overline">How It Works</p>
            <h2>A practical loop for real learning progress</h2>
          </div>
          <div className="entry-workflow-grid">
            {WORKFLOW_STEPS.map((item) => (
              <article className="entry-workflow-card" key={item.step}>
                <span>{item.step}</span>
                <h3>{item.title}</h3>
                <p>{item.description}</p>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section className="entry-features" id="features">
        <div className="entry-container entry-feature-grid">
          {FEATURE_CARDS.map((feature) => (
            <article className="entry-feature-card" key={feature.title}>
              <span className="entry-feature-icon">
                <AppIcon name={feature.icon} />
              </span>
              <h3>{feature.title}</h3>
              <p>{feature.description}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="entry-subjects" id="courses">
        <div className="entry-container entry-subject-grid">
          {COURSE_TEMPLATES.map((subject) => (
            <article className="entry-subject-card" key={subject.title}>
              <span className="entry-subject-tag">{subject.tag}</span>
              <span className="entry-subject-icon">
                <AppIcon name={subject.icon} />
              </span>
              <h4>{subject.title}</h4>
              <p>{subject.description}</p>
              <button onClick={() => navigate("/signup")} type="button">
                Try this template
              </button>
            </article>
          ))}
        </div>
      </section>

      <section className="entry-testimonials" id="reviews">
        <div className="entry-container">
          <div className="entry-section-head">
            <p className="overline">Learner Feedback</p>
            <h2>Students use TutorCoach to stay consistent and unstuck</h2>
          </div>
          <div className="entry-testimonial-grid">
            {TESTIMONIALS.map((item) => (
              <article className="entry-testimonial-card" key={item.name}>
                <p>&ldquo;{item.quote}&rdquo;</p>
                <strong>{item.name}</strong>
                <small>{item.role}</small>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section className="entry-faq" id="faq">
        <div className="entry-container">
          <div className="entry-section-head">
            <p className="overline">FAQ</p>
            <h2>Everything you need before starting</h2>
          </div>
          <div className="entry-faq-list">
            {FAQS.map((item) => (
              <details className="entry-faq-item" key={item.q}>
                <summary>{item.q}</summary>
                <p>{item.a}</p>
              </details>
            ))}
          </div>
        </div>
      </section>

      <section className="entry-bottom-cta">
        <div className="entry-container entry-bottom-cta-inner">
          <div>
            <p className="overline">Ready To Start</p>
            <h2>Build your first custom course and begin today</h2>
          </div>
          <div className="entry-cta-row">
            <button className="btn-entry-primary" onClick={() => navigate("/signup")} type="button">
              Create Account
            </button>
            <button className="btn-entry-link" onClick={startAsGuest} type="button">
              Try Guest Demo
            </button>
          </div>
        </div>
      </section>
    </main>
  );
}
