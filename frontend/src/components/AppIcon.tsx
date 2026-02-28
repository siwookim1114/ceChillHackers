import type { SVGProps } from "react";

export type AppIconName =
  | "nav-dashboard"
  | "nav-practice"
  | "nav-create-course"
  | "nav-planner"
  | "nav-progress"
  | "nav-library"
  | "nav-community"
  | "nav-settings"
  | "menu-edit"
  | "menu-friends"
  | "menu-saved"
  | "menu-folder"
  | "menu-help"
  | "menu-bug"
  | "menu-theme"
  | "menu-signout"
  | "subject-english"
  | "subject-spanish"
  | "subject-french"
  | "level-beginner"
  | "level-intermediate"
  | "level-advanced"
  | "style-socratic"
  | "style-step"
  | "style-concept"
  | "insight-hard"
  | "insight-good"
  | "timeline-start"
  | "timeline-intervention"
  | "timeline-solved"
  | "timeline-default"
  | "erase";

type AppIconProps = {
  name: AppIconName;
  className?: string;
};

export function AppIcon({ name, className }: AppIconProps) {
  const baseProps: SVGProps<SVGSVGElement> = {
    viewBox: "0 0 24 24",
    fill: "none",
    className,
    "aria-hidden": true
  };

  if (name === "subject-english") {
    return (
      <svg {...baseProps}>
        <path d="M4 3h16v18H4z" stroke="currentColor" strokeWidth="1.8" />
        <path d="M8 7h8M8 12h8M8 17h8" stroke="currentColor" strokeLinecap="round" strokeWidth="1.8" />
      </svg>
    );
  }
  if (name === "nav-dashboard") {
    return (
      <svg {...baseProps}>
        <rect x="4" y="4" width="7.5" height="7.5" rx="2" stroke="currentColor" strokeWidth="1.8" />
        <rect x="12.5" y="4" width="7.5" height="5.5" rx="2" stroke="currentColor" strokeWidth="1.8" />
        <rect x="4" y="12.5" width="7.5" height="7.5" rx="2" stroke="currentColor" strokeWidth="1.8" />
        <rect x="12.5" y="10.5" width="7.5" height="9.5" rx="2" stroke="currentColor" strokeWidth="1.8" />
      </svg>
    );
  }
  if (name === "nav-practice") {
    return (
      <svg {...baseProps}>
        <path
          d="M7 6.5h10a2 2 0 0 1 2 2v9H5v-9a2 2 0 0 1 2-2Zm0 0V4.8A1.8 1.8 0 0 1 8.8 3h6.4A1.8 1.8 0 0 1 17 4.8v1.7"
          stroke="currentColor"
          strokeWidth="1.8"
        />
        <path d="M10 11.2h4M10 14.4h4" stroke="currentColor" strokeLinecap="round" strokeWidth="1.8" />
      </svg>
    );
  }
  if (name === "nav-create-course") {
    return (
      <svg {...baseProps}>
        <rect x="4.5" y="4.5" width="15" height="15" rx="3" stroke="currentColor" strokeWidth="1.8" />
        <path d="M12 8v8M8 12h8" stroke="currentColor" strokeLinecap="round" strokeWidth="1.8" />
      </svg>
    );
  }
  if (name === "nav-planner") {
    return (
      <svg {...baseProps}>
        <rect x="4" y="5" width="16" height="15" rx="3" stroke="currentColor" strokeWidth="1.8" />
        <path d="M8 3v4M16 3v4M7.5 11h9M7.5 14.5h6" stroke="currentColor" strokeLinecap="round" strokeWidth="1.8" />
      </svg>
    );
  }
  if (name === "nav-progress") {
    return (
      <svg {...baseProps}>
        <path d="M5 18.5h14" stroke="currentColor" strokeLinecap="round" strokeWidth="1.8" />
        <path d="M7.5 16V10M12 16V7M16.5 16v-4.5" stroke="currentColor" strokeLinecap="round" strokeWidth="1.8" />
      </svg>
    );
  }
  if (name === "nav-library") {
    return (
      <svg {...baseProps}>
        <path d="M6 5h10v14H6zM16 7h2.5v12H8.5" stroke="currentColor" strokeWidth="1.8" />
        <path d="M8.5 9.5h5" stroke="currentColor" strokeLinecap="round" strokeWidth="1.8" />
      </svg>
    );
  }
  if (name === "nav-community") {
    return (
      <svg {...baseProps}>
        <circle cx="8.5" cy="9" r="2.5" stroke="currentColor" strokeWidth="1.8" />
        <circle cx="15.5" cy="10" r="2.5" stroke="currentColor" strokeWidth="1.8" />
        <path d="M4.5 18c.5-2.2 2.2-3.4 4-3.4S12 15.8 12.5 18M11.5 18c.4-2 2-3.1 3.8-3.1s3.2 1.2 3.7 3.1" stroke="currentColor" strokeLinecap="round" strokeWidth="1.8" />
      </svg>
    );
  }
  if (name === "nav-settings") {
    return (
      <svg {...baseProps}>
        <circle cx="12" cy="12" r="2.5" stroke="currentColor" strokeWidth="1.8" />
        <path
          d="m19 12 1.7-1-1.3-2.2-2 .4a7 7 0 0 0-1.2-1.1l.3-2.1h-2.6l-.6 1.9a7 7 0 0 0-1.4 0l-.6-1.9H8.8l.3 2.1c-.4.3-.8.7-1.2 1.1l-2-.4L4.6 11 6.3 12l-.4 1.3-1.7 1 1.3 2.2 2-.4c.4.4.8.8 1.2 1.1l-.3 2.1h2.6l.6-1.9h1.4l.6 1.9h2.6l-.3-2.1c.4-.3.8-.7 1.2-1.1l2 .4 1.3-2.2-1.7-1z"
          stroke="currentColor"
          strokeLinejoin="round"
          strokeWidth="1.4"
        />
      </svg>
    );
  }
  if (name === "menu-edit") {
    return (
      <svg {...baseProps}>
        <path d="m6 16 1.2 2L9.5 17 18 8.5 15.5 6 7 14.5 6 16Z" stroke="currentColor" strokeLinejoin="round" strokeWidth="1.8" />
      </svg>
    );
  }
  if (name === "menu-friends") {
    return (
      <svg {...baseProps}>
        <circle cx="8" cy="10" r="2.5" stroke="currentColor" strokeWidth="1.8" />
        <circle cx="16" cy="10.5" r="2.5" stroke="currentColor" strokeWidth="1.8" />
        <path d="M3.8 18c.4-2.1 2-3.2 4-3.2s3.4 1.1 3.9 3.2M12 18c.3-1.9 1.8-3 3.7-3s3.4 1.1 3.8 3" stroke="currentColor" strokeLinecap="round" strokeWidth="1.8" />
      </svg>
    );
  }
  if (name === "menu-saved") {
    return (
      <svg {...baseProps}>
        <path d="M7 4.5h10a1 1 0 0 1 1 1V20l-6-3.6L6 20V5.5a1 1 0 0 1 1-1Z" stroke="currentColor" strokeLinejoin="round" strokeWidth="1.8" />
      </svg>
    );
  }
  if (name === "menu-folder") {
    return (
      <svg {...baseProps}>
        <path d="M4 7.5h5l1.5 2H20v8.5a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V7.5Z" stroke="currentColor" strokeLinejoin="round" strokeWidth="1.8" />
      </svg>
    );
  }
  if (name === "menu-help") {
    return (
      <svg {...baseProps}>
        <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="1.8" />
        <path d="M12 16h.01M10.9 9.5a1.9 1.9 0 1 1 2.4 1.9c-.8.3-1.2.7-1.2 1.5" stroke="currentColor" strokeLinecap="round" strokeWidth="1.8" />
      </svg>
    );
  }
  if (name === "menu-bug") {
    return (
      <svg {...baseProps}>
        <path d="M8 10.5h8v5a4 4 0 0 1-8 0v-5Z" stroke="currentColor" strokeWidth="1.8" />
        <path d="M9.5 8.5a2.5 2.5 0 0 1 5 0M6 11h2M16 11h2M6 14h2M16 14h2" stroke="currentColor" strokeLinecap="round" strokeWidth="1.8" />
      </svg>
    );
  }
  if (name === "menu-theme") {
    return (
      <svg {...baseProps}>
        <circle cx="12" cy="12" r="4.2" stroke="currentColor" strokeWidth="1.8" />
        <path d="M12 3v2M12 19v2M3 12h2M19 12h2M5.6 5.6l1.4 1.4M17 17l1.4 1.4M5.6 18.4 7 17M17 7l1.4-1.4" stroke="currentColor" strokeLinecap="round" strokeWidth="1.8" />
      </svg>
    );
  }
  if (name === "menu-signout") {
    return (
      <svg {...baseProps}>
        <path d="M10 5H7a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h3" stroke="currentColor" strokeLinecap="round" strokeWidth="1.8" />
        <path d="m14 9 5 3-5 3M19 12h-9" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.8" />
      </svg>
    );
  }
  if (name === "subject-spanish") {
    return (
      <svg {...baseProps}>
        <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="1.8" />
        <path
          d="M3 12h18M12 3a15 15 0 0 1 0 18M12 3a15 15 0 0 0 0 18"
          stroke="currentColor"
          strokeLinecap="round"
          strokeWidth="1.5"
        />
      </svg>
    );
  }
  if (name === "subject-french") {
    return (
      <svg {...baseProps}>
        <path d="M12 3l2 5h4l-3.2 2.6 1.2 4L12 12l-4 2.6 1.2-4L6 8h4z" stroke="currentColor" strokeWidth="1.8" />
      </svg>
    );
  }
  if (name === "level-beginner") {
    return (
      <svg {...baseProps}>
        <path d="M12 21c5 0 9-4.1 9-9.1S17 2.8 12 2.8 3 6.9 3 11.9 7 21 12 21Z" stroke="currentColor" strokeWidth="1.8" />
        <path d="M12 16V9M9 12h6" stroke="currentColor" strokeLinecap="round" strokeWidth="1.8" />
      </svg>
    );
  }
  if (name === "level-intermediate") {
    return (
      <svg {...baseProps}>
        <path d="M6 17 12 3l6 14-6 4-6-4Z" stroke="currentColor" strokeLinejoin="round" strokeWidth="1.8" />
      </svg>
    );
  }
  if (name === "level-advanced") {
    return (
      <svg {...baseProps}>
        <path d="M12 3 4 12h6l-1 9 8-9h-6z" stroke="currentColor" strokeLinejoin="round" strokeWidth="1.8" />
      </svg>
    );
  }
  if (name === "style-socratic") {
    return (
      <svg {...baseProps}>
        <circle cx="12" cy="11" r="8" stroke="currentColor" strokeWidth="1.8" />
        <path d="M12 15h.01M11.2 8.8a1.6 1.6 0 1 1 2.2 1.5c-.8.4-1.2.8-1.2 1.7" stroke="currentColor" strokeLinecap="round" strokeWidth="1.8" />
      </svg>
    );
  }
  if (name === "style-step") {
    return (
      <svg {...baseProps}>
        <path d="M5 18h4v-4h4v-4h4V6h2" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.8" />
        <circle cx="5" cy="18" r="1.5" fill="currentColor" />
      </svg>
    );
  }
  if (name === "style-concept") {
    return (
      <svg {...baseProps}>
        <path d="M12 3a7 7 0 0 0-4 12.7V19a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1v-3.3A7 7 0 0 0 12 3Z" stroke="currentColor" strokeWidth="1.8" />
        <path d="M9 21h6" stroke="currentColor" strokeLinecap="round" strokeWidth="1.8" />
      </svg>
    );
  }
  if (name === "insight-hard") {
    return (
      <svg {...baseProps}>
        <path d="M13 2 6.5 13h4L9 22l8.5-12h-4L15 2z" fill="currentColor" />
      </svg>
    );
  }
  if (name === "insight-good") {
    return (
      <svg {...baseProps}>
        <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="1.8" />
        <path d="m8 12.3 2.4 2.4L16 9.5" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.8" />
      </svg>
    );
  }
  if (name === "timeline-start") {
    return (
      <svg {...baseProps}>
        <path d="m9 7 8 5-8 5z" fill="currentColor" />
      </svg>
    );
  }
  if (name === "timeline-intervention") {
    return (
      <svg {...baseProps}>
        <path d="M12 3 9 10h3l-1 11 4-8h-3l2-10z" fill="currentColor" />
      </svg>
    );
  }
  if (name === "timeline-solved") {
    return (
      <svg {...baseProps}>
        <path d="m7 12.5 3.2 3.2L17 9" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" />
      </svg>
    );
  }
  if (name === "erase") {
    return (
      <svg {...baseProps}>
        <path d="M15 18H8l-4-4 7-7 6 6-5 5Z" stroke="currentColor" strokeLinejoin="round" strokeWidth="1.8" />
        <path d="M14 18h6" stroke="currentColor" strokeLinecap="round" strokeWidth="1.8" />
      </svg>
    );
  }
  return (
    <svg {...baseProps}>
      <circle cx="12" cy="12" fill="currentColor" r="3" />
    </svg>
  );
}
