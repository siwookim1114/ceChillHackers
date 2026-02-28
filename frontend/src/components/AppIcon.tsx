import type { SVGProps } from "react";

export type AppIconName =
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
