type EyeToggleIconProps = {
  visible: boolean;
};

export function EyeToggleIcon({ visible }: EyeToggleIconProps) {
  if (visible) {
    return (
      <svg aria-hidden="true" className="password-eye-icon" fill="none" viewBox="0 0 24 24">
        <path
          d="M3 3L21 21"
          stroke="currentColor"
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth="1.8"
        />
        <path
          d="M10.58 10.58a2 2 0 0 0 2.83 2.83"
          stroke="currentColor"
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth="1.8"
        />
        <path
          d="M9.88 4.24A11.9 11.9 0 0 1 12 4c4.5 0 8.27 2.94 9.54 7-.43 1.38-1.16 2.64-2.1 3.68M6.6 6.6A11.98 11.98 0 0 0 2.46 11C3.73 15.06 7.5 18 12 18c1.12 0 2.2-.18 3.2-.51"
          stroke="currentColor"
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth="1.8"
        />
      </svg>
    );
  }

  return (
    <svg aria-hidden="true" className="password-eye-icon" fill="none" viewBox="0 0 24 24">
      <path
        d="M2.5 12C3.9 7.9 7.64 5 12 5s8.1 2.9 9.5 7c-1.4 4.1-5.14 7-9.5 7s-8.1-2.9-9.5-7Z"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.8"
      />
      <circle cx="12" cy="12" r="2.6" stroke="currentColor" strokeWidth="1.8" />
    </svg>
  );
}
