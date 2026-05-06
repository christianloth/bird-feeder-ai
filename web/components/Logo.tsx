export function LogoMark({ size = 28 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 64 64"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden
    >
      <defs>
        <linearGradient id="lm-grad" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#f4cf8a" />
          <stop offset="100%" stopColor="#b07744" />
        </linearGradient>
      </defs>
      {/* moon */}
      <circle cx="46" cy="18" r="6" fill="url(#lm-grad)" opacity="0.9" />
      {/* abstract bird in flight, single-stroke */}
      <path
        d="M6 42 C 16 28, 22 26, 30 34 C 36 26, 44 28, 54 38 C 46 40, 38 42, 32 46 C 24 42, 14 42, 6 42 Z"
        fill="url(#lm-grad)"
        opacity="0.95"
      />
      <path
        d="M30 34 L 32 50"
        stroke="#0b100d"
        strokeWidth="1.4"
        strokeLinecap="round"
        opacity="0.4"
      />
    </svg>
  );
}

export function GrafanaIcon({ size = 14 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <path d="M3 18 L 9 12 L 13 16 L 21 6" />
      <path d="M21 6 L 21 12" opacity="0.55" />
      <path d="M21 6 L 15 6" opacity="0.55" />
    </svg>
  );
}

export function HomeIcon({ size = 18 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <path d="M3 11l9-8 9 8" />
      <path d="M5 10v10a1 1 0 0 0 1 1h4v-6h4v6h4a1 1 0 0 0 1-1V10" />
    </svg>
  );
}

export function ReviewIcon({ size = 18 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <circle cx="12" cy="12" r="9" />
      <path d="M8.5 12.5l2.5 2.5 4.5-5" />
    </svg>
  );
}
