import { ImageResponse } from "next/og";

export const dynamic = "force-static";
export const size = { width: 180, height: 180 };
export const contentType = "image/png";

export default function AppleIcon() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: "linear-gradient(135deg, #1a221d 0%, #07090a 100%)",
        }}
      >
        <svg width="130" height="130" viewBox="0 0 64 64" fill="none">
          <defs>
            <linearGradient id="ag" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stopColor="#f4cf8a" />
              <stop offset="100%" stopColor="#b07744" />
            </linearGradient>
          </defs>
          <circle cx="46" cy="18" r="7" fill="url(#ag)" opacity="0.95" />
          <path
            d="M6 42 C 16 28, 22 26, 30 34 C 36 26, 44 28, 54 38 C 46 40, 38 42, 32 46 C 24 42, 14 42, 6 42 Z"
            fill="url(#ag)"
          />
        </svg>
      </div>
    ),
    { ...size }
  );
}
