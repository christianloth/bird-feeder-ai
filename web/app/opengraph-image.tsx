import { ImageResponse } from "next/og";

export const dynamic = "force-static";
export const alt = "Feeder Cam — An evening with the birds";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

export default async function OG() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
          padding: "80px 96px",
          background: "#0b100d",
          color: "#ece3cf",
          fontFamily: "ui-serif, Georgia, serif",
          position: "relative",
        }}
      >
        {/* Ember glow — top-left */}
        <div
          style={{
            position: "absolute",
            top: -240,
            left: -200,
            width: 720,
            height: 720,
            background:
              "radial-gradient(circle, rgba(224,169,109,0.32) 0%, rgba(224,169,109,0) 65%)",
            display: "flex",
          }}
        />
        {/* Meadow glow — bottom-right */}
        <div
          style={{
            position: "absolute",
            bottom: -260,
            right: -240,
            width: 820,
            height: 820,
            background:
              "radial-gradient(circle, rgba(127,169,122,0.22) 0%, rgba(127,169,122,0) 65%)",
            display: "flex",
          }}
        />

        {/* Top: brand */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 18,
            color: "#a9b3a8",
            fontSize: 22,
            letterSpacing: "0.28em",
            textTransform: "uppercase",
            fontFamily: "ui-monospace, monospace",
          }}
        >
          {/* Logo mark — moon + bird in flight */}
          <svg width="44" height="44" viewBox="0 0 64 64" fill="none">
            <defs>
              <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stopColor="#f4cf8a" />
                <stop offset="100%" stopColor="#b07744" />
              </linearGradient>
            </defs>
            <circle cx="46" cy="18" r="6" fill="url(#g)" opacity="0.9" />
            <path
              d="M6 42 C 16 28, 22 26, 30 34 C 36 26, 44 28, 54 38 C 46 40, 38 42, 32 46 C 24 42, 14 42, 6 42 Z"
              fill="url(#g)"
              opacity="0.95"
            />
          </svg>
          <span style={{ display: "flex" }}>Feeder Cam · Frisco, TX</span>
        </div>

        {/* Headline */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            lineHeight: 0.96,
            fontStyle: "italic",
            fontWeight: 300,
            letterSpacing: "-0.02em",
          }}
        >
          <div style={{ display: "flex", fontSize: 152, color: "#f6efe1" }}>
            An evening
          </div>
          <div style={{ display: "flex", fontSize: 152, color: "#f4cf8a" }}>
            with the birds.
          </div>
        </div>

        {/* Footer */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "flex-end",
            color: "#8c9a8f",
            fontSize: 26,
            fontFamily: "ui-sans-serif, system-ui, sans-serif",
          }}
        >
          <div style={{ display: "flex", maxWidth: 640, lineHeight: 1.35 }}>
            A feeder in Frisco, watched 24/7 by a Pi. Every visitor pressed
            into a quiet field log.
          </div>
          <div
            style={{
              display: "flex",
              gap: 14,
              fontFamily: "ui-monospace, monospace",
              fontSize: 18,
              letterSpacing: "0.18em",
              textTransform: "uppercase",
              color: "#6e7d72",
            }}
          >
            <span style={{ display: "flex" }}>YOLO11n</span>
            <span style={{ display: "flex", color: "#27302a" }}>/</span>
            <span style={{ display: "flex" }}>ViT-S</span>
            <span style={{ display: "flex", color: "#27302a" }}>/</span>
            <span style={{ display: "flex" }}>Hailo-10H</span>
          </div>
        </div>
      </div>
    ),
    { ...size }
  );
}
