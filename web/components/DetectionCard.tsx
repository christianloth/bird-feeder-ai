"use client";

import { motion } from "motion/react";
import type { Detection } from "@/lib/types";
import { confidenceTier, formatPct, localTime, timeAgo } from "@/lib/format";
import { imageUrl } from "@/lib/api";

interface Props {
  detection: Detection;
  onView: (d: Detection) => void;
  onDelete: (d: Detection) => void;
  index: number;
}

export function DetectionCard({ detection, onView, onDelete, index }: Props) {
  const tier = confidenceTier(detection.confidence);
  const tierColor =
    tier === "high"
      ? "var(--color-meadow-300)"
      : tier === "medium"
      ? "var(--color-harvest-500)"
      : "var(--color-rust-500)";

  const speciesName = detection.corrected_species_name ?? detection.species_name ?? "Unknown";
  const status = detection.is_false_positive
    ? "false-positive"
    : detection.reviewed
    ? "confirmed"
    : "pending";

  return (
    <motion.article
      initial={{ opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: Math.min(index * 0.025, 0.4), ease: [0.22, 1, 0.36, 1] }}
      className="tilt-card glass group relative flex flex-col overflow-hidden rounded-[var(--radius-card)]"
    >
      <button
        type="button"
        onClick={() => onView(detection)}
        className="relative block aspect-[4/3] overflow-hidden bg-[var(--color-ink-900)]"
        aria-label={`View ${speciesName}`}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={imageUrl.crop(detection.id)}
          alt={speciesName}
          loading="lazy"
          className="h-full w-full object-cover transition-transform duration-700 group-hover:scale-105"
        />
        <div
          aria-hidden
          className="pointer-events-none absolute inset-0 bg-gradient-to-t from-[rgba(7,9,10,0.92)] via-transparent to-transparent"
        />
        <span
          className="absolute right-3 top-3 inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[0.65rem] uppercase tracking-[0.16em] backdrop-blur"
          style={{
            background:
              status === "confirmed"
                ? "rgba(127, 169, 122, 0.14)"
                : status === "false-positive"
                ? "rgba(201, 122, 92, 0.14)"
                : "rgba(140, 154, 143, 0.14)",
            color:
              status === "confirmed"
                ? "var(--color-meadow-300)"
                : status === "false-positive"
                ? "var(--color-rust-500)"
                : "var(--color-sage-100)",
            border: `1px solid color-mix(in oklab, ${
              status === "confirmed"
                ? "var(--color-meadow-500)"
                : status === "false-positive"
                ? "var(--color-rust-500)"
                : "var(--color-moss-500)"
            } 50%, transparent)`,
          }}
        >
          {status === "false-positive" ? "false +" : status}
        </span>
      </button>

      <div className="flex flex-col gap-2 p-4">
        <div className="flex items-baseline justify-between gap-3">
          <h3
            className="display-italic line-clamp-1 text-[1.2rem] leading-tight text-[var(--color-cream-50)]"
            title={speciesName}
          >
            {speciesName}
          </h3>
          <span className="eyebrow shrink-0" title={localTime(detection.timestamp)}>
            {timeAgo(detection.timestamp)}
          </span>
        </div>

        <div className="flex items-center gap-3">
          <div className="relative h-1.5 flex-1 overflow-hidden rounded-full bg-[var(--color-ink-700)]">
            <span
              className="fill-grow absolute inset-y-0 left-0 rounded-full"
              style={{
                width: `${Math.max(2, detection.confidence * 100)}%`,
                background: `linear-gradient(90deg, ${tierColor}, color-mix(in oklab, ${tierColor} 65%, var(--color-cream-100)))`,
              }}
            />
          </div>
          <span className="font-mono text-[0.72rem]" style={{ color: tierColor }}>
            {formatPct(detection.confidence, 1)}
          </span>
        </div>

        <div className="flex items-center justify-between text-[0.7rem] text-[var(--color-moss-300)]">
          <span className="font-mono">#{detection.id}</span>
          {detection.source ? (
            <span className="opacity-80">{detection.source}</span>
          ) : null}
        </div>

        <div className="mt-2 flex gap-2">
          <button
            type="button"
            onClick={() => onView(detection)}
            className="btn-quiet flex-1 justify-center !text-[0.78rem]"
          >
            Frame
          </button>
          <button
            type="button"
            onClick={() => onDelete(detection)}
            className="btn-quiet justify-center !text-[0.78rem] hover:!text-[var(--color-rust-500)] hover:!border-[color-mix(in_oklab,var(--color-rust-500)_40%,transparent)]"
            title="Delete detection"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
              <path d="M3 6h18" />
              <path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
              <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
            </svg>
          </button>
        </div>
      </div>
    </motion.article>
  );
}
