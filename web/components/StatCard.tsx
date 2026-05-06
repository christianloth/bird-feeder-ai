"use client";

import { motion } from "motion/react";

interface Props {
  label: string;
  value: string | number | null | undefined;
  hint?: string;
  emphasis?: "ember" | "meadow" | "default";
  index?: number;
}

export function StatCard({ label, value, hint, emphasis = "default", index = 0 }: Props) {
  const valueColor =
    emphasis === "ember"
      ? "text-[var(--color-ember-400)]"
      : emphasis === "meadow"
      ? "text-[var(--color-meadow-300)]"
      : "text-[var(--color-cream-50)]";

  return (
    <motion.div
      initial={{ opacity: 0, y: 14, filter: "blur(4px)" }}
      animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
      transition={{ duration: 0.6, delay: 0.05 * index, ease: [0.22, 1, 0.36, 1] }}
      className="glass tilt-card relative overflow-hidden rounded-[var(--radius-card)] p-5"
    >
      <span className="eyebrow">{label}</span>
      <div className={`mt-2 font-display text-[2.4rem] font-light leading-none ${valueColor}`}>
        {value ?? "—"}
      </div>
      {hint ? (
        <div className="mt-2 text-[0.78rem] text-[var(--color-sage-200)]">{hint}</div>
      ) : null}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-x-0 -bottom-1 h-px bg-gradient-to-r from-transparent via-[var(--color-ember-500)] to-transparent opacity-30"
      />
    </motion.div>
  );
}
