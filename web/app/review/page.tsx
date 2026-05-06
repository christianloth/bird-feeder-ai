"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, imageUrl } from "@/lib/api";
import type { Detection, Species } from "@/lib/types";
import { confidenceTier, formatPct, localTime } from "@/lib/format";

export default function ReviewPage() {
  const qc = useQueryClient();
  const [correctOpen, setCorrectOpen] = useState(false);
  const [search, setSearch] = useState("");
  const searchRef = useRef<HTMLInputElement | null>(null);

  // Highest-confidence pending detection
  const nextQ = useQuery({
    queryKey: ["review-next"],
    queryFn: async (): Promise<Detection | null> => {
      // Sort by confidence desc on the server
      const list = await api.detections({
        limit: 50,
        reviewed: "pending",
      } as never);
      // Backend doesn't filter by reviewed — do it here
      const pending = list.filter((d) => !d.reviewed);
      if (pending.length === 0) return null;
      return pending.slice().sort((a, b) => b.confidence - a.confidence)[0];
    },
  });

  const statsQ = useQuery({ queryKey: ["stats"], queryFn: api.stats });
  const speciesQ = useQuery({ queryKey: ["species"], queryFn: () => api.species() });

  // Pending count: cheap-ish — fetch a handful of pending and use length, but
  // for accuracy we approximate with detections-since query. The simplest:
  // count pending by fetching limit=500 reviewed=pending client filter.
  const pendingCountQ = useQuery({
    queryKey: ["pending-count"],
    queryFn: async () => {
      const list = await api.detections({ limit: 500 });
      return list.filter((d) => !d.reviewed).length;
    },
  });

  const review = useMutation({
    mutationFn: (vars: {
      id: number;
      is_false_positive?: boolean;
      corrected_species_id?: number | null;
    }) =>
      api.review(vars.id, {
        is_false_positive: vars.is_false_positive,
        corrected_species_id: vars.corrected_species_id,
      }),
    onSuccess: () => {
      setCorrectOpen(false);
      setSearch("");
      qc.invalidateQueries({ queryKey: ["review-next"] });
      qc.invalidateQueries({ queryKey: ["pending-count"] });
      qc.invalidateQueries({ queryKey: ["stats"] });
    },
  });

  const detection = nextQ.data;

  const filteredSpecies = useMemo(() => {
    const list = speciesQ.data ?? [];
    if (!search) return list.slice(0, 80);
    const needle = search.toLowerCase();
    return list
      .filter(
        (sp) =>
          sp.common_name.toLowerCase().includes(needle) ||
          sp.scientific_name.toLowerCase().includes(needle)
      )
      .slice(0, 80);
  }, [speciesQ.data, search]);

  // Keyboard shortcuts
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName;
      const isInput = tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";
      if (e.key === "Escape") {
        setCorrectOpen(false);
        if (isInput) (e.target as HTMLElement).blur();
        return;
      }
      if (isInput) return;
      if (!detection) return;
      if (e.key === "a" || e.key === "A" || e.key === "Enter") {
        e.preventDefault();
        review.mutate({ id: detection.id, is_false_positive: false });
      } else if (e.key === "x" || e.key === "X") {
        e.preventDefault();
        review.mutate({ id: detection.id, is_false_positive: true });
      } else if (e.key === "c" || e.key === "C") {
        e.preventDefault();
        setCorrectOpen((o) => !o);
        setTimeout(() => searchRef.current?.focus(), 50);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [detection, review]);

  const total = statsQ.data?.total_detections ?? 0;
  const pending = pendingCountQ.data ?? 0;
  const reviewed = Math.max(total - pending, 0);
  const progress = total > 0 ? (reviewed / total) * 100 : 0;

  return (
    <div className="mx-auto max-w-3xl">
      <section className="rise mb-6 pt-2">
        <span className="eyebrow">Curation desk</span>
        <h1 className="mt-2 font-display text-[2.2rem] leading-tight text-[var(--color-cream-50)] sm:text-[2.8rem]">
          Sift through the day&rsquo;s sightings.{" "}
          <span className="display-italic text-[var(--color-ember-400)]">Confirm or correct.</span>
        </h1>
      </section>

      {/* progress */}
      <section className="glass mb-6 rounded-[var(--radius-card)] p-5">
        <div className="mb-3 flex items-baseline justify-between">
          <span className="eyebrow">Reviewed</span>
          <span className="font-mono text-[0.78rem] text-[var(--color-sage-100)]">
            {reviewed} / {total}{" "}
            <span className="text-[var(--color-ember-400)]">· {pending} pending</span>
          </span>
        </div>
        <div className="relative h-2 overflow-hidden rounded-full bg-[var(--color-ink-700)]">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${progress}%` }}
            transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
            className="absolute inset-y-0 left-0 rounded-full bg-gradient-to-r from-[var(--color-meadow-500)] via-[var(--color-meadow-300)] to-[var(--color-ember-400)]"
          />
        </div>

        <div className="mt-4 hidden flex-wrap items-center gap-3 text-[0.7rem] text-[var(--color-sage-200)] sm:flex">
          <Shortcut keyLabel="A" desc="confirm" />
          <Shortcut keyLabel="X" desc="false positive" />
          <Shortcut keyLabel="C" desc="correct species" />
          <Shortcut keyLabel="Esc" desc="cancel" />
        </div>
      </section>

      {/* Card */}
      <AnimatePresence mode="wait">
        {nextQ.isPending ? (
          <SkeletonCard key="loading" />
        ) : !detection ? (
          <DoneCard key="done" />
        ) : (
          <motion.div
            key={detection.id}
            initial={{ opacity: 0, y: 18, filter: "blur(6px)" }}
            animate={{ opacity: 1, y: 0, filter: "blur(0)" }}
            exit={{ opacity: 0, y: -16, filter: "blur(4px)" }}
            transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
            className="glass overflow-hidden rounded-[var(--radius-card)]"
          >
            <div className="relative bg-black">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={imageUrl.crop(detection.id)}
                alt={detection.species_name ?? "detection"}
                className="block max-h-[55vh] w-full object-contain"
              />
              <div className="pointer-events-none absolute inset-x-0 bottom-0 h-1/3 bg-gradient-to-t from-[rgba(7,9,10,0.9)] to-transparent" />
              <span className="absolute right-3 top-3 rounded-full border border-[var(--color-moss-700)] bg-[rgba(7,9,10,0.6)] px-3 py-1 font-mono text-[0.72rem] text-[var(--color-sage-100)] backdrop-blur">
                #{detection.id}
              </span>
            </div>

            <div className="p-6">
              <div className="flex items-baseline justify-between gap-4">
                <h2 className="display-italic text-[1.8rem] leading-tight text-[var(--color-cream-50)]">
                  {detection.species_name ?? "Unknown"}
                </h2>
                <ConfidenceBadge value={detection.confidence} />
              </div>

              <div className="mt-2 text-[0.78rem] text-[var(--color-sage-200)]">
                {localTime(detection.timestamp)} ·{" "}
                {detection.detection_model ?? "—"}
                {detection.classifier_model ? ` → ${detection.classifier_model}` : ""}{" "}
                · {detection.source ?? "unknown source"}
              </div>

              <div className="mt-6 grid grid-cols-1 gap-2 sm:grid-cols-3">
                <button
                  type="button"
                  className="btn-quiet justify-center !border-[color-mix(in_oklab,var(--color-meadow-500)_45%,transparent)] !text-[var(--color-meadow-300)] hover:!bg-[rgba(127,169,122,0.08)]"
                  onClick={() => review.mutate({ id: detection.id, is_false_positive: false })}
                  disabled={review.isPending}
                >
                  <span className="mr-2 inline-block rounded border border-current px-1.5 py-px font-mono text-[0.65rem] opacity-70">
                    A
                  </span>
                  Confirm
                </button>
                <button
                  type="button"
                  className="btn-quiet justify-center !border-[color-mix(in_oklab,var(--color-rust-500)_45%,transparent)] !text-[var(--color-rust-500)] hover:!bg-[rgba(201,122,92,0.08)]"
                  onClick={() => review.mutate({ id: detection.id, is_false_positive: true })}
                  disabled={review.isPending}
                >
                  <span className="mr-2 inline-block rounded border border-current px-1.5 py-px font-mono text-[0.65rem] opacity-70">
                    X
                  </span>
                  False positive
                </button>
                <button
                  type="button"
                  className="btn-quiet justify-center !border-[color-mix(in_oklab,var(--color-ember-500)_45%,transparent)] !text-[var(--color-ember-400)] hover:!bg-[rgba(224,169,109,0.08)]"
                  onClick={() => {
                    setCorrectOpen((o) => !o);
                    setTimeout(() => searchRef.current?.focus(), 50);
                  }}
                  disabled={review.isPending}
                >
                  <span className="mr-2 inline-block rounded border border-current px-1.5 py-px font-mono text-[0.65rem] opacity-70">
                    C
                  </span>
                  Correct species
                </button>
              </div>

              <AnimatePresence>
                {correctOpen ? (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    exit={{ opacity: 0, height: 0 }}
                    transition={{ duration: 0.2 }}
                    className="mt-4 overflow-hidden"
                  >
                    <div className="rounded-xl border border-[var(--color-moss-700)] bg-[var(--color-ink-900)] p-3">
                      <input
                        ref={searchRef}
                        type="text"
                        placeholder="Search 1,000+ species…"
                        value={search}
                        onChange={(e) => setSearch(e.target.value)}
                        className="input-quiet"
                      />
                      <div className="mt-2 max-h-56 overflow-y-auto">
                        {filteredSpecies.map((sp) => (
                          <SpeciesRow
                            key={sp.id}
                            species={sp}
                            onPick={() =>
                              review.mutate({
                                id: detection.id,
                                is_false_positive: false,
                                corrected_species_id: sp.id,
                              })
                            }
                          />
                        ))}
                        {filteredSpecies.length === 0 ? (
                          <div className="py-6 text-center text-[0.8rem] text-[var(--color-moss-300)]">
                            no matches
                          </div>
                        ) : null}
                      </div>
                    </div>
                  </motion.div>
                ) : null}
              </AnimatePresence>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function ConfidenceBadge({ value }: { value: number }) {
  const tier = confidenceTier(value);
  const color =
    tier === "high"
      ? "var(--color-meadow-300)"
      : tier === "medium"
      ? "var(--color-harvest-500)"
      : "var(--color-rust-500)";
  return (
    <span
      className="inline-flex items-baseline gap-2 rounded-full border px-3 py-1 font-mono text-[0.78rem]"
      style={{
        color,
        borderColor: `color-mix(in oklab, ${color} 40%, transparent)`,
        background: `color-mix(in oklab, ${color} 8%, transparent)`,
      }}
    >
      {formatPct(value, 1)}
      <span className="text-[0.6rem] uppercase tracking-[0.18em] opacity-70">{tier}</span>
    </span>
  );
}

function Shortcut({ keyLabel, desc }: { keyLabel: string; desc: string }) {
  return (
    <span className="inline-flex items-center gap-2">
      <kbd className="rounded border border-[var(--color-moss-700)] bg-[var(--color-ink-900)] px-2 py-0.5 font-mono text-[0.7rem] text-[var(--color-cream-100)]">
        {keyLabel}
      </kbd>
      <span>{desc}</span>
    </span>
  );
}

function SpeciesRow({ species, onPick }: { species: Species; onPick: () => void }) {
  return (
    <button
      type="button"
      onClick={onPick}
      className="flex w-full items-baseline justify-between rounded-md px-3 py-2 text-left text-[0.85rem] transition hover:bg-[var(--color-ink-700)]"
    >
      <span className="text-[var(--color-cream-100)]">{species.common_name}</span>
      <span className="font-display italic text-[0.78rem] text-[var(--color-sage-200)]">
        {species.scientific_name}
      </span>
    </button>
  );
}

function DoneCard() {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.97 }}
      animate={{ opacity: 1, scale: 1 }}
      className="glass rounded-[var(--radius-card)] py-20 text-center"
    >
      <div className="mx-auto mb-4 grid h-14 w-14 place-items-center rounded-full border border-[var(--color-meadow-500)] bg-[rgba(127,169,122,0.08)]">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-[var(--color-meadow-300)]">
          <path d="M5 13l4 4L19 7" />
        </svg>
      </div>
      <h2 className="font-display text-[2rem] italic text-[var(--color-cream-50)]">
        All caught up.
      </h2>
      <p className="mt-2 text-[0.95rem] text-[var(--color-sage-100)]">
        No detections waiting for review.
      </p>
    </motion.div>
  );
}

function SkeletonCard() {
  return (
    <div className="glass animate-pulse rounded-[var(--radius-card)]" style={{ aspectRatio: "0.85" }} />
  );
}
