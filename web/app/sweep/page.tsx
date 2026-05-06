"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { AnimatePresence, motion } from "motion/react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, imageUrl } from "@/lib/api";
import type { Detection, IgnoreRegion } from "@/lib/types";
import { formatNumber, formatPct, localTime, timeAgo } from "@/lib/format";
import { Modal } from "@/components/Modal";
import { Pagination } from "@/components/Pagination";

const PER_PAGE = 30;

// The rotated camera frame (camera.rotation_degrees = -90 in config).
// Used purely for the schematic preview of where each region sits.
const FRAME_W = 1080;
const FRAME_H = 1920;

export default function SweepPage() {
  const qc = useQueryClient();
  const features = useQuery({
    queryKey: ["features"],
    queryFn: api.features,
    staleTime: 60_000,
  });
  const regionsQ = useQuery({
    queryKey: ["ignore-regions"],
    queryFn: api.ignoreRegions,
  });

  const regions = regionsQ.data ?? [];
  const [activeIdx, setActiveIdx] = useState(0);
  const region: IgnoreRegion | undefined = regions[activeIdx];

  const [page, setPage] = useState(1);
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [viewing, setViewing] = useState<Detection | null>(null);
  const [confirmingBulk, setConfirmingBulk] = useState<"selected" | "all" | null>(null);

  const filterParams = useMemo(() => {
    if (!region) return null;
    return {
      region_x1: region.x1,
      region_y1: region.y1,
      region_x2: region.x2,
      region_y2: region.y2,
      region_overlap: region.overlap_threshold,
    };
  }, [region]);

  const countQ = useQuery({
    queryKey: ["sweep-count", filterParams],
    queryFn: () => api.detectionsCount(filterParams ?? {}),
    enabled: !!filterParams,
  });

  const total = countQ.data ?? 0;
  const totalPages = Math.max(1, Math.ceil(total / PER_PAGE));
  useEffect(() => {
    if (page > totalPages) setPage(totalPages);
  }, [page, totalPages]);

  const detectionsQ = useQuery({
    queryKey: ["sweep-detections", filterParams, page],
    queryFn: () =>
      api.detections({
        ...(filterParams ?? {}),
        skip: (page - 1) * PER_PAGE,
        limit: PER_PAGE,
      }),
    enabled: !!filterParams,
  });

  const detections = detectionsQ.data ?? [];

  // Drop selections that vanished (e.g. after a delete or page change).
  useEffect(() => {
    if (selected.size === 0) return;
    const visibleIds = new Set(detections.map((d) => d.id));
    const next = new Set<number>();
    selected.forEach((id) => {
      if (visibleIds.has(id)) next.add(id);
    });
    if (next.size !== selected.size) setSelected(next);
    // Only when the page or detections list itself changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [page, detections]);

  // Reset selection on region change.
  useEffect(() => {
    setSelected(new Set());
    setPage(1);
  }, [activeIdx]);

  const bulkDel = useMutation({
    mutationFn: (ids: number[]) => api.bulkDelete(ids),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["sweep-count"] });
      qc.invalidateQueries({ queryKey: ["sweep-detections"] });
      qc.invalidateQueries({ queryKey: ["stats"] });
      qc.invalidateQueries({ queryKey: ["detections"] });
      qc.invalidateQueries({ queryKey: ["detections-count"] });
      setSelected(new Set());
      setConfirmingBulk(null);
    },
  });

  const allOnPageSelected =
    detections.length > 0 && detections.every((d) => selected.has(d.id));

  const togglePage = () => {
    if (allOnPageSelected) {
      const next = new Set(selected);
      detections.forEach((d) => next.delete(d.id));
      setSelected(next);
    } else {
      const next = new Set(selected);
      detections.forEach((d) => next.add(d.id));
      setSelected(next);
    }
  };

  const toggleOne = (id: number) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  // ── Rendering branches ──────────────────────────────────────────────

  if (features.isPending || regionsQ.isPending) {
    return <PageShell><LoadingShell /></PageShell>;
  }

  if (!features.data?.sweep) {
    return (
      <PageShell>
        <Notice
          title="Sweep is disabled."
          body={
            <>
              Enable it by setting{" "}
              <code className="rounded bg-[var(--color-ink-700)] px-1.5 py-0.5 font-mono text-[0.78rem] text-[var(--color-cream-100)]">
                features.sweep: true
              </code>{" "}
              in <span className="font-mono">config/config.yaml</span> and
              restarting the backend.
            </>
          }
        />
      </PageShell>
    );
  }

  if (regions.length === 0) {
    return (
      <PageShell>
        <Notice
          title="No ignore regions configured."
          body={
            <>
              Add one or more rectangles to{" "}
              <code className="rounded bg-[var(--color-ink-700)] px-1.5 py-0.5 font-mono text-[0.78rem] text-[var(--color-cream-100)]">
                bird_detection.ignore_regions
              </code>{" "}
              in <span className="font-mono">config/config.yaml</span> to start
              sweeping. Each region is{" "}
              <span className="font-mono">[x1, y1, x2, y2]</span> in rotated
              frame coordinates ({FRAME_W}×{FRAME_H}).
            </>
          }
        />
      </PageShell>
    );
  }

  return (
    <PageShell>
      {/* Hero */}
      <section className="rise mb-6 flex flex-col gap-3 pt-2">
        <span className="eyebrow">Decoy sweep</span>
        <h1 className="font-display text-[2.4rem] leading-[1.05] tracking-tight text-[var(--color-cream-50)] sm:text-[3rem]">
          Sweep the watched zones.{" "}
          <span className="display-italic text-[var(--color-ember-400)]">
            Cull the chaff.
          </span>
        </h1>
        <p className="max-w-[60ch] text-[0.95rem] leading-relaxed text-[var(--color-sage-100)]">
          Detections that landed inside an ignored rectangle — almost always the
          fake bird stuck to the feeder. Skim the contact sheet, lasso the
          obvious ones, and discard them in one go.
        </p>
      </section>

      {/* Region picker */}
      <section className="glass mb-6 overflow-hidden rounded-[var(--radius-card)]">
        <div className="grid gap-0 sm:grid-cols-[1fr_220px]">
          <div className="p-5">
            <span className="eyebrow">Watched region</span>
            <div className="mt-3 flex flex-wrap gap-2">
              {regions.map((r, i) => {
                const w = r.x2 - r.x1;
                const h = r.y2 - r.y1;
                const active = i === activeIdx;
                return (
                  <button
                    key={`${r.x1}-${r.y1}-${r.x2}-${r.y2}`}
                    type="button"
                    onClick={() => setActiveIdx(i)}
                    data-active={active}
                    className="region-pill"
                  >
                    <span className="font-mono text-[0.74rem]">
                      {r.x1},{r.y1} → {r.x2},{r.y2}
                    </span>
                    <span className="text-[0.66rem] text-[var(--color-moss-300)]">
                      {w}×{h}
                    </span>
                  </button>
                );
              })}
            </div>

            {region ? (
              <div className="mt-4 flex flex-wrap items-baseline gap-x-5 gap-y-1 text-[0.78rem] text-[var(--color-sage-200)]">
                <span>
                  <span className="eyebrow mr-2">match ≥</span>
                  <span className="font-mono text-[var(--color-cream-100)]">
                    {Math.round(region.overlap_threshold * 100)}%
                  </span>
                </span>
                <span>
                  <span className="eyebrow mr-2">in zone</span>
                  <span className="font-mono text-[var(--color-ember-400)]">
                    {countQ.isPending ? "—" : formatNumber(total)}
                  </span>
                </span>
                {selected.size > 0 ? (
                  <span>
                    <span className="eyebrow mr-2">selected</span>
                    <span className="font-mono text-[var(--color-cream-100)]">
                      {selected.size}
                    </span>
                  </span>
                ) : null}
              </div>
            ) : null}
          </div>

          {region ? (
            <div className="border-l-0 border-t border-[var(--color-moss-700)] bg-[rgba(7,9,10,0.4)] p-4 sm:border-l sm:border-t-0">
              <span className="eyebrow block mb-2">Frame map</span>
              <FramePreview region={region} />
            </div>
          ) : null}
        </div>
      </section>

      {/* Action bar */}
      <section className="mb-4 flex flex-wrap items-center gap-3">
        <button
          type="button"
          className="btn-quiet"
          onClick={togglePage}
          disabled={detections.length === 0}
        >
          <CheckSquareIcon active={allOnPageSelected} />
          <span>{allOnPageSelected ? "Clear page" : "Select page"}</span>
        </button>
        {selected.size > 0 ? (
          <button
            type="button"
            className="btn-quiet"
            onClick={() => setSelected(new Set())}
          >
            Clear all ({selected.size})
          </button>
        ) : null}

        <div className="ml-auto flex items-center gap-3">
          <span className="eyebrow">
            {detectionsQ.isPending
              ? "loading…"
              : total === 0
              ? "0 in zone"
              : `${(page - 1) * PER_PAGE + 1}–${Math.min(
                  page * PER_PAGE,
                  total
                )} of ${formatNumber(total)}`}
          </span>
          {total > 0 ? (
            <button
              type="button"
              onClick={() => setConfirmingBulk("all")}
              className="discard-btn discard-btn--ghost"
              disabled={bulkDel.isPending}
            >
              <TrashIcon />
              <span>Discard all {formatNumber(total)}</span>
            </button>
          ) : null}
        </div>
      </section>

      {/* Grid */}
      {detectionsQ.isError ? (
        <Notice
          title="Couldn't reach the API."
          body="The pipeline or backend may be offline."
        />
      ) : detectionsQ.isPending ? (
        <SkeletonGrid />
      ) : detections.length === 0 ? (
        <Notice
          title="The zone is clear."
          body="Nothing in the database overlaps this region right now."
        />
      ) : (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6">
          {detections.map((d, i) => (
            <SweepCard
              key={d.id}
              detection={d}
              index={i}
              selected={selected.has(d.id)}
              onToggle={() => toggleOne(d.id)}
              onView={() => setViewing(d)}
            />
          ))}
        </div>
      )}

      <Pagination
        page={page}
        totalPages={totalPages}
        onChange={(next) => {
          setPage(next);
          window.scrollTo({ top: 0, behavior: "smooth" });
        }}
      />

      {/* Selected action bar (mobile-friendly sticky footer) */}
      <AnimatePresence>
        {selected.size > 0 ? (
          <motion.div
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 16 }}
            transition={{ duration: 0.25, ease: [0.22, 1, 0.36, 1] }}
            className="sweep-action-bar"
          >
            <div className="glass flex items-center gap-3 rounded-full px-4 py-2 shadow-[var(--shadow-glow)]">
              <span className="eyebrow whitespace-nowrap">
                {selected.size} selected
              </span>
              <button
                type="button"
                onClick={() => setSelected(new Set())}
                className="btn-quiet !min-h-0 !py-1"
              >
                Clear
              </button>
              <button
                type="button"
                onClick={() => setConfirmingBulk("selected")}
                className="discard-btn"
                disabled={bulkDel.isPending}
              >
                <TrashIcon />
                <span>Discard {selected.size}</span>
              </button>
            </div>
          </motion.div>
        ) : null}
      </AnimatePresence>

      {/* Frame preview modal */}
      <Modal open={!!viewing} onClose={() => setViewing(null)} width="max-w-3xl">
        {viewing ? (
          <div className="flex flex-col items-center gap-5">
            <div className="relative w-full">
              <div
                aria-hidden
                className="pointer-events-none absolute -inset-4 rounded-[28px] bg-gradient-to-br from-[rgba(224,169,109,0.14)] via-transparent to-[rgba(127,169,122,0.08)] blur-2xl"
              />
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={imageUrl.annotated(viewing.id)}
                alt={viewing.species_name ?? "frame"}
                className="relative mx-auto block max-h-[58vh] w-auto max-w-full rounded-[var(--radius-card)] border border-[var(--color-moss-700)] shadow-[0_30px_80px_-20px_rgba(0,0,0,0.7)]"
              />
            </div>
            <div className="glass w-full max-w-md rounded-[var(--radius-card)] px-5 py-4 text-center">
              <h3 className="display-italic text-[1.4rem] leading-tight text-[var(--color-cream-50)] sm:text-[1.6rem]">
                {viewing.corrected_species_name ?? viewing.species_name ?? "Unknown"}
              </h3>
              <div className="mt-1.5 text-[0.78rem] text-[var(--color-sage-200)]">
                {localTime(viewing.timestamp)} · #{viewing.id}
              </div>
              <div className="mt-3 inline-flex items-center gap-2 rounded-full border border-[color-mix(in_oklab,var(--color-ember-500)_45%,transparent)] bg-[rgba(224,169,109,0.1)] px-4 py-1.5 font-mono text-[0.8rem] text-[var(--color-ember-400)]">
                {formatPct(viewing.confidence)}
              </div>
            </div>
          </div>
        ) : null}
      </Modal>

      {/* Bulk-delete confirm */}
      <Modal open={!!confirmingBulk} onClose={() => setConfirmingBulk(null)} width="max-w-md">
        {confirmingBulk ? (
          <div className="rounded-[var(--radius-card)] border border-[color-mix(in_oklab,var(--color-rust-500)_40%,var(--color-moss-700))] bg-[var(--color-ink-850)] p-6 text-center">
            <h3 className="font-display text-[1.6rem] italic text-[var(--color-rust-500)]">
              Discard{" "}
              {confirmingBulk === "all"
                ? formatNumber(total)
                : formatNumber(selected.size)}{" "}
              {confirmingBulk === "all" ? "matching" : "selected"}?
            </h3>
            <p className="mt-2 text-[0.9rem] text-[var(--color-sage-100)]">
              Their database rows and image files will be permanently deleted
              from disk.
            </p>
            <div className="mt-5 flex items-center justify-center gap-3">
              <button
                type="button"
                className="btn-quiet"
                onClick={() => setConfirmingBulk(null)}
                disabled={bulkDel.isPending}
              >
                Keep them
              </button>
              <button
                type="button"
                className="btn-ember"
                style={{
                  color: "var(--color-rust-500)",
                  borderColor: "var(--color-rust-500)",
                  background: "rgba(201, 122, 92, 0.12)",
                }}
                onClick={async () => {
                  if (confirmingBulk === "selected") {
                    bulkDel.mutate(Array.from(selected));
                  } else {
                    // Delete all matching the region: page through everything,
                    // collecting IDs, then send one bulk-delete.
                    try {
                      const ids: number[] = [];
                      const chunk = 500;
                      for (let skip = 0; skip < total; skip += chunk) {
                        const batch = await api.detections({
                          ...(filterParams ?? {}),
                          skip,
                          limit: chunk,
                        });
                        batch.forEach((d) => ids.push(d.id));
                        if (batch.length < chunk) break;
                      }
                      bulkDel.mutate(ids);
                    } catch {
                      setConfirmingBulk(null);
                    }
                  }
                }}
                disabled={bulkDel.isPending}
              >
                {bulkDel.isPending ? "removing…" : "yes, discard"}
              </button>
            </div>
          </div>
        ) : null}
      </Modal>
    </PageShell>
  );
}

/* ── Components ──────────────────────────────────────────────────────── */

function PageShell({ children }: { children: React.ReactNode }) {
  return <div className="mx-auto max-w-[1400px] pb-32">{children}</div>;
}

function SweepCard({
  detection,
  selected,
  onToggle,
  onView,
  index,
}: {
  detection: Detection;
  selected: boolean;
  onToggle: () => void;
  onView: () => void;
  index: number;
}) {
  const speciesName =
    detection.corrected_species_name ?? detection.species_name ?? "Unknown";

  return (
    <motion.article
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: Math.min(index * 0.018, 0.3), ease: [0.22, 1, 0.36, 1] }}
      data-selected={selected}
      className="sweep-card group"
    >
      <button
        type="button"
        onClick={onToggle}
        aria-pressed={selected}
        aria-label={`${selected ? "Deselect" : "Select"} #${detection.id} · ${speciesName}`}
        className="relative block aspect-[3/4] w-full overflow-hidden bg-[var(--color-ink-900)]"
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={imageUrl.crop(detection.id)}
          alt={speciesName}
          loading="lazy"
          className="h-full w-full object-cover transition-transform duration-700 group-hover:scale-[1.04]"
        />
        <div
          aria-hidden
          className="pointer-events-none absolute inset-0 bg-gradient-to-t from-[rgba(7,9,10,0.95)] via-[rgba(7,9,10,0.15)] to-transparent"
        />
        {/* selection corner */}
        <span className="sweep-check" data-selected={selected}>
          {selected ? (
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
              <path d="M5 13l4 4L19 7" />
            </svg>
          ) : null}
        </span>

        <span className="absolute bottom-2 left-2 right-2 flex items-baseline justify-between gap-2 text-left">
          <span className="display-italic line-clamp-1 text-[0.95rem] text-[var(--color-cream-50)] drop-shadow-[0_2px_8px_rgba(0,0,0,0.7)]">
            {speciesName}
          </span>
          <span className="font-mono text-[0.7rem] text-[var(--color-ember-400)] drop-shadow">
            {formatPct(detection.confidence, 0)}
          </span>
        </span>
      </button>

      <div className="flex items-center justify-between px-2 py-1.5 text-[0.66rem] text-[var(--color-moss-300)]">
        <span className="font-mono">#{detection.id}</span>
        <span className="truncate" title={localTime(detection.timestamp)}>
          {timeAgo(detection.timestamp)}
        </span>
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onView();
          }}
          className="rounded p-1 text-[var(--color-sage-200)] transition hover:bg-[rgba(255,255,255,0.04)] hover:text-[var(--color-cream-100)]"
          title="View frame"
          aria-label="View full frame"
        >
          <EyeIcon />
        </button>
      </div>
    </motion.article>
  );
}

function FramePreview({ region }: { region: IgnoreRegion }) {
  // 1080×1920 portrait — render at fixed width with aspect preserved.
  const w = 110;
  const h = (w * FRAME_H) / FRAME_W;
  const sx = w / FRAME_W;
  const sy = h / FRAME_H;
  const rx = region.x1 * sx;
  const ry = region.y1 * sy;
  const rw = (region.x2 - region.x1) * sx;
  const rh = (region.y2 - region.y1) * sy;

  return (
    <div className="flex items-start gap-3">
      <svg
        width={w}
        height={h}
        viewBox={`0 0 ${w} ${h}`}
        className="shrink-0"
        aria-hidden
      >
        <rect
          x="0"
          y="0"
          width={w}
          height={h}
          rx="6"
          fill="rgba(20, 26, 22, 0.6)"
          stroke="var(--color-moss-500)"
          strokeWidth="0.7"
        />
        {/* gridlines */}
        {[0.25, 0.5, 0.75].map((t) => (
          <line
            key={`v${t}`}
            x1={w * t}
            x2={w * t}
            y1="0"
            y2={h}
            stroke="var(--color-moss-700)"
            strokeWidth="0.4"
            strokeDasharray="2 3"
          />
        ))}
        {[0.25, 0.5, 0.75].map((t) => (
          <line
            key={`h${t}`}
            x1="0"
            x2={w}
            y1={h * t}
            y2={h * t}
            stroke="var(--color-moss-700)"
            strokeWidth="0.4"
            strokeDasharray="2 3"
          />
        ))}
        {/* region */}
        <rect
          x={rx}
          y={ry}
          width={rw}
          height={rh}
          rx="1.5"
          fill="rgba(244, 207, 138, 0.18)"
          stroke="var(--color-ember-400)"
          strokeWidth="0.9"
        />
      </svg>
      <div className="flex flex-col gap-1 text-[0.66rem] leading-tight text-[var(--color-moss-300)]">
        <span>
          <span className="eyebrow">frame</span>
          <span className="ml-1.5 font-mono text-[var(--color-sage-100)]">
            {FRAME_W}×{FRAME_H}
          </span>
        </span>
        <span>
          <span className="eyebrow">zone</span>
          <span className="ml-1.5 font-mono text-[var(--color-ember-400)]">
            {region.x2 - region.x1}×{region.y2 - region.y1}
          </span>
        </span>
      </div>
    </div>
  );
}

function SkeletonGrid() {
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6">
      {Array.from({ length: 18 }).map((_, i) => (
        <div
          key={i}
          className="glass animate-pulse rounded-[10px]"
          style={{ aspectRatio: "0.75", animationDelay: `${i * 40}ms` }}
        />
      ))}
    </div>
  );
}

function LoadingShell() {
  return (
    <div className="glass mt-12 rounded-[var(--radius-card)] py-16 text-center">
      <div className="font-display text-[1.6rem] italic text-[var(--color-cream-50)]">
        Loading sweep…
      </div>
    </div>
  );
}

function Notice({
  title,
  body,
}: {
  title: string;
  body: React.ReactNode;
}) {
  return (
    <div className="glass mt-6 rounded-[var(--radius-card)] py-12 text-center">
      <div className="font-display text-[1.6rem] italic text-[var(--color-cream-50)]">
        {title}
      </div>
      <div className="mx-auto mt-2 max-w-md text-[0.9rem] text-[var(--color-sage-200)]">
        {body}
      </div>
      <div className="mt-5">
        <Link href="/dashboard" className="btn-quiet">
          ← back to dashboard
        </Link>
      </div>
    </div>
  );
}

function CheckSquareIcon({ active }: { active: boolean }) {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <rect x="3" y="3" width="18" height="18" rx="3" />
      {active ? <path d="M8 12l3 3 5-6" /> : null}
    </svg>
  );
}

function TrashIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M3 6h18" />
      <path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
      <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
    </svg>
  );
}

function EyeIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12z" />
      <circle cx="12" cy="12" r="3" />
    </svg>
  );
}
