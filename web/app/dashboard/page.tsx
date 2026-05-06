"use client";

import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, imageUrl } from "@/lib/api";
import type { Detection } from "@/lib/types";
import { formatNumber, formatPct, localTime } from "@/lib/format";
import { StatCard } from "@/components/StatCard";
import { Filters, EMPTY_FILTERS, type FilterValues } from "@/components/Filters";
import { DetectionCard } from "@/components/DetectionCard";
import { Modal } from "@/components/Modal";

const PER_PAGE = 24;

export default function DashboardPage() {
  const [filters, setFilters] = useState<FilterValues>(EMPTY_FILTERS);
  const [limit, setLimit] = useState(PER_PAGE);
  const [viewing, setViewing] = useState<Detection | null>(null);
  const [pendingDelete, setPendingDelete] = useState<Detection | null>(null);
  const qc = useQueryClient();

  const stats = useQuery({ queryKey: ["stats"], queryFn: api.stats });
  const system = useQuery({ queryKey: ["system"], queryFn: api.systemStatus });
  const speciesQ = useQuery({ queryKey: ["species"], queryFn: api.species });

  const detectionsParams = useMemo(
    () => ({
      species_id: filters.species_id || undefined,
      since: filters.since ? `${filters.since}T00:00:00` : undefined,
      until: filters.until ? `${filters.until}T23:59:59` : undefined,
      min_confidence: filters.min_confidence || undefined,
      limit,
    }),
    [filters, limit]
  );

  const detectionsQ = useQuery({
    queryKey: ["detections", detectionsParams, filters.reviewed],
    queryFn: async () => {
      const all = await api.detections(detectionsParams);
      // Backend doesn't filter by review status — apply client-side
      if (!filters.reviewed) return all;
      return all.filter((d) => {
        if (filters.reviewed === "pending") return !d.reviewed;
        if (filters.reviewed === "reviewed") return d.reviewed && !d.is_false_positive;
        if (filters.reviewed === "false_positive") return d.is_false_positive;
        return true;
      });
    },
  });

  const del = useMutation({
    mutationFn: (id: number) => api.deleteDetection(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["detections"] });
      qc.invalidateQueries({ queryKey: ["stats"] });
      setPendingDelete(null);
    },
  });

  const detections = detectionsQ.data ?? [];
  const speciesList = speciesQ.data ?? [];
  const visibleSpecies = useMemo(() => {
    // Only include species we've actually detected, ordered by name
    const seen = new Set<number>();
    detections.forEach((d) => d.species_id && seen.add(d.species_id));
    if (seen.size === 0) return speciesList.slice().sort((a, b) => a.common_name.localeCompare(b.common_name));
    return speciesList.slice().sort((a, b) => a.common_name.localeCompare(b.common_name));
  }, [speciesList, detections]);

  return (
    <>
      {/* Hero */}
      <section className="rise mb-8 flex flex-col gap-3 pt-2">
        <h1 className="font-display text-[2.6rem] leading-[1.05] tracking-tight text-[var(--color-cream-50)] sm:text-[3.4rem]">
          An evening{" "}
          <span className="display-italic text-[var(--color-cream-100)]">
            with the birds
          </span>
        </h1>
        <p className="max-w-[60ch] text-[0.95rem] leading-relaxed text-[var(--color-sage-100)]">
          Twenty-four hours a day, a small camera watches the feeder. Every visitor is
          spotted, identified, and pressed gently into this log — annotated with the
          time, weather, and the model&rsquo;s confidence in its guess.
        </p>
      </section>

      {/* Stats */}
      <section className="grid grid-cols-2 gap-3 md:grid-cols-3 lg:grid-cols-5">
        <StatCard
          index={0}
          label="Total visits"
          value={stats.data ? formatNumber(stats.data.total_detections) : "—"}
          hint={stats.data?.most_common_species ?? undefined}
        />
        <StatCard
          index={1}
          label="Unique species"
          value={stats.data?.unique_species ?? "—"}
          hint="seen since launch"
          emphasis="meadow"
        />
        <StatCard
          index={2}
          label="Today"
          value={stats.data?.detections_today ?? "—"}
          hint="visits since midnight"
          emphasis="ember"
        />
        <StatCard
          index={3}
          label="Avg confidence"
          value={stats.data ? `${(stats.data.avg_confidence * 100).toFixed(1)}%` : "—"}
          hint="across all detections"
        />
        <StatCard
          index={4}
          label="Disk"
          value={system.data ? `${system.data.disk_usage_mb.toFixed(1)} MB` : "—"}
          hint="of saved frames"
        />
      </section>

      <div className="my-8 h-px bg-gradient-to-r from-transparent via-[var(--color-moss-700)] to-transparent" />

      {/* Filters */}
      <Filters values={filters} onChange={(f) => { setFilters(f); setLimit(PER_PAGE); }} species={visibleSpecies} />

      {/* Gallery */}
      <section>
        <header className="mb-4 flex items-baseline justify-between">
          <h2 className="font-display text-[1.4rem] italic text-[var(--color-cream-50)]">
            Recent visitors
          </h2>
          <span className="eyebrow">
            {detectionsQ.isPending
              ? "loading…"
              : detections.length > 0
              ? `${detections.length} shown`
              : "0 in range"}
          </span>
        </header>

        {detectionsQ.isError ? (
          <EmptyState title="The log is silent." subtitle="Couldn't reach the API. The pipeline may be offline." />
        ) : detectionsQ.isPending ? (
          <SkeletonGrid />
        ) : detections.length === 0 ? (
          <EmptyState
            title="No visitors yet."
            subtitle="Try widening your filters, or wait for the camera to spot something."
          />
        ) : (
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {detections.map((d, i) => (
              <DetectionCard
                key={d.id}
                detection={d}
                index={i}
                onView={(x) => setViewing(x)}
                onDelete={(x) => setPendingDelete(x)}
              />
            ))}
          </div>
        )}

        {detections.length === limit ? (
          <div className="mt-8 flex justify-center">
            <button
              type="button"
              className="btn-quiet"
              onClick={() => setLimit((l) => l + PER_PAGE)}
            >
              load more →
            </button>
          </div>
        ) : null}
      </section>

      {/* Frame modal */}
      <Modal open={!!viewing} onClose={() => setViewing(null)}>
        {viewing ? (
          <div className="overflow-hidden rounded-[var(--radius-card)] border border-[var(--color-moss-700)] bg-[var(--color-ink-850)]">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={imageUrl.annotated(viewing.id)}
              alt={viewing.species_name ?? "frame"}
              className="block max-h-[78vh] w-full object-contain bg-black"
            />
            <div className="flex flex-wrap items-baseline justify-between gap-3 px-5 py-4">
              <div>
                <h3 className="display-italic text-[1.4rem] text-[var(--color-cream-50)]">
                  {viewing.corrected_species_name ?? viewing.species_name ?? "Unknown"}
                </h3>
                <div className="mt-1 text-[0.78rem] text-[var(--color-sage-200)]">
                  {localTime(viewing.timestamp)} · #{viewing.id} ·{" "}
                  {viewing.detection_model ?? "—"}{" "}
                  {viewing.classifier_model ? `→ ${viewing.classifier_model}` : ""}
                </div>
              </div>
              <span className="font-mono text-[var(--color-ember-400)]">
                {formatPct(viewing.confidence)}
              </span>
            </div>
          </div>
        ) : null}
      </Modal>

      {/* Delete confirm */}
      <Modal open={!!pendingDelete} onClose={() => setPendingDelete(null)} width="max-w-md">
        {pendingDelete ? (
          <div className="rounded-[var(--radius-card)] border border-[color-mix(in_oklab,var(--color-rust-500)_40%,var(--color-moss-700))] bg-[var(--color-ink-850)] p-6 text-center">
            <h3 className="font-display text-[1.6rem] italic text-[var(--color-rust-500)]">
              Discard this entry?
            </h3>
            <p className="mt-2 text-[0.9rem] text-[var(--color-sage-100)]">
              The image and database row will be permanently deleted from disk.
            </p>
            <div className="mt-5 flex items-center justify-center gap-3">
              <button
                type="button"
                className="btn-quiet"
                onClick={() => setPendingDelete(null)}
              >
                Keep it
              </button>
              <button
                type="button"
                className="btn-ember"
                style={{
                  color: "var(--color-rust-500)",
                  borderColor: "var(--color-rust-500)",
                  background: "rgba(201, 122, 92, 0.12)",
                }}
                onClick={() => del.mutate(pendingDelete.id)}
                disabled={del.isPending}
              >
                {del.isPending ? "removing…" : "yes, discard"}
              </button>
            </div>
          </div>
        ) : null}
      </Modal>
    </>
  );
}

function SkeletonGrid() {
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
      {Array.from({ length: 8 }).map((_, i) => (
        <div
          key={i}
          className="glass animate-pulse rounded-[var(--radius-card)]"
          style={{ aspectRatio: "0.78", animationDelay: `${i * 60}ms` }}
        />
      ))}
    </div>
  );
}

function EmptyState({ title, subtitle }: { title: string; subtitle: string }) {
  return (
    <div className="glass rounded-[var(--radius-card)] py-16 text-center">
      <div className="font-display text-[1.6rem] italic text-[var(--color-cream-50)]">{title}</div>
      <div className="mx-auto mt-2 max-w-md text-[0.9rem] text-[var(--color-sage-200)]">
        {subtitle}
      </div>
    </div>
  );
}
