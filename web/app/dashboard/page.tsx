"use client";

import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, imageUrl } from "@/lib/api";
import type { Detection } from "@/lib/types";
import { formatNumber, formatPct, localTime } from "@/lib/format";
import { StatCard } from "@/components/StatCard";
import { Filters, EMPTY_FILTERS, type FilterValues } from "@/components/Filters";
import { DetectionCard } from "@/components/DetectionCard";
import { Modal } from "@/components/Modal";
import { Pagination } from "@/components/Pagination";

const PER_PAGE = 24;

export default function DashboardPage() {
  return (
    <Suspense>
      <DashboardInner />
    </Suspense>
  );
}

function DashboardInner() {
  const router = useRouter();
  const searchParams = useSearchParams();

  const viewingId = useMemo<number | null>(() => {
    const raw = searchParams.get("d");
    if (!raw) return null;
    const n = Number(raw);
    return Number.isFinite(n) && n > 0 ? n : null;
  }, [searchParams]);

  // URL is a best-effort deep link, not the source of truth for "is the
  // modal open." Some webviews (Telegram's iOS in-app browser in particular)
  // don't propagate History API updates back to useSearchParams, so we have
  // to drive the modal from local state and update the URL on the side.
  const syncUrl = (id: number | null) => {
    const sp = new URLSearchParams(searchParams.toString());
    if (id === null) sp.delete("d");
    else sp.set("d", String(id));
    const qs = sp.toString();
    const path = qs ? `/dashboard/?${qs}` : "/dashboard/";
    if (id === null) router.replace(path, { scroll: false });
    else router.push(path, { scroll: false });
  };

  const [filters, setFilters] = useState<FilterValues>(EMPTY_FILTERS);
  const [page, setPage] = useState(1);
  const [pendingDelete, setPendingDelete] = useState<Detection | null>(null);
  const [viewing, setViewing] = useState<Detection | null>(null);
  // Tracks ids the user has explicitly dismissed so a later data refetch
  // (or a stuck URL in webviews) can't re-open the modal via the URL→state
  // effect below. Held in a ref so mutating it doesn't trigger renders.
  const dismissedIdsRef = useRef<Set<number>>(new Set());
  // Tracks the last viewingId we synced from the URL. When the URL changes
  // (e.g. user pastes a fresh link, or browser back), we clear dismissals
  // so a previously-dismissed id can be deep-linked back into in this tab.
  const lastViewingIdRef = useRef<number | null | undefined>(undefined);
  const qc = useQueryClient();

  const stats = useQuery({ queryKey: ["stats"], queryFn: api.stats });
  const system = useQuery({ queryKey: ["system"], queryFn: api.systemStatus });
  // Filter dropdown only lists species that have at least one detection.
  const speciesQ = useQuery({
    queryKey: ["species", "with-detections"],
    queryFn: () => api.species({ withDetections: true }),
  });

  const filterParams = useMemo(
    () => ({
      species_id: filters.species_id || undefined,
      since: filters.since ? `${filters.since}T00:00:00` : undefined,
      until: filters.until ? `${filters.until}T23:59:59` : undefined,
      min_confidence: filters.min_confidence || undefined,
      reviewed: filters.reviewed || undefined,
    }),
    [filters]
  );

  const countQ = useQuery({
    queryKey: ["detections-count", filterParams],
    queryFn: () => api.detectionsCount(filterParams),
  });

  const total = countQ.data ?? 0;
  const totalPages = Math.max(1, Math.ceil(total / PER_PAGE));

  // Clamp page if filter shrinks the result set under the current page
  useEffect(() => {
    if (page > totalPages) setPage(totalPages);
  }, [page, totalPages]);

  const detectionsQ = useQuery({
    queryKey: ["detections", filterParams, page],
    queryFn: () =>
      api.detections({
        ...filterParams,
        skip: (page - 1) * PER_PAGE,
        limit: PER_PAGE,
      }),
  });

  const del = useMutation({
    mutationFn: (id: number) => api.deleteDetection(id),
    onSuccess: (_data, deletedId) => {
      qc.invalidateQueries({ queryKey: ["detections"] });
      qc.invalidateQueries({ queryKey: ["detections-count"] });
      qc.invalidateQueries({ queryKey: ["stats"] });
      setPendingDelete(null);
      if (viewing?.id === deletedId) closeViewer();
    },
  });

  // Memoize so the deps array of the URL→state effect doesn't see a fresh
  // `[]` reference on every render while data is loading.
  const detections = useMemo<Detection[]>(
    () => detectionsQ.data ?? [],
    [detectionsQ.data]
  );
  const speciesList = speciesQ.data ?? [];

  // Fall back to a single-detection fetch when the deep-linked id isn't on
  // the current page (e.g. a shared link to an older detection).
  const singleQ = useQuery({
    queryKey: ["detection", viewingId],
    queryFn: () => api.detection(viewingId!),
    enabled: viewingId !== null && !detections.some((d) => d.id === viewingId),
  });

  // URL → state. Runs on initial deep link, browser back/forward, and any
  // detections / singleQ.data refetch. Skips ids the user has explicitly
  // dismissed so a stale URL (Telegram in-app browser swallows History API
  // updates) can't re-open the modal after a refetch. URL transitions
  // clear the dismissal set so a previously-closed id can be deep-linked
  // back into within the same tab.
  useEffect(() => {
    if (lastViewingIdRef.current !== viewingId) {
      dismissedIdsRef.current.clear();
      lastViewingIdRef.current = viewingId;
    }
    if (viewingId === null) {
      setViewing(null);
      return;
    }
    if (dismissedIdsRef.current.has(viewingId)) return;
    const found =
      detections.find((d) => d.id === viewingId) ?? singleQ.data ?? null;
    if (found) setViewing(found);
  }, [viewingId, detections, singleQ.data]);

  const openViewer = (d: Detection) => {
    // Switching from another view: dismiss the prior id so a stuck URL
    // can't pull it back through the effect above.
    if (viewing && viewing.id !== d.id) {
      dismissedIdsRef.current.add(viewing.id);
    }
    dismissedIdsRef.current.delete(d.id);
    setViewing(d);
    syncUrl(d.id);
  };
  const closeViewer = () => {
    if (viewing) dismissedIdsRef.current.add(viewing.id);
    setViewing(null);
    syncUrl(null);
  };

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
      <Filters
        values={filters}
        onChange={(f) => {
          setFilters(f);
          setPage(1);
        }}
        species={speciesList}
      />

      {/* Gallery */}
      <section>
        <header className="mb-4 flex items-baseline justify-between">
          <h2 className="font-display text-[1.4rem] italic text-[var(--color-cream-50)]">
            Recent visitors
          </h2>
          <span className="eyebrow">
            {detectionsQ.isPending
              ? "loading…"
              : total === 0
              ? "0 in range"
              : `${(page - 1) * PER_PAGE + 1}–${Math.min(page * PER_PAGE, total)} of ${formatNumber(total)}`}
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
                onView={(x) => openViewer(x)}
                onDelete={(x) => setPendingDelete(x)}
              />
            ))}
          </div>
        )}

        <Pagination
          page={page}
          totalPages={totalPages}
          onChange={(next) => {
            setPage(next);
            // Smooth scroll up so the user doesn't end up looking at the pagination after clicking
            window.scrollTo({ top: 0, behavior: "smooth" });
          }}
        />
      </section>

      {/* Frame modal */}
      <Modal open={!!viewing} onClose={closeViewer} width="max-w-3xl">
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
                {viewing.detection_model ? ` · ${viewing.detection_model}` : ""}
                {viewing.classifier_model ? ` → ${viewing.classifier_model}` : ""}
              </div>
              <div className="mt-3 inline-flex items-center gap-2 rounded-full border border-[color-mix(in_oklab,var(--color-ember-500)_45%,transparent)] bg-[rgba(224,169,109,0.1)] px-4 py-1.5 font-mono text-[0.8rem] text-[var(--color-ember-400)]">
                {formatPct(viewing.confidence)}
              </div>
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
