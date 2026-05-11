"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, imageUrl } from "@/lib/api";
import type { IgnoreRegion } from "@/lib/types";
import { SnapshotCanvas, type DraftRect } from "./SnapshotCanvas";
import { RegionsList } from "./RegionsList";

export default function RegionsPage() {
  const qc = useQueryClient();
  const regionsQ = useQuery({
    queryKey: ["ignore-regions"],
    queryFn: api.ignoreRegions.list,
  });
  const settingsQ = useQuery({
    queryKey: ["ignore-regions", "settings"],
    queryFn: api.ignoreRegions.settings,
  });

  const regions = regionsQ.data ?? [];
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [drawMode, setDrawMode] = useState(false);
  const [refreshKey, setRefreshKey] = useState<number>(() => Date.now());
  const [snapshotMissing, setSnapshotMissing] = useState(false);
  const [imgSize, setImgSize] = useState<{ w: number; h: number } | null>(null);
  // Local optimistic copy for in-flight drag mutations — keeps the
  // rectangle perfectly stuck to the pointer instead of stuttering on
  // each network round-trip. Cleared once the server response arrives.
  const [optimistic, setOptimistic] = useState<Map<number, DraftRect>>(
    () => new Map()
  );

  // ── Mutations ────────────────────────────────────────────────────────
  const createRegion = useMutation({
    mutationFn: (rect: DraftRect) =>
      api.ignoreRegions.create({ ...rect, label: "" }),
    onSuccess: (created) => {
      qc.invalidateQueries({ queryKey: ["ignore-regions"] });
      setSelectedId(created.id);
    },
  });

  const updateRegion = useMutation({
    mutationFn: ({ id, ...patch }: { id: number } & Partial<IgnoreRegion>) =>
      api.ignoreRegions.update(id, patch),
    onSettled: (_data, _err, vars) => {
      qc.invalidateQueries({ queryKey: ["ignore-regions"] });
      // Clear the optimistic stash for this id once server state arrives.
      setOptimistic((prev) => {
        if (!prev.has(vars.id)) return prev;
        const next = new Map(prev);
        next.delete(vars.id);
        return next;
      });
    },
  });

  const deleteRegion = useMutation({
    mutationFn: (id: number) => api.ignoreRegions.remove(id),
    onSuccess: (_d, id) => {
      qc.invalidateQueries({ queryKey: ["ignore-regions"] });
      if (selectedId === id) setSelectedId(null);
    },
  });

  const updateSettings = useMutation({
    mutationFn: (overlap_threshold: number) =>
      api.ignoreRegions.updateSettings({ overlap_threshold }),
    onSuccess: (data) => {
      qc.setQueryData(["ignore-regions", "settings"], data);
    },
  });

  // ── Geometry patching during drag (optimistic, debounced PATCH) ──────
  const patchTimers = useRef<Map<number, number>>(new Map());

  const patchRegionGeometry = (id: number, rect: DraftRect) => {
    setOptimistic((prev) => {
      const next = new Map(prev);
      next.set(id, rect);
      return next;
    });
    const existing = patchTimers.current.get(id);
    if (existing) window.clearTimeout(existing);
    const handle = window.setTimeout(() => {
      patchTimers.current.delete(id);
      updateRegion.mutate({ id, ...rect });
    }, 80); // small debounce so we don't spam the API mid-drag
    patchTimers.current.set(id, handle);
  };

  // ── Threshold slider with debounced PATCH ────────────────────────────
  const [localThreshold, setLocalThreshold] = useState<number | null>(null);
  const thresholdSyncRef = useRef<number | null>(null);
  useEffect(() => {
    if (localThreshold === null && settingsQ.data) {
      setLocalThreshold(settingsQ.data.overlap_threshold);
    }
  }, [settingsQ.data, localThreshold]);

  const onThresholdChange = (v: number) => {
    setLocalThreshold(v);
    if (thresholdSyncRef.current !== null) {
      window.clearTimeout(thresholdSyncRef.current);
    }
    thresholdSyncRef.current = window.setTimeout(() => {
      thresholdSyncRef.current = null;
      updateSettings.mutate(v);
    }, 220);
  };

  // ── Apply optimistic geometry to the regions before rendering ───────
  const renderedRegions = useMemo<IgnoreRegion[]>(() => {
    if (optimistic.size === 0) return regions;
    return regions.map((r) => {
      const o = optimistic.get(r.id);
      return o ? { ...r, ...o } : r;
    });
  }, [regions, optimistic]);

  // ── Deselect on any pointer-down that lands outside a rectangle ─────
  // Covers taps on the snapshot background, the threshold strip, the
  // regions list, the page chrome — anywhere that isn't the rectangle
  // body or one of its handles.
  useEffect(() => {
    if (selectedId === null) return;
    const onDown = (e: PointerEvent) => {
      const target = e.target as HTMLElement | null;
      if (!target) return;
      if (target.closest(".regions-rect") || target.closest(".regions-handle")) {
        return;
      }
      setSelectedId(null);
    };
    document.addEventListener("pointerdown", onDown);
    return () => document.removeEventListener("pointerdown", onDown);
  }, [selectedId]);

  // ── Esc cancels draw mode, Backspace deletes selected ────────────────
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setDrawMode(false);
        setSelectedId(null);
      } else if (
        (e.key === "Backspace" || e.key === "Delete") &&
        selectedId !== null
      ) {
        const target = e.target as HTMLElement | null;
        if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA")) {
          return;
        }
        e.preventDefault();
        deleteRegion.mutate(selectedId);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selectedId, deleteRegion]);

  const refreshSnapshot = () => {
    setSnapshotMissing(false);
    setRefreshKey(Date.now());
  };

  // ── UI ──────────────────────────────────────────────────────────────
  const thresholdPct = Math.round((localThreshold ?? 0) * 100);
  const enabledCount = renderedRegions.filter((r) => r.enabled).length;

  return (
    <div className="rise pb-10">
      {/* Hero */}
      <section className="mb-6 flex flex-col gap-3 pt-2 sm:mb-8">
        <span className="eyebrow">Field surveyor</span>
        <h1 className="font-display text-[2.4rem] leading-[1.05] tracking-tight text-[var(--color-cream-50)] sm:text-[3rem]">
          Mark the corners of stillness.{" "}
          <span className="display-italic text-[var(--color-ember-400)]">
            Let the birds through.
          </span>
        </h1>
        <p className="max-w-[60ch] text-[0.95rem] leading-relaxed text-[var(--color-sage-100)]">
          Draw rectangles over decoys, perches, or anything else the camera
          shouldn&apos;t flag as a bird. Detections that overlap an enabled
          region by more than the threshold are filtered before classification.
        </p>
      </section>

      {/* Two-column layout on desktop, stacked on mobile */}
      <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_380px]">
        {/* Snapshot column */}
        <div className="flex flex-col gap-3">
          {/* Toolbar */}
          <div className="glass flex flex-wrap items-center gap-3 rounded-[var(--radius-card)] px-4 py-2.5">
            <span className="firefly inline-block h-1.5 w-1.5 shrink-0 rounded-full bg-[var(--color-ember-400)]" />
            <span className="eyebrow whitespace-nowrap">
              Viewfinder
              {imgSize ? (
                <>
                  <span className="mx-2 opacity-40">/</span>
                  <span className="font-mono text-[var(--color-cream-100)]">
                    {imgSize.w}×{imgSize.h}
                  </span>
                </>
              ) : null}
            </span>
            <button
              type="button"
              onClick={refreshSnapshot}
              className="btn-quiet ml-auto !py-1.5 !text-[0.78rem]"
              title="Refresh the camera frame"
            >
              <RefreshIcon />
              Refresh
            </button>
          </div>

          <SnapshotCanvas
            imageUrl={imageUrl.snapshot()}
            regions={renderedRegions}
            selectedId={selectedId}
            onSelect={setSelectedId}
            drawMode={drawMode}
            onCancelDrawMode={() => setDrawMode(false)}
            onCommitDraft={(rect) => createRegion.mutate(rect)}
            onPatchRegion={patchRegionGeometry}
            onDimensions={(w, h) => setImgSize({ w, h })}
            onMissing={() => setSnapshotMissing(true)}
            refreshKey={refreshKey}
          />

          {/* Hint strip below the snapshot — stays helpful without being noisy */}
          <p className="px-1 text-[0.74rem] text-[var(--color-moss-300)]">
            <span className="eyebrow mr-2">tip</span>
            Drag on the frame to draw · tap a region to edit · drag handles to
            resize · Esc to cancel · Backspace to delete.
          </p>
        </div>

        {/* Side rail on desktop / stacked on mobile */}
        <div className="flex flex-col gap-4">
          {/* Threshold strip */}
          <div className="glass rounded-[var(--radius-card)] p-4">
            <div className="flex items-baseline justify-between gap-3">
              <span className="eyebrow">Overlap threshold</span>
              <span className="font-mono text-[1.1rem] text-[var(--color-ember-400)]">
                {thresholdPct}%
              </span>
            </div>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={localThreshold ?? 0.5}
              onChange={(e) => onThresholdChange(parseFloat(e.target.value))}
              className="threshold-range mt-1"
              style={{
                ["--val" as string]: `${thresholdPct}%`,
              } as React.CSSProperties}
              aria-label="Global overlap threshold"
            />
            <div className="mt-1 flex justify-between font-mono text-[0.62rem] text-[var(--color-moss-300)]">
              <span>loose · 0%</span>
              <span>strict · 100%</span>
            </div>
            <p className="mt-2 text-[0.74rem] leading-relaxed text-[var(--color-sage-200)]">
              A detection is dropped when this much of its bounding box sits
              inside any enabled region. Higher = strictly inside.
            </p>
            <div className="mt-3 flex items-baseline gap-2 text-[0.74rem]">
              <span className="eyebrow">active</span>
              <span className="font-mono text-[var(--color-cream-100)]">
                {enabledCount} of {renderedRegions.length}
              </span>
            </div>
          </div>

          <RegionsList
            regions={renderedRegions}
            selectedId={selectedId}
            onSelect={setSelectedId}
            onToggle={(id, enabled) => updateRegion.mutate({ id, enabled })}
            onRename={(id, label) => updateRegion.mutate({ id, label })}
            onDelete={(id) => deleteRegion.mutate(id)}
            onAdd={() => {
              setDrawMode((d) => !d);
              setSelectedId(null);
            }}
            drawMode={drawMode}
          />

          {snapshotMissing ? (
            <div className="rounded-[var(--radius-card)] border border-dashed border-[var(--color-rust-600)] bg-[rgba(180,92,63,0.08)] p-3 text-[0.78rem] text-[var(--color-rust-500)]">
              The pipeline isn&apos;t writing snapshots yet. Make sure it&apos;s
              running, then hit{" "}
              <button
                type="button"
                onClick={refreshSnapshot}
                className="underline-offset-4 hover:underline"
              >
                refresh
              </button>
              .
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}

function RefreshIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <path d="M3 12a9 9 0 0 1 15.6-6L21 8" />
      <path d="M21 3v5h-5" />
      <path d="M21 12a9 9 0 0 1-15.6 6L3 16" />
      <path d="M3 21v-5h5" />
    </svg>
  );
}
