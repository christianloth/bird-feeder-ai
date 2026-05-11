"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties, PointerEvent as ReactPointerEvent } from "react";
import type { IgnoreRegion } from "@/lib/types";

export interface DraftRect {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

type DragMode =
  | { kind: "draft"; startX: number; startY: number }
  | { kind: "move"; regionId: number; startX: number; startY: number; orig: DraftRect }
  | {
      kind: "resize";
      regionId: number;
      handle: "nw" | "n" | "ne" | "e" | "se" | "s" | "sw" | "w";
      startX: number;
      startY: number;
      orig: DraftRect;
    };

interface Props {
  imageUrl: string;
  regions: IgnoreRegion[];
  selectedId: number | null;
  onSelect: (id: number | null) => void;
  drawMode: boolean;
  onCancelDrawMode: () => void;
  onCommitDraft: (rect: DraftRect) => void;
  onPatchRegion: (id: number, rect: DraftRect) => void;
  /** Reports the natural dimensions of the snapshot once it loads, so the
   *  parent can show "1080×1920" in the toolbar and validate region geometry. */
  onDimensions: (w: number, h: number) => void;
  /** Called when the snapshot 404s (pipeline not yet writing frames). */
  onMissing: () => void;
  /** A timestamp the parent bumps to force a snapshot reload. */
  refreshKey: number;
}

const HANDLE_KEYS = ["nw", "n", "ne", "e", "se", "s", "sw", "w"] as const;
const MIN_SIZE = 12; // source-pixel minimum so accidental taps don't make a 1×1 region

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

function clampRect(r: DraftRect, w: number, h: number): DraftRect {
  return {
    x1: clamp(Math.min(r.x1, r.x2), 0, w),
    y1: clamp(Math.min(r.y1, r.y2), 0, h),
    x2: clamp(Math.max(r.x1, r.x2), 0, w),
    y2: clamp(Math.max(r.y1, r.y2), 0, h),
  };
}

export function SnapshotCanvas({
  imageUrl,
  regions,
  selectedId,
  onSelect,
  drawMode,
  onCancelDrawMode,
  onCommitDraft,
  onPatchRegion,
  onDimensions,
  onMissing,
  refreshKey,
}: Props) {
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const [imgSize, setImgSize] = useState<{ w: number; h: number } | null>(null);
  const [imgError, setImgError] = useState(false);
  const [draft, setDraft] = useState<DraftRect | null>(null);
  const dragRef = useRef<DragMode | null>(null);
  // Track the displayed (CSS) size of the image so we can map between source
  // pixels (stored in the DB) and displayed pixels (the pointer events).
  const [displayed, setDisplayed] = useState<{ w: number; h: number } | null>(null);

  // Recompute displayed size on resize.
  useEffect(() => {
    const el = imgRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      const rect = el.getBoundingClientRect();
      if (rect.width > 0 && rect.height > 0) {
        setDisplayed({ w: rect.width, h: rect.height });
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [imgSize]);

  // Notify parent of natural dimensions on load.
  useEffect(() => {
    if (imgSize) onDimensions(imgSize.w, imgSize.h);
  }, [imgSize, onDimensions]);

  // Translate a client (page) point to source-pixel coords.
  const clientToSource = (clientX: number, clientY: number) => {
    const el = imgRef.current;
    if (!el || !imgSize) return null;
    const rect = el.getBoundingClientRect();
    const x = ((clientX - rect.left) / rect.width) * imgSize.w;
    const y = ((clientY - rect.top) / rect.height) * imgSize.h;
    return { x: clamp(x, 0, imgSize.w), y: clamp(y, 0, imgSize.h) };
  };

  // Draft rectangle drawing (new region).
  const onCanvasPointerDown = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (!drawMode) return;
    e.preventDefault();
    e.currentTarget.setPointerCapture(e.pointerId);
    const p = clientToSource(e.clientX, e.clientY);
    if (!p) return;
    dragRef.current = { kind: "draft", startX: p.x, startY: p.y };
    setDraft({ x1: p.x, y1: p.y, x2: p.x, y2: p.y });
  };

  const onCanvasPointerMove = (e: ReactPointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag) return;
    const p = clientToSource(e.clientX, e.clientY);
    if (!p || !imgSize) return;

    if (drag.kind === "draft") {
      setDraft({ x1: drag.startX, y1: drag.startY, x2: p.x, y2: p.y });
    } else if (drag.kind === "move") {
      const dx = p.x - drag.startX;
      const dy = p.y - drag.startY;
      const w = drag.orig.x2 - drag.orig.x1;
      const h = drag.orig.y2 - drag.orig.y1;
      let nx = drag.orig.x1 + dx;
      let ny = drag.orig.y1 + dy;
      nx = clamp(nx, 0, imgSize.w - w);
      ny = clamp(ny, 0, imgSize.h - h);
      onPatchRegion(drag.regionId, { x1: nx, y1: ny, x2: nx + w, y2: ny + h });
    } else if (drag.kind === "resize") {
      const next: DraftRect = { ...drag.orig };
      const h = drag.handle;
      if (h.includes("w")) next.x1 = p.x;
      if (h.includes("e")) next.x2 = p.x;
      if (h.includes("n")) next.y1 = p.y;
      if (h.includes("s")) next.y2 = p.y;
      // Enforce min size and bounds.
      const norm = clampRect(next, imgSize.w, imgSize.h);
      if (norm.x2 - norm.x1 < MIN_SIZE || norm.y2 - norm.y1 < MIN_SIZE) return;
      onPatchRegion(drag.regionId, norm);
    }
  };

  const onCanvasPointerUp = (e: ReactPointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    dragRef.current = null;
    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId);
    }
    if (!drag) return;

    if (drag.kind === "draft" && draft && imgSize) {
      const norm = clampRect(draft, imgSize.w, imgSize.h);
      setDraft(null);
      onCancelDrawMode();
      if (norm.x2 - norm.x1 >= MIN_SIZE && norm.y2 - norm.y1 >= MIN_SIZE) {
        onCommitDraft(norm);
      }
    }
  };

  // ── Hit-test helpers (positioned divs handle their own pointerdown via
  //    stopPropagation, so the canvas only sees background clicks). ───────

  const startMove = (
    e: ReactPointerEvent<HTMLDivElement>,
    region: IgnoreRegion,
  ) => {
    // Only intercept the pointer for a move when the region is already
    // selected. An unselected rectangle should let the tap fall through
    // to onClick (so it just selects), and let the browser handle native
    // pinch-zoom / pan gestures freely. After it's selected, its CSS sets
    // touch-action: none so we can own the drag.
    if (region.id !== selectedId) return;
    e.stopPropagation();
    e.preventDefault();
    const p = clientToSource(e.clientX, e.clientY);
    if (!p) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    dragRef.current = {
      kind: "move",
      regionId: region.id,
      startX: p.x,
      startY: p.y,
      orig: { x1: region.x1, y1: region.y1, x2: region.x2, y2: region.y2 },
    };
  };

  const startResize = (
    e: ReactPointerEvent<HTMLDivElement>,
    region: IgnoreRegion,
    handle: (typeof HANDLE_KEYS)[number],
  ) => {
    e.stopPropagation();
    e.preventDefault();
    const p = clientToSource(e.clientX, e.clientY);
    if (!p) return;
    onSelect(region.id);
    e.currentTarget.setPointerCapture(e.pointerId);
    dragRef.current = {
      kind: "resize",
      regionId: region.id,
      handle,
      startX: p.x,
      startY: p.y,
      orig: { x1: region.x1, y1: region.y1, x2: region.x2, y2: region.y2 },
    };
  };

  const finishHandleDrag = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId);
    }
    dragRef.current = null;
  };

  // ── Coordinate translation for rendering ────────────────────────────────

  const scale = useMemo(() => {
    if (!imgSize || !displayed) return null;
    return { sx: displayed.w / imgSize.w, sy: displayed.h / imgSize.h };
  }, [imgSize, displayed]);

  const rectStyle = (r: DraftRect): CSSProperties => {
    if (!scale) return { display: "none" };
    return {
      left: r.x1 * scale.sx,
      top: r.y1 * scale.sy,
      width: (r.x2 - r.x1) * scale.sx,
      height: (r.y2 - r.y1) * scale.sy,
    };
  };

  const onCanvasClick = () => {
    if (drawMode) return;
    onSelect(null);
  };

  return (
    <div
      ref={wrapRef}
      className="regions-frame relative overflow-hidden rounded-[var(--radius-card)] border border-[var(--color-moss-700)] bg-[var(--color-ink-900)]"
    >
      <span className="reticle bl" />
      <span className="reticle br" />

      {imgError ? (
        <div className="flex aspect-[9/16] items-center justify-center px-6 text-center">
          <div className="max-w-[40ch]">
            <span className="eyebrow block">No frame yet</span>
            <p className="mt-3 font-display italic text-[1.4rem] text-[var(--color-cream-50)]">
              The pipeline isn&apos;t writing snapshots — start it and refresh.
            </p>
            <p className="mt-2 text-[0.85rem] text-[var(--color-sage-200)]">
              <span className="font-mono">scripts/start_pipeline.sh</span>
            </p>
          </div>
        </div>
      ) : (
        <>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            ref={imgRef}
            src={`${imageUrl}${imageUrl.includes("?") ? "&" : "?"}t=${refreshKey}`}
            alt="Live camera frame"
            className="block h-auto w-full select-none"
            draggable={false}
            onLoad={(e) => {
              const el = e.currentTarget;
              setImgSize({ w: el.naturalWidth, h: el.naturalHeight });
              setImgError(false);
              const rect = el.getBoundingClientRect();
              if (rect.width > 0 && rect.height > 0) {
                setDisplayed({ w: rect.width, h: rect.height });
              }
            }}
            onError={() => {
              setImgError(true);
              onMissing();
            }}
          />

          {/* Scanline / grain overlay */}
          <div className="regions-overlay-grain" aria-hidden />

          {/* Drawing surface — sits on top of the image. Pointer events
              go to the rectangles first; bg clicks deselect. */}
          <div
            className="regions-canvas absolute inset-0"
            data-mode={drawMode ? "draw" : "idle"}
            onPointerDown={onCanvasPointerDown}
            onPointerMove={onCanvasPointerMove}
            onPointerUp={onCanvasPointerUp}
            onPointerCancel={onCanvasPointerUp}
            onClick={onCanvasClick}
          >
            {regions.map((r) => {
              const isSelected = r.id === selectedId;
              return (
                <div
                  key={r.id}
                  className="regions-rect"
                  style={rectStyle(r)}
                  data-selected={isSelected}
                  data-disabled={!r.enabled}
                  onPointerDown={(e) => startMove(e, r)}
                  onPointerMove={onCanvasPointerMove}
                  onPointerUp={(e) => {
                    finishHandleDrag(e);
                    onCanvasPointerUp(e);
                  }}
                  onPointerCancel={(e) => {
                    finishHandleDrag(e);
                  }}
                  onClick={(e) => {
                    e.stopPropagation();
                    onSelect(r.id);
                  }}
                >
                  {isSelected && scale
                    ? HANDLE_KEYS.map((h) => {
                        const w = (r.x2 - r.x1) * scale.sx;
                        const ht = (r.y2 - r.y1) * scale.sy;
                        const pos: CSSProperties = {};
                        if (h.includes("w")) pos.left = 0;
                        if (h.includes("e")) pos.left = w;
                        if (h === "n" || h === "s") pos.left = w / 2;
                        if (h.includes("n")) pos.top = 0;
                        if (h.includes("s")) pos.top = ht;
                        if (h === "e" || h === "w") pos.top = ht / 2;
                        return (
                          <div
                            key={h}
                            className={`regions-handle h-${h}`}
                            style={pos}
                            onPointerDown={(e) => startResize(e, r, h)}
                            onPointerMove={onCanvasPointerMove}
                            onPointerUp={(e) => {
                              finishHandleDrag(e);
                              onCanvasPointerUp(e);
                            }}
                            onPointerCancel={(e) => finishHandleDrag(e)}
                          />
                        );
                      })
                    : null}
                </div>
              );
            })}

            {draft ? (
              <div className="regions-marquee" style={rectStyle(draft)} />
            ) : null}
          </div>
        </>
      )}
    </div>
  );
}
