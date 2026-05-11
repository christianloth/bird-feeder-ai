"use client";

import { useState } from "react";
import type { IgnoreRegion } from "@/lib/types";

interface Props {
  regions: IgnoreRegion[];
  selectedId: number | null;
  onSelect: (id: number | null) => void;
  onToggle: (id: number, enabled: boolean) => void;
  onRename: (id: number, label: string) => void;
  onDelete: (id: number) => void;
  onAdd: () => void;
  drawMode: boolean;
}

export function RegionsList({
  regions,
  selectedId,
  onSelect,
  onToggle,
  onRename,
  onDelete,
  onAdd,
  drawMode,
}: Props) {
  const [editingId, setEditingId] = useState<number | null>(null);
  const [draft, setDraft] = useState<string>("");
  const [confirmDeleteId, setConfirmDeleteId] = useState<number | null>(null);

  const beginRename = (r: IgnoreRegion) => {
    setEditingId(r.id);
    setDraft(r.label);
  };

  const commitRename = (r: IgnoreRegion) => {
    const next = draft.trim();
    if (next !== r.label) onRename(r.id, next);
    setEditingId(null);
  };

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <span className="eyebrow">Regions ({regions.length})</span>
        <button
          type="button"
          onClick={onAdd}
          data-active={drawMode}
          className="btn-ember !py-1.5 !text-[0.78rem]"
        >
          {drawMode ? (
            <>
              <CrosshairIcon /> drawing — drag on frame
            </>
          ) : (
            <>
              <PlusIcon /> Add region
            </>
          )}
        </button>
      </div>

      {regions.length === 0 ? (
        <div className="rounded-[var(--radius-card)] border border-dashed border-[var(--color-moss-500)] bg-[rgba(7,9,10,0.45)] p-6 text-center">
          <span className="eyebrow block">empty</span>
          <p className="mt-2 font-display italic text-[1.1rem] leading-snug text-[var(--color-cream-50)]">
            Nothing watched.
          </p>
          <p className="mt-1 text-[0.82rem] text-[var(--color-sage-200)]">
            Tap <span className="font-mono">Add region</span> and drag a
            rectangle on the frame.
          </p>
        </div>
      ) : (
        <ul className="flex flex-col gap-2.5">
          {regions.map((r) => {
            const isSelected = r.id === selectedId;
            const isEditing = editingId === r.id;
            const w = Math.round(r.x2 - r.x1);
            const h = Math.round(r.y2 - r.y1);
            return (
              <li
                key={r.id}
                className="regions-row"
                data-selected={isSelected}
                data-disabled={!r.enabled}
                onClick={() => onSelect(isSelected ? null : r.id)}
              >
                <span
                  className="regions-toggle"
                  data-on={r.enabled}
                  role="switch"
                  aria-checked={r.enabled}
                  aria-label={r.enabled ? "Disable region" : "Enable region"}
                  onClick={(e) => {
                    e.stopPropagation();
                    onToggle(r.id, !r.enabled);
                  }}
                />

                <div className="min-w-0">
                  {isEditing ? (
                    <input
                      autoFocus
                      value={draft}
                      onChange={(e) => setDraft(e.target.value)}
                      onClick={(e) => e.stopPropagation()}
                      onBlur={() => commitRename(r)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") {
                          e.preventDefault();
                          commitRename(r);
                        } else if (e.key === "Escape") {
                          setEditingId(null);
                        }
                      }}
                      placeholder="Label"
                      className="input-quiet !h-8 !text-[0.86rem]"
                    />
                  ) : (
                    <button
                      type="button"
                      className="block w-full truncate text-left font-display italic text-[1.05rem] leading-tight text-[var(--color-cream-50)] hover:text-[var(--color-ember-300)]"
                      onClick={(e) => {
                        e.stopPropagation();
                        beginRename(r);
                      }}
                      title="Rename"
                    >
                      {r.label || "Untitled"}
                    </button>
                  )}
                  <div className="mt-0.5 flex items-baseline gap-2 font-mono text-[0.66rem] text-[var(--color-sage-200)]">
                    <span>
                      {Math.round(r.x1)},{Math.round(r.y1)}
                    </span>
                    <span className="text-[var(--color-moss-300)]">→</span>
                    <span>
                      {Math.round(r.x2)},{Math.round(r.y2)}
                    </span>
                    <span className="ml-auto text-[var(--color-moss-300)]">
                      {w}×{h}
                    </span>
                  </div>
                </div>

                <button
                  type="button"
                  aria-label="Delete region"
                  className="ml-1 inline-flex h-9 w-9 items-center justify-center rounded-full text-[var(--color-rust-500)] transition-all hover:bg-[rgba(201,122,92,0.12)] hover:text-[var(--color-cream-50)]"
                  onClick={(e) => {
                    e.stopPropagation();
                    setConfirmDeleteId(
                      confirmDeleteId === r.id ? null : r.id
                    );
                  }}
                >
                  {confirmDeleteId === r.id ? (
                    <span
                      className="text-[0.7rem] font-mono"
                      onClick={(e) => {
                        e.stopPropagation();
                        onDelete(r.id);
                        setConfirmDeleteId(null);
                      }}
                    >
                      sure?
                    </span>
                  ) : (
                    <TrashIcon />
                  )}
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}

function PlusIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden>
      <path d="M12 5v14M5 12h14" />
    </svg>
  );
}

function CrosshairIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" aria-hidden>
      <circle cx="12" cy="12" r="8" />
      <path d="M12 2v4M12 18v4M2 12h4M18 12h4" />
      <circle cx="12" cy="12" r="1.6" fill="currentColor" />
    </svg>
  );
}

function TrashIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M3.5 6.5h17M9 6.5V4.5a1.5 1.5 0 0 1 1.5-1.5h3A1.5 1.5 0 0 1 15 4.5v2" />
      <path d="M5.5 6.5l1 13a2 2 0 0 0 2 1.8h7a2 2 0 0 0 2-1.8l1-13" />
      <path d="M10 11v6M14 11v6" />
    </svg>
  );
}
