"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { Species } from "@/lib/types";

interface Props {
  value: string; // species_id as string, "" = all
  onChange: (next: string) => void;
  species: Species[];
}

// Species-with-detections is a small list in practice, but cap the rendered
// rows so a future large list can't blow up the dropdown.
const MAX_RESULTS = 60;

// A type-to-search replacement for the species <select>. Matches the
// `input-quiet` / `glass` styling so it sits in the filter grid unchanged,
// filters by common *and* scientific name, and is keyboard-navigable.
export function SpeciesCombobox({ value, onChange, species }: Props) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [highlight, setHighlight] = useState(0);
  const rootRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLUListElement>(null);

  const selected = useMemo(
    () => species.find((s) => String(s.id) === value) ?? null,
    [species, value],
  );

  // While the menu is closed the input mirrors the current selection. This
  // also picks up external resets ("reset all") and the species list
  // resolving after first paint.
  useEffect(() => {
    if (!open) setQuery(selected ? selected.common_name : "");
  }, [selected, open]);

  const q = query.trim().toLowerCase();
  const matches = useMemo(() => {
    if (!q) return species.slice(0, MAX_RESULTS);
    return species
      .filter(
        (s) =>
          s.common_name.toLowerCase().includes(q) ||
          s.scientific_name.toLowerCase().includes(q),
      )
      .slice(0, MAX_RESULTS);
  }, [q, species]);

  // The "All species" reset row sits at the top only when not actively
  // searching, so it occupies row index 0 and shifts the matches down by one.
  const showAll = q.length === 0;
  const rowCount = matches.length + (showAll ? 1 : 0);

  // Reset the highlight to the top whenever the result set changes.
  useEffect(() => {
    setHighlight(0);
  }, [q]);

  // Close on a click outside the combobox.
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [open]);

  // Keep the highlighted row scrolled into view during keyboard nav.
  useEffect(() => {
    if (!open || !listRef.current) return;
    listRef.current
      .querySelector<HTMLElement>(`[data-row="${highlight}"]`)
      ?.scrollIntoView({ block: "nearest" });
  }, [highlight, open]);

  const openMenu = () => {
    setOpen(true);
    // Clear the mirrored name so the full list shows and the first keystroke
    // starts a fresh search. Blur restores it via the effect above.
    setQuery("");
  };

  const choose = (next: string) => {
    onChange(next);
    setOpen(false);
    const sp = species.find((s) => String(s.id) === next);
    setQuery(sp ? sp.common_name : "");
    inputRef.current?.blur();
  };

  // Resolve the row at a highlight index to its selection action.
  const pick = (index: number) => {
    if (showAll && index === 0) return choose("");
    const sp = matches[index - (showAll ? 1 : 0)];
    if (sp) choose(String(sp.id));
  };

  const clear = () => {
    onChange("");
    setQuery("");
    setOpen(true);
    inputRef.current?.focus();
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      if (!open) return openMenu();
      setHighlight((h) => Math.min(h + 1, rowCount - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setHighlight((h) => Math.max(h - 1, 0));
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (open) pick(highlight);
    } else if (e.key === "Escape") {
      setOpen(false);
      setQuery(selected ? selected.common_name : "");
      inputRef.current?.blur();
    }
  };

  const rowClass = (active: boolean) =>
    `flex cursor-pointer items-center justify-between gap-3 rounded-lg px-3 py-2 text-[0.86rem] transition-colors ${
      active
        ? "bg-[rgba(61,165,217,0.12)] text-[var(--color-cream-50)]"
        : "text-[var(--color-sage-100)]"
    }`;

  return (
    <div ref={rootRef} className="relative">
      {/* search glyph */}
      <svg
        aria-hidden
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
        className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-[var(--color-sage-200)]"
      >
        <circle cx="11" cy="11" r="7" />
        <path d="m20 20-3.5-3.5" />
      </svg>

      <input
        ref={inputRef}
        type="text"
        role="combobox"
        aria-label="Species"
        aria-expanded={open}
        aria-autocomplete="list"
        aria-controls="species-combobox-list"
        autoComplete="off"
        spellCheck={false}
        className="input-quiet"
        style={{ paddingLeft: "2.25rem", paddingRight: "2.25rem" }}
        placeholder={open ? "Search species…" : "All species"}
        value={query}
        onChange={(e) => {
          setQuery(e.target.value);
          setOpen(true);
        }}
        onFocus={openMenu}
        onKeyDown={onKeyDown}
        onBlur={(e) => {
          if (!rootRef.current?.contains(e.relatedTarget as Node)) setOpen(false);
        }}
      />

      {(query || value) && (
        <button
          type="button"
          aria-label="Clear species filter"
          onClick={clear}
          className="absolute right-2 top-1/2 flex size-6 -translate-y-1/2 items-center justify-center rounded-full text-[var(--color-sage-200)] transition hover:bg-[rgba(255,255,255,0.05)] hover:text-[var(--color-ember-400)]"
        >
          <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinecap="round"
            className="size-3.5"
          >
            <path d="M6 6l12 12M18 6 6 18" />
          </svg>
        </button>
      )}

      {open && (
        <ul
          ref={listRef}
          id="species-combobox-list"
          role="listbox"
          className="absolute z-30 mt-1.5 max-h-64 w-full overflow-auto rounded-xl border border-[var(--color-moss-700)] bg-[var(--color-ink-800)] p-1 shadow-[0_24px_60px_-20px_rgba(0,0,0,0.7)]"
        >
          {showAll && (
            <li
              role="option"
              aria-selected={value === ""}
              data-row={0}
              onMouseDown={(e) => {
                e.preventDefault();
                choose("");
              }}
              onMouseEnter={() => setHighlight(0)}
              className={rowClass(highlight === 0)}
            >
              <span className={value === "" ? "text-[var(--color-ember-400)]" : ""}>
                All species
              </span>
              {value === "" && <Check />}
            </li>
          )}

          {matches.length === 0 ? (
            <li className="px-3 py-2 text-[0.82rem] text-[var(--color-sage-200)]">
              No species match &ldquo;{query.trim()}&rdquo;
            </li>
          ) : (
            matches.map((sp, i) => {
              const row = i + (showAll ? 1 : 0);
              const isSelected = String(sp.id) === value;
              return (
                <li
                  key={sp.id}
                  role="option"
                  aria-selected={isSelected}
                  data-row={row}
                  onMouseDown={(e) => {
                    e.preventDefault();
                    choose(String(sp.id));
                  }}
                  onMouseEnter={() => setHighlight(row)}
                  className={rowClass(highlight === row)}
                >
                  <span className="min-w-0 truncate">
                    <span className={isSelected ? "text-[var(--color-ember-400)]" : ""}>
                      {sp.common_name}
                    </span>
                    {sp.scientific_name && (
                      <span className="display-italic ml-2 text-[0.78rem] text-[var(--color-sage-200)]">
                        {sp.scientific_name}
                      </span>
                    )}
                  </span>
                  {isSelected && <Check />}
                </li>
              );
            })
          )}
        </ul>
      )}
    </div>
  );
}

function Check() {
  return (
    <svg
      aria-hidden
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="size-4 shrink-0 text-[var(--color-ember-400)]"
    >
      <path d="M20 6 9 17l-5-5" />
    </svg>
  );
}
