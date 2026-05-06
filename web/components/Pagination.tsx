"use client";

interface Props {
  page: number;
  totalPages: number;
  onChange: (next: number) => void;
}

export function Pagination({ page, totalPages, onChange }: Props) {
  if (totalPages <= 1) return null;

  const items = pageItems(page, totalPages);

  return (
    <nav
      aria-label="Pagination"
      className="mt-10 flex flex-wrap items-center justify-center gap-1.5"
    >
      <button
        type="button"
        onClick={() => onChange(Math.max(1, page - 1))}
        disabled={page <= 1}
        aria-label="Previous page"
        className="page-btn"
      >
        <Chevron direction="left" />
      </button>

      {items.map((it, idx) =>
        it === "…" ? (
          <span key={`gap-${idx}`} className="page-gap" aria-hidden>
            …
          </span>
        ) : (
          <button
            key={it}
            type="button"
            onClick={() => onChange(it)}
            data-active={it === page}
            aria-current={it === page ? "page" : undefined}
            aria-label={`Page ${it}`}
            className="page-btn"
          >
            {it}
          </button>
        )
      )}

      <button
        type="button"
        onClick={() => onChange(Math.min(totalPages, page + 1))}
        disabled={page >= totalPages}
        aria-label="Next page"
        className="page-btn"
      >
        <Chevron direction="right" />
      </button>
    </nav>
  );
}

function Chevron({ direction }: { direction: "left" | "right" }) {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      {direction === "left" ? <path d="M15 18l-6-6 6-6" /> : <path d="M9 18l6-6-6-6" />}
    </svg>
  );
}

function pageItems(current: number, total: number): (number | "…")[] {
  if (total <= 7) {
    return Array.from({ length: total }, (_, i) => i + 1);
  }
  const out: (number | "…")[] = [1];
  if (current > 3) out.push("…");
  const start = Math.max(2, current - 1);
  const end = Math.min(total - 1, current + 1);
  for (let i = start; i <= end; i++) out.push(i);
  if (current < total - 2) out.push("…");
  out.push(total);
  return out;
}
