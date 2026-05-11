"use client";

import { useEffect, useState } from "react";
import { usePathname } from "next/navigation";

interface Entry {
  ts: string;
  kind: "click" | "path" | "error";
  text: string;
}

const STORAGE_KEY = "bfa.debugClicks";

/**
 * Floating bottom-left debug panel that captures every click on the page
 * plus pathname changes and global JS errors.
 *
 * Opt-in. Hidden by default. To turn on, append `?debug=1` to any URL —
 * the flag persists in localStorage so it survives navigation. To turn
 * off, append `?debug=0` (or clear `bfa.debugClicks` from localStorage).
 */
export function DebugClickLogger() {
  const pathname = usePathname() ?? "/";
  const [enabled, setEnabled] = useState(false);
  const [entries, setEntries] = useState<Entry[]>([]);
  const [collapsed, setCollapsed] = useState(false);

  // Read the URL param on mount AND on pathname change, since hard
  // navigations re-read window.location while soft pushes update it.
  useEffect(() => {
    if (typeof window === "undefined") return;
    const params = new URLSearchParams(window.location.search);
    const flag = params.get("debug");
    if (flag === "1") {
      try {
        window.localStorage.setItem(STORAGE_KEY, "1");
      } catch {
        /* ignore */
      }
      setEnabled(true);
      return;
    }
    if (flag === "0") {
      try {
        window.localStorage.removeItem(STORAGE_KEY);
      } catch {
        /* ignore */
      }
      setEnabled(false);
      return;
    }
    try {
      setEnabled(window.localStorage.getItem(STORAGE_KEY) === "1");
    } catch {
      setEnabled(false);
    }
  }, [pathname]);

  // Log every pathname change so we can tell whether navigation actually
  // fired or stalled before reaching the new page.
  useEffect(() => {
    if (!enabled) return;
    setEntries((prev) => {
      const next: Entry = { ts: stamp(), kind: "path", text: `→ ${pathname}` };
      return [next, ...prev].slice(0, 10);
    });
  }, [enabled, pathname]);

  // Capture every click on the page. Reports the deepest target and the
  // closest <a> ancestor's href, plus a marker for whether defaultPrevented
  // was already true by the time it bubbled to the document.
  useEffect(() => {
    if (!enabled) return;
    const onClick = (e: MouseEvent) => {
      const target = e.target as HTMLElement | null;
      if (!target) return;
      const tag = target.tagName.toLowerCase();
      const cls = (target.className && typeof target.className === "string")
        ? `.${target.className.split(/\s+/).filter(Boolean).slice(0, 2).join(".")}`
        : "";
      const anchor = target.closest("a");
      const href = anchor?.getAttribute("href") ?? "—";
      const prevented = e.defaultPrevented ? " ⛔" : "";
      setEntries((prev) => {
        const next: Entry = {
          ts: stamp(),
          kind: "click",
          text: `${tag}${cls} → ${href}${prevented}`,
        };
        return [next, ...prev].slice(0, 10);
      });
    };
    // Use bubble phase so we see whether anything ahead of us called
    // preventDefault. Document-level so we catch every click.
    document.addEventListener("click", onClick);
    return () => document.removeEventListener("click", onClick);
  }, [enabled]);

  // Capture global JS errors and unhandled promise rejections — these
  // could explain a stuck navigation.
  useEffect(() => {
    if (!enabled) return;
    const onError = (e: ErrorEvent) => {
      setEntries((prev) => {
        const next: Entry = {
          ts: stamp(),
          kind: "error",
          text: `err: ${e.message.slice(0, 80)}`,
        };
        return [next, ...prev].slice(0, 10);
      });
    };
    const onReject = (e: PromiseRejectionEvent) => {
      const msg =
        e.reason instanceof Error
          ? e.reason.message
          : String(e.reason ?? "unknown");
      setEntries((prev) => {
        const next: Entry = {
          ts: stamp(),
          kind: "error",
          text: `rej: ${msg.slice(0, 80)}`,
        };
        return [next, ...prev].slice(0, 10);
      });
    };
    window.addEventListener("error", onError);
    window.addEventListener("unhandledrejection", onReject);
    return () => {
      window.removeEventListener("error", onError);
      window.removeEventListener("unhandledrejection", onReject);
    };
  }, [enabled]);

  if (!enabled) return null;

  return (
    <div
      style={{
        position: "fixed",
        left: 8,
        bottom: 8,
        zIndex: 9999,
        fontFamily: "var(--font-mono)",
        fontSize: 10,
        lineHeight: 1.3,
        maxWidth: 320,
        background: "rgba(7, 9, 10, 0.9)",
        border: "1px solid rgba(244, 207, 138, 0.5)",
        borderRadius: 8,
        padding: 8,
        color: "#f6efe1",
        backdropFilter: "blur(6px)",
        pointerEvents: "auto",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          marginBottom: collapsed ? 0 : 6,
          gap: 8,
        }}
      >
        <span style={{ color: "#f4cf8a", fontWeight: 600 }}>
          DEBUG · {pathname}
        </span>
        <button
          type="button"
          onClick={() => setCollapsed((c) => !c)}
          style={{
            background: "transparent",
            border: "1px solid rgba(244, 207, 138, 0.45)",
            color: "#f4cf8a",
            borderRadius: 4,
            padding: "1px 6px",
            fontSize: 10,
            cursor: "pointer",
          }}
        >
          {collapsed ? "show" : "hide"}
        </button>
      </div>
      {!collapsed && (
        <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
          {entries.length === 0 ? (
            <span style={{ opacity: 0.6 }}>no events yet…</span>
          ) : (
            entries.map((e, i) => (
              <div
                key={i}
                style={{
                  display: "flex",
                  gap: 6,
                  color:
                    e.kind === "error"
                      ? "#c97a5c"
                      : e.kind === "path"
                      ? "#a3c79e"
                      : "#ece3cf",
                }}
              >
                <span style={{ opacity: 0.55 }}>{e.ts}</span>
                <span style={{ overflow: "hidden", textOverflow: "ellipsis" }}>
                  {e.text}
                </span>
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
}

function stamp() {
  const d = new Date();
  return `${String(d.getMinutes()).padStart(2, "0")}:${String(d.getSeconds()).padStart(2, "0")}.${String(
    d.getMilliseconds(),
  ).padStart(3, "0")}`;
}
