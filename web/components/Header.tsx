"use client";

import { useEffect, useState } from "react";
import { usePathname } from "next/navigation";
import Link from "next/link";
import { LogoMark, GrafanaIcon, HomeIcon, ReviewIcon } from "./Logo";

export function Header() {
  const pathname = usePathname() ?? "/dashboard";
  const isDashboard = pathname.startsWith("/dashboard") || pathname === "/";
  const isReview = pathname.startsWith("/review");
  const [now, setNow] = useState<string>("");

  useEffect(() => {
    const tick = () =>
      setNow(
        new Date().toLocaleTimeString(undefined, {
          hour: "numeric",
          minute: "2-digit",
        })
      );
    tick();
    const id = window.setInterval(tick, 30_000);
    return () => window.clearInterval(id);
  }, []);

  return (
    <header className="sticky top-0 z-30 backdrop-blur-md">
      <div className="absolute inset-0 -z-10 border-b border-[var(--color-moss-700)] bg-[linear-gradient(180deg,rgba(7,9,10,0.85),rgba(7,9,10,0.55))]" />

      <div className="mx-auto flex max-w-[1400px] items-center justify-between gap-3 px-4 py-3 sm:gap-4 sm:px-6 sm:py-4">
        <Link
          href="/dashboard"
          className="group flex min-w-0 items-center gap-2.5 sm:gap-3"
        >
          <span className="grid h-9 w-9 shrink-0 place-items-center rounded-full border border-[var(--color-moss-700)] bg-[rgba(20,26,22,0.6)] transition-all group-hover:border-[var(--color-ember-500)]">
            <LogoMark size={22} />
          </span>
          <span className="min-w-0 leading-tight">
            <span className="block whitespace-nowrap font-display italic tracking-tight text-[clamp(1.05rem,4vw,1.35rem)] text-[var(--color-cream-50)]">
              Feeder Cam
            </span>
            <span className="eyebrow hidden truncate sm:block">
              Frisco · TX <span className="mx-2 opacity-40">/</span> 32.78°N 96.83°W
            </span>
          </span>
        </Link>

        <nav className="flex shrink-0 items-center gap-1.5 sm:gap-2">
          <Link
            href="/dashboard"
            data-active={isDashboard}
            aria-label="Dashboard"
            className="nav-btn"
          >
            <HomeIcon size={18} />
            <span className="hidden sm:inline">Dashboard</span>
          </Link>
          <Link
            href="/review"
            data-active={isReview}
            aria-label="Review"
            className="nav-btn"
          >
            <ReviewIcon size={18} />
            <span className="hidden sm:inline">Review</span>
          </Link>
          <a
            href="/grafana"
            title="Open Grafana analytics"
            aria-label="Grafana"
            className="nav-btn nav-btn--ember"
          >
            <GrafanaIcon size={14} />
            <span className="hidden sm:inline">Grafana</span>
          </a>
        </nav>
      </div>

      <div className="mx-auto flex max-w-[1400px] items-center gap-3 px-4 pb-2.5 text-[0.68rem] sm:gap-4 sm:px-6 sm:pb-3 sm:text-[0.72rem]">
        <span className="firefly inline-block h-1.5 w-1.5 shrink-0 rounded-full bg-[var(--color-ember-400)]" />
        <span className="eyebrow whitespace-nowrap">
          <span className="hidden sm:inline">live · last sweep </span>
          <span className="sm:hidden">live · </span>
          {now || "—"}
        </span>
        <span className="eyebrow ml-auto hidden truncate opacity-70 md:block">
          two-stage pipeline · YOLO11n → ViT-S · Hailo-10H NPU
        </span>
        <span className="eyebrow ml-auto whitespace-nowrap opacity-60 sm:hidden">
          Frisco · TX
        </span>
      </div>
    </header>
  );
}
