"use client";

import { useEffect, useState } from "react";
import { usePathname } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { LogoMark, GrafanaIcon, HomeIcon, RegionsIcon, ReviewIcon, SweepIcon } from "./Logo";

// Header uses plain <a> tags for cross-page navigation rather than
// next/link. With output: "export" + React 19, Link's router.push()
// sometimes silently no-ops on phones (see "Add temporary on-screen
// click logger" commit for the diagnostic that exposed this). A hard
// page navigation is slightly less snappy but works every time.

export function Header() {
  const pathname = usePathname() ?? "/dashboard";
  const isDashboard = pathname.startsWith("/dashboard") || pathname === "/";
  const isReview = pathname.startsWith("/review");
  const isRegions = pathname.startsWith("/regions");
  const isSweep = pathname.startsWith("/sweep");
  const features = useQuery({
    queryKey: ["features"],
    queryFn: api.features,
    staleTime: 60_000,
  });
  const sweepEnabled = features.data?.sweep ?? false;
  const [now, setNow] = useState<string>("");
  // Grafana is gated behind basic-auth for now; the nav button shows a
  // transient "coming soon" bubble instead of triggering the password prompt.
  const [comingSoon, setComingSoon] = useState(false);
  useEffect(() => {
    if (!comingSoon) return;
    const id = window.setTimeout(() => setComingSoon(false), 2200);
    return () => window.clearTimeout(id);
  }, [comingSoon]);

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
        <a
          href="/dashboard/"
          className="group flex min-w-0 items-center gap-2.5 sm:gap-3"
        >
          <span className="grid h-9 w-9 shrink-0 place-items-center rounded-full border border-[var(--color-moss-700)] bg-[rgba(20,26,22,0.6)] transition-all group-hover:border-[var(--color-ember-500)]">
            <LogoMark size={22} />
          </span>
          <span className="min-w-0 leading-tight">
            <span className="block whitespace-nowrap font-display italic tracking-tight text-[clamp(1.05rem,4vw,1.35rem)] text-[var(--color-cream-50)]">
              Feeder Cam
            </span>
            <span className="eyebrow hidden whitespace-nowrap lg:block">
              Frisco · TX <span className="mx-2 opacity-40">/</span> 32.78°N 96.83°W
            </span>
          </span>
        </a>

        <nav className="flex shrink-0 items-center gap-1.5 sm:gap-2">
          <a
            href="/dashboard/"
            data-active={isDashboard}
            aria-label="Dashboard"
            className="nav-btn"
          >
            <HomeIcon size={18} />
            <span className="hidden sm:inline">Dashboard</span>
          </a>
          <a
            href="/review/"
            data-active={isReview}
            aria-label="Review"
            className="nav-btn"
          >
            <ReviewIcon size={18} />
            <span className="hidden sm:inline">Review</span>
          </a>
          <a
            href="/regions/"
            data-active={isRegions}
            aria-label="Regions"
            className="nav-btn"
          >
            <RegionsIcon size={18} />
            <span className="hidden sm:inline">Regions</span>
          </a>
          {sweepEnabled ? (
            <a
              href="/sweep/"
              data-active={isSweep}
              aria-label="Sweep"
              className="nav-btn"
            >
              <SweepIcon size={18} />
              <span className="hidden sm:inline">Sweep</span>
            </a>
          ) : null}
          <div className="relative">
            <button
              type="button"
              onClick={() => setComingSoon(true)}
              title="Grafana analytics — coming soon"
              aria-label="Grafana — coming soon"
              className="nav-btn nav-btn--ember"
            >
              <GrafanaIcon size={14} />
              <span className="hidden sm:inline">Grafana</span>
            </button>
            {comingSoon ? (
              <span
                role="status"
                className="rise absolute right-0 top-full z-40 mt-2 whitespace-nowrap rounded-full border border-[var(--color-moss-700)] bg-[rgba(7,9,10,0.94)] px-3 py-1.5 font-mono text-[0.68rem] tracking-wide text-[var(--color-cream-100)] shadow-[0_8px_24px_-8px_rgba(0,0,0,0.7)]"
              >
                Coming soon
              </span>
            ) : null}
          </div>
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
