"use client";

import { useEffect, useState } from "react";
import { usePathname } from "next/navigation";
import Link from "next/link";
import { LogoMark, GrafanaIcon } from "./Logo";

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
      <div className="absolute inset-0 -z-10 bg-[linear-gradient(180deg,rgba(7,9,10,0.85),rgba(7,9,10,0.55))] border-b border-[var(--color-moss-700)]" />
      <div className="mx-auto flex max-w-[1400px] items-center justify-between gap-4 px-6 py-4">
        <Link href="/dashboard" className="group flex items-center gap-3">
          <span className="grid h-9 w-9 place-items-center rounded-full border border-[var(--color-moss-700)] bg-[rgba(20,26,22,0.6)] transition-all group-hover:border-[var(--color-ember-500)]">
            <LogoMark size={22} />
          </span>
          <span className="leading-tight">
            <span className="block font-display text-[1.35rem] italic tracking-tight text-[var(--color-cream-50)]">
              Bird Feeder
              <span className="text-[var(--color-ember-400)]">.</span>
            </span>
            <span className="eyebrow block">
              Frisco · TX <span className="mx-2 opacity-40">/</span> 32.78°N 96.83°W
            </span>
          </span>
        </Link>

        <nav className="flex items-center gap-1 sm:gap-2">
          <Link
            href="/dashboard"
            className="btn-quiet"
            data-active={isDashboard}
          >
            <span className="hidden sm:inline">Dashboard</span>
            <span className="sm:hidden">Home</span>
          </Link>
          <Link href="/review" className="btn-quiet" data-active={isReview}>
            Review
          </Link>
          <a href="/grafana" className="btn-ember" title="Open Grafana analytics">
            <GrafanaIcon size={14} />
            <span>Grafana</span>
          </a>
        </nav>
      </div>

      <div className="mx-auto flex max-w-[1400px] items-center gap-4 px-6 pb-3 text-[0.72rem]">
        <span className="firefly inline-block h-1.5 w-1.5 rounded-full bg-[var(--color-ember-400)]" />
        <span className="eyebrow">live · last sweep {now || "—"}</span>
        <span className="ml-auto eyebrow opacity-70 hidden md:block">
          two-stage pipeline · YOLO11n → ViT-S · Hailo-10H NPU
        </span>
      </div>
    </header>
  );
}
