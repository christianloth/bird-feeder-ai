import type { Metadata } from "next";
import { DM_Sans, Fraunces, JetBrains_Mono } from "next/font/google";
import { Providers } from "./providers";
import { Header } from "@/components/Header";
import "./globals.css";

const display = Fraunces({
  subsets: ["latin"],
  variable: "--font-display",
  display: "swap",
  axes: ["opsz", "SOFT"],
});

const sans = DM_Sans({
  subsets: ["latin"],
  variable: "--font-sans",
  display: "swap",
});

const mono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Bird Feeder · Field Notes",
  description:
    "A 24/7 bird-detection log running on a Raspberry Pi 5 with a Hailo-10H NPU, watching a feeder in Frisco, TX.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${display.variable} ${sans.variable} ${mono.variable}`}>
      <body>
        <Providers>
          <Header />
          <main className="mx-auto max-w-[1400px] px-6 pb-20 pt-6">{children}</main>
          <footer className="mx-auto max-w-[1400px] px-6 pb-10 pt-6 text-[0.72rem]">
            <div className="flex items-center justify-between border-t border-[var(--color-moss-700)] pt-4 text-[var(--color-moss-300)]">
              <span className="eyebrow">an evening with the birds</span>
              <span className="font-mono opacity-70">v0.1 · pi5 · hailo-10h</span>
            </div>
          </footer>
        </Providers>
      </body>
    </html>
  );
}
