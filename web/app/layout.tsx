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

const siteUrl = process.env.NEXT_PUBLIC_SITE_URL || "https://feedercam.loth.me";

export const metadata: Metadata = {
  metadataBase: new URL(siteUrl),
  title: "Feeder Cam — An evening with the birds",
  description:
    "A feeder in Frisco, watched by a Pi. Every visitor logged with the time, the weather, and the model's best guess.",
  openGraph: {
    title: "Feeder Cam",
    description: "An evening with the birds. A feeder in Frisco, watched 24/7 by a Pi.",
    siteName: "Feeder Cam",
    type: "website",
    locale: "en_US",
  },
  twitter: {
    card: "summary_large_image",
    title: "Feeder Cam",
    description: "An evening with the birds. A feeder in Frisco, watched 24/7 by a Pi.",
  },
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
          <main className="mx-auto max-w-[1400px] px-4 pb-20 pt-6 sm:px-6">{children}</main>
          <footer className="mx-auto max-w-[1400px] px-4 pb-10 pt-6 text-[0.72rem] sm:px-6">
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
