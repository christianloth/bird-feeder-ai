import type {
  BulkDeleteResponse,
  Detection,
  DetectionStats,
  FeatureFlags,
  IgnoreRegion,
  IgnoreRegionCreate,
  IgnoreRegionUpdate,
  IgnoreSettings,
  Species,
  SystemStatus,
  Weather,
} from "./types";

function buildQuery(params: Record<string, string | number | undefined | null>): string {
  const qs = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v === undefined || v === "" || v === null) continue;
    qs.set(k, String(v));
  }
  return qs.toString();
}

// Admin-token plumbing. Writes (POST/PATCH/PUT/DELETE) need the token;
// reads stay anonymous. Token is persisted in localStorage so a device
// only ever has to be unlocked once.
const TOKEN_KEY = "bfa.adminToken";
const READ_METHODS = new Set(["GET", "HEAD", "OPTIONS", undefined]);

function getStoredToken(): string {
  if (typeof window === "undefined") return "";
  try {
    return window.localStorage.getItem(TOKEN_KEY) ?? "";
  } catch {
    return "";
  }
}

// "Is this client likely an admin?" — based on token presence in localStorage.
// The UI uses this to decide whether to render admin-only affordances (e.g.
// the live camera preview on /regions, the Pin button). A wrong token still
// 401s on the actual request, so this is a UX hint, not an auth boundary.
export function hasAdminToken(): boolean {
  return getStoredToken().length > 0;
}

function setStoredToken(value: string) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(TOKEN_KEY, value);
  } catch {
    /* private mode / quota */
  }
}

function clearStoredToken() {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* ignore */
  }
}

// 401 recovery shared by every token-bearing request: drop the stale token
// (so hasAdminToken() stops claiming admin) and ask for a fresh one.
// Returns the trimmed candidate, or "" if the user cancelled.
function promptForNewToken(message: string): string {
  clearStoredToken();
  const entered = window.prompt(message, "");
  return entered?.trim() ?? "";
}

function buildHeaders(init: RequestInit | undefined, token: string): HeadersInit {
  const headers: Record<string, string> = {
    Accept: "application/json",
    ...(init?.body ? { "Content-Type": "application/json" } : {}),
    ...((init?.headers as Record<string, string>) ?? {}),
  };
  if (token) headers["X-Admin-Token"] = token;
  return headers;
}

async function jsonFetch<T>(url: string, init?: RequestInit): Promise<T> {
  const isWrite = !READ_METHODS.has(init?.method?.toUpperCase());
  const token = isWrite ? getStoredToken() : "";
  let res = await fetch(url, { ...init, headers: buildHeaders(init, token) });

  // One-shot recovery: if a write came back 401, prompt for a token and
  // retry exactly once. The entered token is persisted ONLY after the retry
  // gets past the auth gate — storing it up front would make hasAdminToken()
  // report admin with a garbage token, flipping the UI into admin mode that
  // can never work (e.g. /regions swaps the public pinned image for a live
  // snapshot fetch that 401s forever).
  if (res.status === 401 && isWrite && typeof window !== "undefined") {
    const candidate = promptForNewToken(
      "This action needs the admin token. Enter it to continue:",
    );
    if (candidate.length > 0) {
      res = await fetch(url, {
        ...init,
        headers: buildHeaders(init, candidate),
      });
      if (res.status === 401) {
        throw new Error("Wrong admin token — the action was not performed.");
      }
      // Any non-401 means the token passed the gate (even if the request
      // itself failed validation), so it's safe to remember.
      setStoredToken(candidate);
    }
  }

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}: ${text || url}`);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export const api = {
  stats: () => jsonFetch<DetectionStats>("/api/stats"),
  systemStatus: () => jsonFetch<SystemStatus>("/api/system/status"),
  species: (opts: { withDetections?: boolean } = {}) => {
    const qs = new URLSearchParams({ limit: "1000" });
    if (opts.withDetections) qs.set("with_detections", "true");
    return jsonFetch<Species[]>(`/api/species?${qs}`);
  },
  weather: () => jsonFetch<Weather>("/api/weather/current"),

  detections: (params: Record<string, string | number | undefined>) => {
    const qs = buildQuery(params);
    return jsonFetch<Detection[]>(`/api/detections?${qs}`);
  },

  detectionsCount: async (params: Record<string, string | number | undefined>) => {
    const qs = buildQuery(params);
    const res = await jsonFetch<{ count: number }>(`/api/detections/count?${qs}`);
    return res.count;
  },

  nextPending: () =>
    jsonFetch<Detection[]>("/api/detections?reviewed=pending&limit=1"),

  detection: (id: number) => jsonFetch<Detection>(`/api/detections/${id}`),

  review: (id: number, body: { is_false_positive?: boolean; corrected_species_id?: number | null }) =>
    jsonFetch<unknown>(`/api/detections/${id}/review`, {
      method: "PATCH",
      body: JSON.stringify(body),
    }),

  deleteDetection: (id: number) =>
    jsonFetch<unknown>(`/api/detections/${id}`, { method: "DELETE" }),

  bulkDelete: (ids: number[]) =>
    jsonFetch<BulkDeleteResponse>("/api/detections/bulk-delete", {
      method: "POST",
      body: JSON.stringify({ ids }),
    }),

  features: () => jsonFetch<FeatureFlags>("/api/features"),

  // Admin-only: upload the bytes of the live frame the admin is currently
  // looking at, so what gets pinned is exactly what they saw (no ~1.5s race
  // against the pipeline's snapshot writer).
  pinCameraFrame: (jpegBlob: Blob) =>
    jsonFetch<{ ok: boolean; size_bytes: number }>("/api/camera/pin", {
      method: "POST",
      body: jpegBlob,
      headers: { "Content-Type": "image/jpeg" },
    }),

  // Admin-only: fetch the live camera snapshot with the admin token in a
  // header (browsers don't send headers on plain <img src=...> requests).
  // Returns the raw Blob so the caller can both display it (via object URL)
  // and re-upload it for pin-exactly-what-I-see.
  fetchLiveSnapshotBlob: async (): Promise<Blob> => {
    const token = getStoredToken();
    let res = await fetch(`/api/camera/snapshot?t=${Date.now()}`, {
      headers: token ? { "X-Admin-Token": token } : {},
      cache: "no-store",
    });
    // Token rotated / revoked server-side: same recovery as writes — prompt
    // once for a fresh token. On cancel or another 401 the stored token stays
    // cleared, so callers can fall back to public (pinned-image) mode.
    if (res.status === 401 && typeof window !== "undefined") {
      const candidate = promptForNewToken(
        "Your admin token is no longer valid. Enter a new one to keep the live view:",
      );
      if (candidate.length > 0) {
        res = await fetch(`/api/camera/snapshot?t=${Date.now()}`, {
          headers: { "X-Admin-Token": candidate },
          cache: "no-store",
        });
        if (res.status === 401) {
          throw new Error("Wrong admin token.");
        }
        setStoredToken(candidate);
      }
    }
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`${res.status} ${res.statusText}: ${text || "snapshot"}`);
    }
    return await res.blob();
  },

  ignoreRegions: {
    list: () => jsonFetch<IgnoreRegion[]>("/api/ignore-regions"),
    create: (body: IgnoreRegionCreate) =>
      jsonFetch<IgnoreRegion>("/api/ignore-regions", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    update: (id: number, body: IgnoreRegionUpdate) =>
      jsonFetch<IgnoreRegion>(`/api/ignore-regions/${id}`, {
        method: "PATCH",
        body: JSON.stringify(body),
      }),
    remove: (id: number) =>
      jsonFetch<unknown>(`/api/ignore-regions/${id}`, { method: "DELETE" }),
    settings: () =>
      jsonFetch<IgnoreSettings>("/api/ignore-regions/settings"),
    updateSettings: (body: IgnoreSettings) =>
      jsonFetch<IgnoreSettings>("/api/ignore-regions/settings", {
        method: "PATCH",
        body: JSON.stringify(body),
      }),
  },

  pendingCount: async (): Promise<{ pending: number; total: number }> => {
    const stats = await api.stats();
    const pending = await jsonFetch<Detection[]>(
      "/api/detections?reviewed=pending&limit=1"
    );
    // Quick sniff — the count endpoint isn't separate, so we approximate:
    // we'll instead count by querying with a higher limit if needed elsewhere.
    return { pending: pending.length, total: stats.total_detections };
  },
};

export const imageUrl = {
  crop: (id: number) => `/api/detections/${id}/crop`,
  frame: (id: number) => `/api/detections/${id}/frame`,
  annotated: (id: number) => `/api/detections/${id}/annotated`,
  // Cache-busted snapshot for the /regions page so the browser refetches
  // when the user hits the refresh button. Admin-only — anonymous viewers
  // see imageUrl.pinned() instead.
  snapshot: (cacheBuster?: string | number) =>
    `/api/camera/snapshot${cacheBuster !== undefined ? `?t=${cacheBuster}` : ""}`,
  // Public-facing pinned frame for /regions. Updated only when an admin
  // POSTs /api/camera/pin, so a public viewer never sees the live feed.
  pinned: (cacheBuster?: string | number) =>
    `/api/camera/pinned${cacheBuster !== undefined ? `?t=${cacheBuster}` : ""}`,
};
