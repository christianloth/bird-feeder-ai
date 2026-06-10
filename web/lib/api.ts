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

  // One-shot recovery: if a write came back 401, prompt for a token,
  // store it, and retry exactly once. Wrong token on retry throws.
  if (res.status === 401 && isWrite && typeof window !== "undefined") {
    clearStoredToken();
    const entered = window.prompt(
      "This action needs the admin token. Enter it to continue:",
      "",
    );
    if (entered && entered.trim().length > 0) {
      setStoredToken(entered.trim());
      res = await fetch(url, {
        ...init,
        headers: buildHeaders(init, entered.trim()),
      });
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

  // Admin-only: pin the current live frame so the public /regions canvas shows it.
  pinCameraFrame: () =>
    jsonFetch<{ ok: boolean; size_bytes: number }>("/api/camera/pin", {
      method: "POST",
    }),

  // Admin-only: fetch the live camera snapshot with the admin token in a header
  // (browsers don't send headers on plain <img src=...> requests). Returns an
  // object URL the caller must revoke when replaced/unmounted to free memory.
  fetchLiveSnapshotBlobUrl: async (): Promise<string> => {
    const token = getStoredToken();
    const res = await fetch(`/api/camera/snapshot?t=${Date.now()}`, {
      headers: token ? { "X-Admin-Token": token } : {},
      cache: "no-store",
    });
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`${res.status} ${res.statusText}: ${text || "snapshot"}`);
    }
    const blob = await res.blob();
    return URL.createObjectURL(blob);
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
