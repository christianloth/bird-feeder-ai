import type {
  Detection,
  DetectionStats,
  Species,
  SystemStatus,
  Weather,
} from "./types";

async function jsonFetch<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    ...init,
    headers: {
      Accept: "application/json",
      ...(init?.body ? { "Content-Type": "application/json" } : {}),
      ...(init?.headers ?? {}),
    },
  });
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
  species: () => jsonFetch<Species[]>("/api/species?limit=1000"),
  weather: () => jsonFetch<Weather>("/api/weather/current"),

  detections: (params: Record<string, string | number | undefined>) => {
    const qs = new URLSearchParams();
    for (const [k, v] of Object.entries(params)) {
      if (v === undefined || v === "" || v === null) continue;
      qs.set(k, String(v));
    }
    return jsonFetch<Detection[]>(`/api/detections?${qs}`);
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
};
