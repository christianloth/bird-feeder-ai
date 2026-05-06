export interface Species {
  id: number;
  common_name: string;
  scientific_name: string;
  family: string | null;
  class_index: number;
}

export interface Detection {
  id: number;
  timestamp: string;
  species_id: number | null;
  species_name: string | null;
  confidence: number;
  detection_model: string | null;
  classifier_model: string | null;
  bbox_x1: number | null;
  bbox_y1: number | null;
  bbox_x2: number | null;
  bbox_y2: number | null;
  frame_path: string | null;
  reviewed: boolean;
  is_false_positive: boolean;
  corrected_species_id: number | null;
  corrected_species_name: string | null;
  source: string | null;
}

export interface DetectionStats {
  total_detections: number;
  unique_species: number;
  most_common_species: string | null;
  most_common_count: number;
  detections_today: number;
  avg_confidence: number;
}

export interface SystemStatus {
  camera_connected: boolean;
  pipeline_running: boolean;
  total_detections: number;
  disk_usage_mb: number;
  uptime_seconds: number;
}

export interface Weather {
  temperature_c: number | null;
  humidity_pct: number | null;
  wind_speed_kmh: number | null;
  precipitation_mm: number | null;
  cloud_cover_pct: number | null;
  weather_code: number | null;
  weather_description: string | null;
  timestamp: string | null;
}

export type ReviewFilter = "" | "pending" | "reviewed" | "false_positive";

export interface DetectionsQuery {
  species_id?: number;
  since?: string;
  until?: string;
  min_confidence?: number;
  reviewed?: ReviewFilter;
  skip?: number;
  limit?: number;
  region_x1?: number;
  region_y1?: number;
  region_x2?: number;
  region_y2?: number;
  region_overlap?: number;
}

export interface IgnoreRegion {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  label: string;
  overlap_threshold: number;
}

export interface FeatureFlags {
  sweep: boolean;
}

export interface BulkDeleteResponse {
  deleted: number;
  not_found: number[];
}
