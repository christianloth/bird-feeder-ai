"use client";

import type { Species, ReviewFilter } from "@/lib/types";

export interface FilterValues {
  species_id: string;
  since: string;
  until: string;
  min_confidence: string;
  reviewed: ReviewFilter;
}

export const EMPTY_FILTERS: FilterValues = {
  species_id: "",
  since: "",
  until: "",
  min_confidence: "",
  reviewed: "",
};

interface Props {
  values: FilterValues;
  onChange: (next: FilterValues) => void;
  species: Species[];
}

export function Filters({ values, onChange, species }: Props) {
  const upd = <K extends keyof FilterValues>(k: K, v: FilterValues[K]) =>
    onChange({ ...values, [k]: v });

  const isDirty = Object.values(values).some((v) => v !== "");

  return (
    <section className="glass mb-6 rounded-[var(--radius-card)] p-5">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="eyebrow">Filter the field log</h2>
        <button
          type="button"
          onClick={() => onChange(EMPTY_FILTERS)}
          disabled={!isDirty}
          className="text-[0.75rem] text-[var(--color-sage-200)] underline-offset-2 transition hover:text-[var(--color-ember-400)] hover:underline disabled:cursor-default disabled:opacity-30 disabled:hover:no-underline"
        >
          reset all
        </button>
      </div>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-5">
        <Field label="Species">
          <select
            className="input-quiet"
            value={values.species_id}
            onChange={(e) => upd("species_id", e.target.value)}
          >
            <option value="">All species</option>
            {species.map((sp) => (
              <option key={sp.id} value={sp.id}>
                {sp.common_name}
              </option>
            ))}
          </select>
        </Field>

        <Field label="Since">
          <input
            type="date"
            className="input-quiet"
            value={values.since}
            onChange={(e) => upd("since", e.target.value)}
          />
        </Field>

        <Field label="Until">
          <input
            type="date"
            className="input-quiet"
            value={values.until}
            onChange={(e) => upd("until", e.target.value)}
          />
        </Field>

        <Field label="Min confidence">
          <select
            className="input-quiet"
            value={values.min_confidence}
            onChange={(e) => upd("min_confidence", e.target.value)}
          >
            <option value="">Any</option>
            <option value="0.9">≥ 90%</option>
            <option value="0.8">≥ 80%</option>
            <option value="0.7">≥ 70%</option>
            <option value="0.5">≥ 50%</option>
          </select>
        </Field>

        <Field label="Status">
          <select
            className="input-quiet"
            value={values.reviewed}
            onChange={(e) => upd("reviewed", e.target.value as ReviewFilter)}
          >
            <option value="">All</option>
            <option value="pending">Pending review</option>
            <option value="reviewed">Confirmed</option>
            <option value="false_positive">False positives</option>
          </select>
        </Field>
      </div>
    </section>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="eyebrow">{label}</span>
      {children}
    </label>
  );
}
