const API_BASE = "http://127.0.0.1:8000";
const STATIC_BASE = `${API_BASE}/static`;

function staticUrl(relPath) {
  if (!relPath) return "";
  return `${STATIC_BASE}/${relPath}`;
}

const Api = {
  async health() {
    const r = await fetch(`${API_BASE}/api/health`);
    return r.json();
  },

  async generateSyntheticPair(params = {}) {
    const form = new FormData();
    Object.entries(params).forEach(([k, v]) => {
      if (v !== undefined && v !== null && v !== "") form.append(k, v);
    });
    const r = await fetch(`${API_BASE}/api/analyze/synthetic-pair`, { method: "POST", body: form });
    if (!r.ok) throw new Error(await r.text());
    return r.json();
  },

  async runAnalysis({ sourceSensor, referenceSensor, detector, isSynthetic, syntheticPairId, sourceFile, referenceFile }) {
    const form = new FormData();
    form.append("source_sensor", sourceSensor);
    form.append("reference_sensor", referenceSensor);
    form.append("detector", detector);
    form.append("is_synthetic", isSynthetic ? "true" : "false");
    if (isSynthetic && syntheticPairId) {
      form.append("synthetic_pair_id", syntheticPairId);
    } else {
      form.append("source_image", sourceFile);
      form.append("reference_image", referenceFile);
    }
    const r = await fetch(`${API_BASE}/api/analyze/run`, { method: "POST", body: form });
    if (!r.ok) throw new Error(await r.text());
    return r.json();
  },

  async saveObservation(payload) {
    const r = await fetch(`${API_BASE}/api/observations/save`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!r.ok) throw new Error(await r.text());
    return r.json();
  },

  async listObservations() {
    const r = await fetch(`${API_BASE}/api/observations`);
    return r.json();
  },

  async getObservation(id) {
    const r = await fetch(`${API_BASE}/api/observations/${id}`);
    return r.json();
  },

  async buildMosaic(observationIds) {
    const r = await fetch(`${API_BASE}/api/mosaic/build`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ observation_ids: observationIds }),
    });
    if (!r.ok) throw new Error(await r.text());
    return r.json();
  },
};
