// REST API client
import { mockAnalyzeResponse } from './mockData';

// Initial trajectory baseline
export const DEMO_HISTORY = [
  { timestamp: "2026-08-25T00:00:00Z", latitude: 13.80, longitude: 80.10, wind_speed_kmh: 105, pressure_hpa: 986, sst: 28.2, wind_u: 10.0, wind_v: 7.0 },
  { timestamp: "2026-08-25T06:00:00Z", latitude: 14.35, longitude: 80.55, wind_speed_kmh: 114, pressure_hpa: 979, sst: 28.3, wind_u: 10.5, wind_v: 6.8 },
  { timestamp: "2026-08-25T12:00:00Z", latitude: 14.98, longitude: 81.10, wind_speed_kmh: 122, pressure_hpa: 971, sst: 28.5, wind_u: 11.2, wind_v: 6.5 },
  { timestamp: "2026-08-25T18:00:00Z", latitude: 15.62, longitude: 81.58, wind_speed_kmh: 133, pressure_hpa: 962, sst: 28.5, wind_u: 11.5, wind_v: 6.6 },
  { timestamp: "2026-08-26T00:00:00Z", latitude: 16.52, longitude: 82.31, wind_speed_kmh: 145, pressure_hpa: 950, sst: 28.6, wind_u: 11.8, wind_v: 6.7 },
];

// Target storm metadata
export const DEMO_META = {
  systemId: "USER-TRACK",
  systemName: "Custom Track Simulation",
  basin: "Bay of Bengal",
  lastPass: "2026-08-26T05:30:00Z",
};

export const USE_MOCK = false;
const API_BASE = '/api';

function wait(ms){ return new Promise(res => setTimeout(res, ms)); }

// JSON request dispatcher
async function postJSON(path, body){
  const res = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if(!res.ok){
    let detail = `HTTP ${res.status}`;
    try{
      const err = await res.json();
      if(err?.error?.message) detail = `${err.error.code} · ${err.error.message}`;
    }catch(_){ /* non-JSON body */ }
    throw new Error(`Request to ${path} failed: ${detail}`);
  }
  return res.json();
}

// Satellite image upload
export async function uploadSatelliteImage(file){
  if(USE_MOCK){
    await wait(200);
    return {
      status: 'success',
      detection: mockAnalyzeResponse.detection,
      classification: mockAnalyzeResponse.classification,
      satellite: mockAnalyzeResponse.satellite,
      provenance: {
        ...mockAnalyzeResponse.provenance,
        image_source: 'USER-UPLOADED IMAGE',
        notes: ['image_source: USER-UPLOADED IMAGE (mock)'],
      },
      sourceLabel: 'USER-UPLOADED IMAGE',
    };
  }
  const fd = new FormData();
  fd.append('file', file);
  const res = await fetch(`${API_BASE}/image`, { method: 'POST', body: fd });
  if(!res.ok){
    let detail = `HTTP ${res.status}`;
    try{
      const err = await res.json();
      if(err?.error?.message) detail = `${err.error.code} · ${err.error.message}`;
    }catch(_){ /* non-JSON body */ }
    throw new Error(`Image upload to /api/image failed: ${detail}`);
  }
  const body = await res.json();
  return { ...body, sourceLabel: body?.provenance?.image_source || 'USER-UPLOADED IMAGE' };
}

// Satellite detection query
export async function fetchDetect(history, meta){
  if(USE_MOCK){ await wait(150); return mockAnalyzeResponse.detection; }
  const payload = {};
  if(history && history.length > 0) payload.history = history;
  if(meta) payload.meta = meta;
  return postJSON('/detect', payload);
}

// Intensity classification query
export async function fetchClassify(history, meta){
  if(USE_MOCK){ await wait(150); return mockAnalyzeResponse.classification; }
  const payload = {};
  if(history && history.length > 0) payload.history = history;
  if(meta) payload.meta = meta;
  return postJSON('/classify', payload);
}

// Track forecast query
export async function fetchForecast(history, meta){
  if(USE_MOCK){ await wait(150); return mockAnalyzeResponse.forecast; }
  const payload = {};
  if(history && history.length > 0) payload.history = history;
  if(meta) payload.meta = meta;
  const body = await postJSON('/forecast', payload);
  return body.forecast;
}

// System health check
export async function fetchHealth(){
  const res = await fetch(`${API_BASE}/health`);
  if(!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return res.json();
}

// Unified multi-model analysis
export async function fetchAnalyze(history, meta){
  if(USE_MOCK){
    await wait(300);
    return mockAnalyzeResponse;
  }
  const payload = {};
  if(history && history.length > 0) payload.history = history;
  if(meta) payload.meta = meta;
  return await postJSON('/analyze', payload);
}