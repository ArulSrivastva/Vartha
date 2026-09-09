// Mock data shaped to match the final /api/analyze response contract exactly
// (see api/client.js). When the backend is ready, swap USE_MOCK to false and
// nothing else in the UI needs to change — every component reads this same
// shape whether it comes from here or from a live fetch.

export const mockAnalyzeResponse = {
  meta: {
    systemId: "USER-TRACK",
    systemName: "Custom Track Simulation",
    basin: "Bay of Bengal",
    // mock "latest available satellite pass" timestamp for the near-real-time
    // status line in the header
    lastPass: "2026-08-26T05:30:00Z",
    source: "INSAT-3D IR · illustrative sample data",
  },

  detection: {
    detected: true,
    confidence: 94,
    location: { lat: 16.52, lon: 82.31 },
    movementDirection: "North-West (315°)",
    movementSpeedKmh: 14,
  },

  classification: {
    category: "Severe Cyclonic Storm",
    scale: "IMD",
    windSpeedKmh: 145,
    pressureHpa: 950,
    confidence: 91,
    structuralPattern: "Eye visible",
  },

  // Projected future positions only — the current position comes from
  // `detection.location` above, not from this array (see MapView, which
  // stitches detection.location + this array together for the drawn track).
  forecast: [
    { hour: 6,  label: "+6h",  lat: 17.10, lon: 83.02, windSpeedKmh: 150, pressureHpa: 946, confidence: 89 },
    { hour: 12, label: "+12h", lat: 17.78, lon: 83.74, windSpeedKmh: 152, pressureHpa: 944, confidence: 83 },
    { hour: 24, label: "+24h", lat: 18.42, lon: 84.91, windSpeedKmh: 138, pressureHpa: 958, confidence: 71 },
  ],

  landfall: {
    estimated: true,
    latitude: 18.42,
    longitude: 84.91,
    estimated_time: "2026-08-27T02:00:00Z",
    predictedWindKmh: 135,
    distanceToLandKm: 12,
  },

  risk: { score: 82, level: "HIGH" },

  // --- supplementary fields used by the map / charts / satellite viewer ---
  // Not part of the official /api/analyze contract above, but kept in the
  // same object so the whole dashboard can be driven from one response.
  historicalTrack: [
    { lat: 13.80, lon: 80.10, timestamp: "2026-08-24T06:00:00Z" },
    { lat: 14.35, lon: 80.55, timestamp: "2026-08-24T12:00:00Z" },
    { lat: 14.98, lon: 81.10, timestamp: "2026-08-24T18:00:00Z" },
    { lat: 15.62, lon: 81.58, timestamp: "2026-08-25T00:00:00Z" },
    { lat: 16.05, lon: 81.92, timestamp: "2026-08-25T12:00:00Z" },
    { lat: 16.52, lon: 82.31, timestamp: "2026-08-26T05:30:00Z" },
  ],

  windHistory: [
    { t: "-24h", value: 105 }, { t: "-18h", value: 114 }, { t: "-12h", value: 122 },
    { t: "-6h",  value: 133 }, { t: "Now",  value: 145 },
  ],
  pressureHistory: [
    { t: "-24h", value: 986 }, { t: "-18h", value: 979 }, { t: "-12h", value: 971 },
    { t: "-6h",  value: 962 }, { t: "Now",  value: 950 },
  ],
  confidenceHistory: [
    { t: "-24h", value: 78 }, { t: "-18h", value: 82 }, { t: "-12h", value: 87 },
    { t: "-6h",  value: 90 }, { t: "Now",  value: 94 },
  ],
  sstHistory: [
    { t: "-24h", value: 28.2 }, { t: "-18h", value: 28.3 }, { t: "-12h", value: 28.5 },
    { t: "-6h",  value: 28.5 }, { t: "Now",  value: 28.6 },
  ],
  envWindHistory: [
    { t: "-24h", value: 12.1 }, { t: "-18h", value: 12.4 }, { t: "-12h", value: 13.0 },
    { t: "-6h",  value: 13.2 }, { t: "Now",  value: 13.5 },
  ],

  satellite: {
    label: "INSAT-3D IR pass · illustrative sample",
    // normalized (0-1) bounding box over the sample image, so it scales with
    // whatever image is displayed — swap for real model output later
    boundingBox: { x: 0.34, y: 0.28, w: 0.32, h: 0.34, confidence: 94 },
  },
};

// Wind-speed → category colour scale, reused by the map, satellite bbox and
// classification card so intensity is colour-coded consistently everywhere.
export const windScale = [
  { max: 40,  color: "#7A9A7E", label: "Depression" },
  { max: 60,  color: "#A9B268", label: "Deep Depression" },
  { max: 90,  color: "#C7A05C", label: "Cyclonic Storm" },
  { max: 120, color: "#C7855C", label: "Severe Cyclonic Storm" },
  { max: 220, color: "#B85C4C", label: "Very / Extremely Severe" },
];

export function colorForWind(kmh){
  return windScale.find(b => kmh <= b.max) || windScale[windScale.length - 1];
}
