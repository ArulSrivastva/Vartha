# VARTHA — Tropical Cyclone Intelligence Dashboard

A React + Vite frontend for displaying the output of the AI/ML cyclone
identification, classification and prediction system. This project covers
**frontend/UI only** — no model training, no backend. It's built to consume
mock data now and switch to the real API with a one-line change later.

## Stack

- React 19 + Vite
- Tailwind CSS v4 (utility styling, custom theme tokens for the brand palette)
- Leaflet + react-leaflet (interactive map)
- Recharts (wind / pressure / confidence trend charts)

## Getting started

```bash
npm install
npm run dev
```

Open the URL Vite prints (usually `http://localhost:5173`).

To build for production:

```bash
npm run build
npm run preview   # serve the production build locally
```

## Project structure

```
src/
  api/
    mockData.js      — dummy dataset shaped exactly like the real API response
    client.js         — fetch functions (fetchDetect, fetchClassify,
                         fetchForecast, fetchAnalyze) — mock now, real later
  components/
    Header.jsx         — title + near-real-time status line
    CardShell.jsx       — shared card layout used by every panel
    DetectionCard.jsx
    ClassificationCard.jsx
    ForecastCard.jsx
    LandfallPanel.jsx
    RiskIndicator.jsx   — 0–100 gauge with HIGH/MEDIUM/LOW band
    SatelliteViewer.jsx — upload/zoom/bounding-box viewer
    map/
      MapView.jsx        — Leaflet map: historical/current/forecast + cone
      uncertaintyCone.js  — forecast-cone polygon geometry helper
    charts/
      MiniLineChart.jsx
      ChartsPanel.jsx
  lib/
    format.js          — date formatting, risk-band color helpers
  App.jsx               — fetches data once, passes it down to every panel
  index.css             — Tailwind import + design tokens
```

## Swapping mock data for the real backend

Everything in the UI reads from the return value of `fetchAnalyze()` in
`src/api/client.js` — no component imports `mockData.js` directly except
that client file. To go live:

1. Open `src/api/client.js`.
2. Set `USE_MOCK = false`.
3. Point `API_BASE` at your backend (e.g. `''` if served from the same
   origin, or a full URL).
4. Make sure `POST /api/analyze` (or `GET`, adjust as needed) returns JSON
   shaped like this:

```json
{
  "detection": {
    "detected": true,
    "confidence": 94,
    "location": { "lat": 16.52, "lon": 82.31 }
  },
  "classification": {
    "category": "Severe Cyclonic Storm",
    "scale": "IMD",
    "windSpeedKmh": 145,
    "pressureHpa": 950,
    "confidence": 91,
    "structuralPattern": "Eye visible"
  },
  "forecast": [
    { "hour": 6, "label": "+6h", "lat": 17.10, "lon": 83.02, "windSpeedKmh": 150, "pressureHpa": 946, "confidence": 89 },
    { "hour": 12, "label": "+12h", "...": "..." },
    { "hour": 24, "label": "+24h", "...": "..." }
  ],
  "landfall": {
    "estimated": true,
    "latitude": 18.42,
    "longitude": 84.91,
    "estimated_time": "2026-08-27T02:00:00Z",
    "predictedWindKmh": 135
  },
  "risk": { "score": 82, "level": "HIGH" }
}
```

That's the exact contract given in the project brief. The dashboard maps
each field as follows:

| Response field    | UI destination                                    |
|--------------------|-----------------------------------------------------|
| `detection`        | Detection card + current position marker on map    |
| `classification`   | Classification card                                 |
| `forecast`          | Forecast card + forecast track/markers on map        |
| `landfall`           | Landfall panel + landfall marker on map               |
| `risk`               | AI Risk Indicator gauge                                |

### Supplementary fields (not in the official contract)

The map's historical track and the three trend charts need a bit more data
than the four fields above provide. For now these live alongside the mock
response as `historicalTrack`, `windHistory`, `pressureHistory`,
`confidenceHistory`, `meta` (system name/basin/last-pass timestamp for the
header) and `satellite` (bounding box for the image viewer). When the real
`/api/analyze` is ready, either fold these into that same response or fetch
them from separate endpoints and merge client-side in `fetchAnalyze()` —
no other file needs to change either way.

### Dev-time separate endpoints

`fetchDetect()`, `fetchClassify()` and `fetchForecast()` in `client.js` are
already wired to call `/api/detect`, `/api/classify`, `/api/forecast`
respectively once `USE_MOCK` is off, in case those land before the combined
`/api/analyze` endpoint does. `fetchAnalyze()` will automatically fall back
to combining them client-side if a direct call to `/api/analyze` fails.

## Notes on specific panels

- **Map** — historical track (dashed grey), forecast track (dashed rose)
  with per-point markers colored by wind-speed band, a landfall marker, and
  a widening "cone of uncertainty" polygon around the forecast track. The
  cone's width-per-hour is currently a placeholder formula in
  `uncertaintyCone.js` — replace with real per-hour uncertainty radii from
  the backend when available.
- **Satellite viewer** — supports image upload (stores an object URL,
  nothing is sent anywhere) and zoom via buttons. A synthetic placeholder
  graphic is shown until an image is uploaded, since no real satellite
  imagery is wired up yet. The bounding box is drawn from normalized (0–1)
  coordinates so it scales correctly with zoom and with whatever image is
  loaded — swap `satellite.boundingBox` for real model output later.
- **Risk indicator** — pure UI, doesn't compute risk itself. Displays
  whatever `score`/`level` the backend sends, with the required
  "AI-derived risk indicator — not an official warning" disclaimer always
  visible.
