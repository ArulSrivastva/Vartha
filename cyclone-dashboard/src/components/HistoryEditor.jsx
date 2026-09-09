import { useMemo, useState, useRef } from 'react';
import CardShell from './CardShell';

const FIELDS = [
  { key: 'timestamp', label: 'Timestamp (ISO-UTC)', type: 'text' },
  { key: 'latitude', label: 'Lat', type: 'number' },
  { key: 'longitude', label: 'Lon', type: 'number' },
  { key: 'wind_speed_kmh', label: 'Wind km/h', type: 'number' },
  { key: 'pressure_hpa', label: 'Pressure hPa', type: 'number' },
  { key: 'sst', label: 'SST °C', type: 'number' },
  { key: 'wind_u', label: 'Wind-u', type: 'number' },
  { key: 'wind_v', label: 'Wind-v', type: 'number' },
];

function clone(rows){ return rows.map(r => ({ ...r })); }

function blankRow(idx = 0){
  const base = new Date();
  base.setHours(base.getHours() - (4 - idx) * 6);
  const ts = base.toISOString().substring(0, 14) + "00:00Z";
  return {
    timestamp: ts,
    latitude: 14.0 + idx * 0.7,
    longitude: 82.0 + idx * 0.4,
    wind_speed_kmh: 80 + idx * 15,
    pressure_hpa: 990 - idx * 8,
    sst: 29.0,
    wind_u: 8.0,
    wind_v: 10.0,
  };
}

function parseTrackCSV(text){
  const lines = text.trim().split(/\r?\n/).filter(line => line.trim().length > 0);
  if(lines.length < 2){
    throw new Error('CSV must contain a header row and at least 1 track row.');
  }

  const rawHeaders = lines[0].split(',').map(h => h.trim().toLowerCase().replace(/['"]/g, ''));
  const headerMap = {};
  rawHeaders.forEach((h, idx) => {
    if(['timestamp', 'time', 'date', 'datetime', 'iso_time', 'utc_time'].includes(h)){
      headerMap.timestamp = idx;
    } else if(['latitude', 'lat'].includes(h)){
      headerMap.latitude = idx;
    } else if(['longitude', 'lon', 'long', 'lng'].includes(h)){
      headerMap.longitude = idx;
    } else if(['wind_speed_kmh', 'wind', 'wind_speed', 'wind_kmh', 'vmax', 'wind_speed_km_h'].includes(h)){
      headerMap.wind_speed_kmh = idx;
    } else if(['pressure_hpa', 'pressure', 'mslp', 'pres', 'central_pressure'].includes(h)){
      headerMap.pressure_hpa = idx;
    } else if(['sst', 'sea_surface_temp', 'sea_temp'].includes(h)){
      headerMap.sst = idx;
    } else if(['wind_u', 'u', 'u10'].includes(h)){
      headerMap.wind_u = idx;
    } else if(['wind_v', 'v', 'v10'].includes(h)){
      headerMap.wind_v = idx;
    }
  });

  if(headerMap.latitude === undefined || headerMap.longitude === undefined){
    throw new Error('CSV must contain "latitude" (or "lat") and "longitude" (or "lon") columns.');
  }

  const dataRows = lines.slice(1);
  const parsed = [];

  for(let i = 0; i < dataRows.length; i++){
    const cols = dataRows[i].split(',').map(c => c.trim().replace(/^["']|["']$/g, ''));
    if(cols.length <= 1) continue;

    const lat = parseFloat(cols[headerMap.latitude]);
    const lon = parseFloat(cols[headerMap.longitude]);
    if(Number.isNaN(lat) || Number.isNaN(lon)) continue;

    let ts = headerMap.timestamp !== undefined ? cols[headerMap.timestamp] : '';
    if(!ts || Number.isNaN(Date.parse(ts))){
      const now = new Date();
      now.setHours(now.getHours() - (dataRows.length - 1 - i) * 6);
      ts = now.toISOString().substring(0, 14) + "00:00Z";
    }

    const wind = (headerMap.wind_speed_kmh !== undefined && !Number.isNaN(parseFloat(cols[headerMap.wind_speed_kmh])))
      ? parseFloat(cols[headerMap.wind_speed_kmh])
      : 120.0;

    const pres = (headerMap.pressure_hpa !== undefined && !Number.isNaN(parseFloat(cols[headerMap.pressure_hpa])))
      ? parseFloat(cols[headerMap.pressure_hpa])
      : 960.0;

    const sst = (headerMap.sst !== undefined && !Number.isNaN(parseFloat(cols[headerMap.sst])))
      ? parseFloat(cols[headerMap.sst])
      : 29.0;

    const wind_u = (headerMap.wind_u !== undefined && !Number.isNaN(parseFloat(cols[headerMap.wind_u])))
      ? parseFloat(cols[headerMap.wind_u])
      : 10.0;

    const wind_v = (headerMap.wind_v !== undefined && !Number.isNaN(parseFloat(cols[headerMap.wind_v])))
      ? parseFloat(cols[headerMap.wind_v])
      : 8.0;

    parsed.push({
      timestamp: ts,
      latitude: lat,
      longitude: lon,
      wind_speed_kmh: wind,
      pressure_hpa: pres,
      sst: sst,
      wind_u: wind_u,
      wind_v: wind_v,
    });
  }

  if(parsed.length === 0){
    throw new Error('No valid track coordinates found in the uploaded CSV.');
  }

  // Ensure sequence has exactly 5 rows (operational TC-AI 5-observation window)
  let finalRows = parsed;
  if(finalRows.length > 5){
    finalRows = finalRows.slice(-5);
  } else if(finalRows.length < 5){
    while(finalRows.length < 5){
      const first = finalRows[0];
      const prevDate = new Date(Date.parse(first.timestamp) - 6 * 3600000);
      const prevRow = {
        ...first,
        timestamp: prevDate.toISOString().substring(0, 14) + "00:00Z",
        latitude: parseFloat((first.latitude - 0.5).toFixed(2)),
        longitude: parseFloat((first.longitude - 0.3).toFixed(2)),
        wind_speed_kmh: Math.max(40, first.wind_speed_kmh - 10),
        pressure_hpa: Math.min(1005, first.pressure_hpa + 5),
      };
      finalRows.unshift(prevRow);
    }
  }

  // Normalize timestamps to exact 6h increments required by causal models
  for(let i = 1; i < finalRows.length; i++){
    const prevTs = Date.parse(finalRows[i - 1].timestamp);
    const currTs = Date.parse(finalRows[i].timestamp);
    if(Number.isNaN(currTs) || Math.abs((currTs - prevTs) / 3600000 - 6) > 0.001){
      const expected = new Date(prevTs + 6 * 3600000);
      finalRows[i].timestamp = expected.toISOString().substring(0, 14) + "00:00Z";
    }
  }

  return finalRows;
}

function downloadSampleTrackCSV(){
  const content = [
    "timestamp,latitude,longitude,wind_speed_kmh,pressure_hpa,sst,wind_u,wind_v",
    "2026-08-25T00:00:00Z,14.0,82.0,80,990,29.0,8.0,10.0",
    "2026-08-25T06:00:00Z,14.7,82.4,95,982,29.1,8.5,9.5",
    "2026-08-25T12:00:00Z,15.4,82.8,110,974,29.2,9.0,9.0",
    "2026-08-25T18:00:00Z,16.1,83.2,125,965,29.1,9.5,8.5",
    "2026-08-26T00:00:00Z,16.8,83.6,140,955,29.0,10.0,8.0"
  ].join("\n");

  const blob = new Blob([content], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.setAttribute('download', 'cyclone_track_template.csv');
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function validate(rows){
  const errors = [];
  if(!Array.isArray(rows) || rows.length === 0){
    return [];
  }
  if(rows.length !== 5){
    return ['A track sequence must contain exactly 5 observations (found ' + (rows?.length || 0) + ').'];
  }
  rows.forEach((r, i) => {
    const ts = Date.parse(r.timestamp);
    if(Number.isNaN(ts)) errors.push(`Observation ${i + 1}: timestamp is not a valid ISO-UTC time.`);
    if(typeof r.latitude !== 'number' || !isFinite(r.latitude) || r.latitude < -90 || r.latitude > 90)
      errors.push(`Observation ${i + 1}: latitude must be between -90 and 90.`);
    if(typeof r.longitude !== 'number' || !isFinite(r.longitude) || r.longitude < -180 || r.longitude > 180)
      errors.push(`Observation ${i + 1}: longitude must be between -180 and 180.`);
    if(typeof r.wind_speed_kmh !== 'number' || !isFinite(r.wind_speed_kmh) || r.wind_speed_kmh < 0)
      errors.push(`Observation ${i + 1}: wind speed must be ≥ 0.`);
    if(typeof r.pressure_hpa !== 'number' || !isFinite(r.pressure_hpa) || r.pressure_hpa < 850 || r.pressure_hpa > 1050)
      errors.push(`Observation ${i + 1}: pressure must be between 850 and 1050 hPa.`);
    if(typeof r.sst !== 'number' || !isFinite(r.sst))
      errors.push(`Observation ${i + 1}: SST must be a number.`);
  });
  // chronological + 6h spacing
  for(let i = 1; i < rows.length; i++){
    const a = Date.parse(rows[i - 1].timestamp);
    const b = Date.parse(rows[i].timestamp);
    if(!Number.isNaN(a) && !Number.isNaN(b)){
      if(b <= a) errors.push(`Observations must be strictly chronological (obs ${i + 1} must be later than obs ${i}).`);
      else {
        const h = (b - a) / 3600000;
        if(Math.abs(h - 6) > 0.00001) errors.push(`Observations must be spaced 6h apart (obs ${i}→obs ${i + 1} is ${h}h).`);
      }
    }
  }
  return errors;
}

export default function HistoryEditor({ onAnalyze }){
  const [rows, setRows] = useState([]);
  const [showManualEditor, setShowManualEditor] = useState(false);
  const [dirty, setDirty] = useState(false);
  const [csvStatus, setCsvStatus] = useState(null);
  const csvInputRef = useRef(null);
  const errors = useMemo(() => validate(rows), [rows]);

  function updateCell(rowIdx, key, raw){
    setDirty(true);
    setRows(prev => {
      const next = clone(prev);
      let value;
      if(key === 'timestamp') value = raw;
      else if(raw.trim() === '') value = Number.NaN;
      else value = Number(raw);
      next[rowIdx] = { ...next[rowIdx], [key]: value };
      return next;
    });
  }

  function handleCSVUpload(e){
    const file = e.target.files?.[0];
    if(!file) return;
    const reader = new FileReader();
    reader.onload = (evt) => {
      try {
        const text = evt.target?.result;
        if(typeof text !== 'string') return;
        const parsed = parseTrackCSV(text);
        setRows(parsed);
        setDirty(true);
        setShowManualEditor(true);
        setCsvStatus({ type: 'success', message: `Parsed "${file.name}": 5 track observations loaded and ready for forecast.` });
      } catch (err) {
        setCsvStatus({ type: 'error', message: err.message || 'Failed to parse CSV file.' });
      } finally {
        if(csvInputRef.current) csvInputRef.current.value = '';
      }
    };
    reader.readAsText(file);
  }

  function resetToLive(){
    setRows([]);
    setDirty(false);
    setShowManualEditor(false);
    setCsvStatus(null);
    onAnalyze();
  }

  function createCustomTrack(){
    const initial = [0, 1, 2, 3, 4].map(blankRow);
    setRows(initial);
    setDirty(true);
    setShowManualEditor(true);
    setCsvStatus(null);
  }

  function addRow(){
    setDirty(true);
    setRows(prev => [...prev, blankRow(prev.length)]);
  }

  function deleteRow(rowIdx){
    setDirty(true);
    setRows(prev => prev.filter((_, i) => i !== rowIdx));
  }

  function clear(){
    setDirty(true);
    setRows([]);
    setCsvStatus(null);
  }

  function submit(){
    if(errors.length || rows.length !== 5) return;
    const hist = rows.map(r => ({
      timestamp: r.timestamp,
      latitude: r.latitude,
      longitude: r.longitude,
      wind_speed_kmh: r.wind_speed_kmh,
      pressure_hpa: r.pressure_hpa,
      sst: r.sst,
      wind_u: r.wind_u,
      wind_v: r.wind_v,
    }));
    onAnalyze(hist);
    setDirty(false);
  }

  return (
    <CardShell
      eyebrow="Input Telemetry &amp; Spatial Track"
      title="Cyclone Track &amp; Environmental Data"
      right={
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-[9.5px] font-bold text-ink-faint border border-border/60 bg-card-alt/40 rounded px-2 py-1 uppercase tracking-wider font-mono">
            {rows.length === 0 ? "LIVE TELEMETRY (MOSDAC)" : (dirty ? "CUSTOM SIMULATION" : "SIMULATION ACTIVE")}
          </span>

          {/* CSV File Input */}
          <input
            ref={csvInputRef}
            type="file"
            accept=".csv,text/csv"
            onChange={handleCSVUpload}
            className="hidden"
          />

          <button
            onClick={() => csvInputRef.current?.click()}
            className="text-[11.5px] font-semibold text-white bg-accent-strong rounded-lg px-3.5 py-1.5 hover:opacity-90 active:scale-95 transition-all shadow-sm cursor-pointer flex items-center gap-1"
          >
            Upload Track CSV
          </button>

          <button
            onClick={downloadSampleTrackCSV}
            title="Download formatted CSV track template"
            className="text-[11px] font-medium text-ink-soft hover:text-accent border border-border/70 bg-card rounded-lg px-2.5 py-1.5 hover:border-accent transition-all cursor-pointer"
          >
            Sample CSV
          </button>

          <button
            onClick={resetToLive}
            className="text-[11.5px] font-semibold text-ink border border-border bg-card rounded-lg px-3 py-1.5 hover:border-accent hover:bg-card-alt transition-all cursor-pointer"
          >
            Reset to Live
          </button>

          {rows.length > 0 && (
            <button
              onClick={submit}
              disabled={errors.length > 0 || rows.length !== 5}
              className="text-[11.5px] font-semibold text-white bg-accent-strong rounded-lg px-4 py-1.5 hover:opacity-90 active:scale-95 transition-all shadow-sm disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer flex items-center gap-1.5"
            >
              Run AI Track Forecast
            </button>
          )}
        </div>
      }
    >
      <div className="space-y-4">
        {/* CSV Status Feedback */}
        {csvStatus && (
          <div className={`text-[11.5px] rounded-lg px-3.5 py-2 flex items-center justify-between gap-2 border font-mono ${
            csvStatus.type === 'success' ? 'bg-risk-low/10 border-risk-low/30 text-risk-low' : 'bg-risk-high/10 border-risk-high/30 text-risk-high'
          }`}>
            <span>{csvStatus.message}</span>
            <button onClick={() => setCsvStatus(null)} className="text-[11px] font-bold opacity-70 hover:opacity-100 cursor-pointer">x</button>
          </div>
        )}

        {rows.length === 0 ? (
          <div className="bg-card-alt/40 border border-border/50 rounded-xl p-4 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
            <div>
              <div className="text-[13px] font-bold text-ink flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-risk-low" />
                Live Ingest Mode Active
              </div>
              <p className="text-[11.5px] text-ink-soft mt-0.5">
                Monitoring live ISRO MOSDAC feeds. To evaluate custom storm coordinates, upload a track CSV or initialize manual coordinates.
              </p>
            </div>
            <div className="flex items-center gap-2 shrink-0 flex-wrap">
              <button
                onClick={() => csvInputRef.current?.click()}
                className="text-[11.5px] font-bold text-white bg-accent-strong rounded-lg px-3.5 py-1.5 hover:opacity-90 transition-all cursor-pointer shadow-sm"
              >
                Upload Track CSV
              </button>
              <button
                onClick={createCustomTrack}
                className="text-[11.5px] font-semibold text-accent-strong border border-accent/40 bg-card rounded-lg px-3.5 py-1.5 hover:bg-accent-soft hover:border-accent transition-all cursor-pointer"
              >
                + Manual Track Table
              </button>
            </div>
          </div>
        ) : null}

        {/* Collapsible Manual History Editor */}
        <div className="border-t border-border/50 pt-3">
          <div className="flex items-center justify-between">
            <button
              onClick={() => setShowManualEditor(prev => !prev)}
              className="text-[11.5px] font-bold text-accent-strong hover:underline flex items-center gap-1.5 cursor-pointer"
            >
              <span>{showManualEditor ? 'v' : '>'}</span>
              <span>{showManualEditor ? 'Hide Advanced Coordinate Editor' : 'Advanced: Inspect & Edit Raw 5-Observation Timesteps'}</span>
            </button>

            {showManualEditor && (
              <div className="flex items-center gap-2">
                <button
                  onClick={addRow}
                  className="text-[10.5px] font-semibold text-ink border border-border bg-card rounded px-2.5 py-1 hover:border-accent"
                >
                  + Add observation
                </button>
                <button
                  onClick={clear}
                  className="text-[10.5px] font-semibold text-ink border border-border bg-card rounded px-2.5 py-1 hover:border-accent"
                >
                  Clear
                </button>
              </div>
            )}
          </div>

          {showManualEditor && (
            <div className="mt-3 space-y-3">
              <div className="overflow-x-auto">
                <table className="w-full text-[11px] border-collapse">
                  <thead>
                    <tr>
                      {FIELDS.map(f => (
                        <th key={f.key} className="text-left text-ink-faint font-semibold px-1.5 py-1 border-b border-border">{f.label}</th>
                      ))}
                      <th className="text-left text-ink-faint font-semibold px-1.5 py-1 border-b border-border">Del</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((row, i) => (
                      <tr key={i} className="align-top">
                        {FIELDS.map(f => (
                          <td key={f.key} className="px-1 py-1 border-b border-border/40">
                            <input
                              type={f.type}
                              step={f.type === 'number' ? 'any' : undefined}
                              value={f.key === 'timestamp' ? row[f.key] : (Number.isFinite(row[f.key]) ? String(row[f.key]) : '')}
                              onChange={e => updateCell(i, f.key, e.target.value)}
                              className="w-full text-[11px] bg-card-alt/60 border border-border/50 rounded px-1.5 py-0.5 text-ink focus:border-accent outline-none font-mono"
                            />
                          </td>
                        ))}
                        <td className="px-1 py-1 border-b border-border/40">
                          <button
                            onClick={() => deleteRow(i)}
                            title="Delete this observation"
                            className="text-[10px] font-bold text-risk-high bg-risk-high/10 border border-risk-high/25 rounded px-1.5 py-0.5 hover:bg-risk-high/20 cursor-pointer"
                          >
                            x
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {errors.length > 0 && (
                <div className="text-[11px] text-risk-high bg-risk-high/10 border border-risk-high/25 rounded-lg px-3 py-2 font-mono">
                  {errors.map((e, i) => <div key={i}>• {e}</div>)}
                </div>
              )}

              <p className="text-[10px] text-ink-faint leading-relaxed font-mono">
                Observation history requires 5 points spaced 6h apart. Sent directly to POST /api/analyze for causal feature matrix generation and EXP005 GRU inference.
              </p>
            </div>
          )}
        </div>
      </div>
    </CardShell>
  );
}
