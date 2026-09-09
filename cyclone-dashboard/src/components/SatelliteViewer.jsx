import { useRef, useState } from 'react';
import CardShell from './CardShell';
import { formatDateTime } from '../lib/format';
import { uploadSatelliteImage } from '../api/client';

export default function SatelliteViewer({ satellite, meta, loading, onImageResult }){
  const [preview, setPreview] = useState(null);     // object-URL of the local preview
  const [tensorPreview, setTensorPreview] = useState(null); // base64 of 224x224
  const [viewMode, setViewMode] = useState('raw');  // 'raw' | 'tensor'
  const [scale, setScale] = useState(1);
  const [state, setState] = useState('idle');       // idle|selected|uploading|done|failed
  const [fileMeta, setFileMeta] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);
  const [sourceLabel, setSourceLabel] = useState('LIVE SATELLITE INGEST');
  const [localBbox, setLocalBbox] = useState(null);
  const inputRef = useRef(null);

  async function handleFile(e){
    const file = e.target.files?.[0];
    if(!file) return;
    if(preview) URL.revokeObjectURL(preview);
    setPreview(URL.createObjectURL(file));
    setTensorPreview(null);
    setLocalBbox(null);
    setFileMeta({ name: file.name, size: (file.size / 1024 / 1024).toFixed(2) + " MB" });
    setState('uploading');
    setErrorMsg(null);
    setSourceLabel('USER IMAGE · ANALYZING');
    try {
      const result = await uploadSatelliteImage(file);
      setSourceLabel(result.sourceLabel || 'USER-UPLOADED IMAGE');
      if (result.preprocessedPreview) setTensorPreview(result.preprocessedPreview);
      if (result.satellite?.boundingBox) setLocalBbox(result.satellite.boundingBox);
      setState('done');
      if(onImageResult) onImageResult(result);
    } catch (err) {
      setErrorMsg(err.message);
      setState('failed');
      setSourceLabel('USER IMAGE · FAILED');
    }
  }

  const bbox = localBbox !== null ? localBbox : satellite?.boundingBox;
  const refSource = satellite?.source || '';

  return (
    <CardShell
      eyebrow="Primary Input Feeds"
      title="Satellite Observation Viewer &amp; Preprocessing Inspector"
      right={
        <div className="flex flex-wrap items-center gap-2.5">
          {/* Preprocessing Inspector Toggle */}
          <div className="flex items-center bg-card-alt border border-border rounded-lg p-0.5 text-[11px]">
            <button
              onClick={() => setViewMode('raw')}
              className={`px-3 py-1 rounded-md transition-colors cursor-pointer font-semibold ${
                viewMode === 'raw' ? 'bg-accent-strong text-white' : 'text-ink-soft hover:text-ink'
              }`}
            >
              Raw IR View
            </button>
            <button
              onClick={() => setViewMode('tensor')}
              className={`px-3 py-1 rounded-md transition-colors cursor-pointer font-semibold ${
                viewMode === 'tensor' ? 'bg-accent-strong text-white' : 'text-ink-soft hover:text-ink'
              }`}
            >
              224×224 Tensor Input
            </button>
          </div>

          <button
            onClick={() => inputRef.current?.click()}
            className="text-[11.5px] font-semibold text-white bg-accent-strong rounded-lg px-3.5 py-1.5 hover:opacity-90 transition-opacity cursor-pointer shadow-sm"
          >
            Upload Custom Image
          </button>
          <input ref={inputRef} type="file" accept="image/*" onChange={handleFile} className="hidden" />
        </div>
      }
    >
      <div className="space-y-3">
        {/* Live MOSDAC Telemetry Status Header */}
        <div className="bg-card-alt/40 border border-border/50 rounded-xl p-2.5 flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="w-2 h-2 rounded-full bg-risk-low animate-pulse shrink-0" />
            <span className="text-[10px] font-bold text-ink-faint uppercase tracking-wider">
              Active Input:
            </span>
            <span className="text-[11px] font-mono font-semibold text-ink bg-card border border-border px-2 py-0.5 rounded">
              {preview ? `Uploaded: ${fileMeta?.name || "Frame"}` : `MOSDAC · ${meta?.datasetId || "3RIMG_L2B_SST"}`}
            </span>
            <span className="text-[11px] text-ink-soft">
              {bbox ? `Vortex Detected (${bbox.confidence}% confidence)` : (satellite?.detected ? "Vortex Observed" : "Basin Clear · No Cyclone Detected")}
            </span>
          </div>

          <span className="text-[10.5px] font-mono font-medium text-ink-faint">
            {viewMode === 'tensor' ? 'Mode: 224×224 Preprocessed Tensor' : 'Mode: Full Resolution View'}
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">

          {/* Image Container Panel */}
          <div className="md:col-span-3 relative rounded-xl overflow-hidden border border-border-soft bg-[#0d1420] h-[340px] select-none flex items-center justify-center p-2">
            <div
              className="relative h-full aspect-square max-w-full flex items-center justify-center"
              style={{ transform: `scale(${scale})`, transformOrigin: 'center center', transition: 'transform 0.2s ease' }}
            >
              {viewMode === 'tensor' && tensorPreview ? (
                <div className="flex flex-col items-center justify-center h-full w-full bg-black/90 p-2">
                  <img src={tensorPreview} alt="224x224 Normalized Tensor Input" className="w-[224px] h-[224px] object-contain border border-white/20 shadow-lg rounded" />
                  <span className="text-[9.5px] font-mono text-ink-faint mt-2 uppercase tracking-wider">
                    MobileNetV3 224×224 Normalization Transform
                  </span>
                </div>
              ) : preview ? (
                <img src={preview} alt="Satellite image" className="w-full h-full object-contain rounded-lg shadow-md" />
              ) : (
                <div className="w-full h-full">
                  <SyntheticSatelliteImage detected={satellite?.detected} />
                </div>
              )}

              {bbox && !loading && (
                <div
                  className="absolute border-2 rounded-sm shadow-[0_0_12px_rgba(242,169,59,0.8)] pointer-events-none transition-all duration-200"
                  style={{
                    borderColor: '#F2A93B',
                    left: `${bbox.x * 100}%`, top: `${bbox.y * 100}%`,
                    width: `${bbox.w * 100}%`, height: `${bbox.h * 100}%`,
                  }}
                >
                  <span className="absolute -top-6 left-0 text-[10px] font-bold text-[#F2A93B] bg-[#0d1420]/95 px-2 py-0.5 rounded border border-[#F2A93B]/50 whitespace-nowrap uppercase tracking-wider font-mono shadow-sm">
                    Cyclone Center: {bbox.confidence}%
                  </span>
                </div>
              )}

              {!loading && (
                <div className="absolute top-2 left-2 flex items-center gap-1.5 pointer-events-none">
                  {state === 'uploading' && (
                    <span className="text-[10px] font-bold text-amber-300 bg-[#0d1420]/90 px-2 py-0.5 rounded border border-amber-300/40 uppercase tracking-wider font-mono animate-pulse">
                      Analyzing through CenterNet &amp; ResNet…
                    </span>
                  )}
                  {state === 'failed' && (
                    <span className="text-[10px] font-bold text-red-400 bg-[#0d1420]/90 px-2 py-0.5 rounded border border-red-400/40 uppercase tracking-wider font-mono">
                      Analysis failed — see metadata
                    </span>
                  )}
                </div>
              )}
            </div>

            {/* Zoom controls */}
            <div className="absolute bottom-3 right-3 flex items-center gap-1.5 bg-[#0d1420]/80 backdrop-blur rounded-lg px-2 py-1.5 border border-white/10 z-10">
              <button onClick={() => setScale(s => Math.max(1, +(s - 0.25).toFixed(2)))} className="w-7 h-7 rounded-md text-white/90 hover:bg-white/10 text-sm leading-none font-bold">−</button>
              <span className="text-[11px] text-white/80 w-9 text-center font-mono tabular-nums">{Math.round(scale * 100)}%</span>
              <button onClick={() => setScale(s => Math.min(3, +(s + 0.25).toFixed(2)))} className="w-7 h-7 rounded-md text-white/90 hover:bg-white/10 text-sm leading-none font-bold">+</button>
              <button onClick={() => setScale(1)} className="ml-1 text-[10.5px] text-white/60 hover:text-white px-1.5 font-semibold">Reset</button>
            </div>
          </div>

        {/* Info panel */}
        <div className="md:col-span-1 flex flex-col justify-between p-3.5 bg-card-alt/60 border border-border/50 rounded-xl space-y-4 text-[11.5px] overflow-hidden min-w-0">
          <div className="space-y-3 min-w-0">
            <span className="text-[9.5px] uppercase tracking-wider text-ink-faint font-bold block border-b border-border/50 pb-1">Image Metadata</span>

            <div className="min-w-0">
              <span className="text-ink-faint text-[9.5px] uppercase block tracking-wider">Source Platform</span>
              <span className="text-ink font-semibold break-words">
                {preview ? "User-uploaded satellite image" : "ISRO MOSDAC INSAT-3DR (config.json)"}
              </span>
            </div>

            <div className="min-w-0">
              <span className="text-ink-faint text-[9.5px] uppercase block tracking-wider">Dataset / Frame</span>
              <span className="text-ink font-mono text-[10px] break-all leading-tight block">
                {preview ? (fileMeta?.name || "user image") : (meta?.latestFile || meta?.datasetId || "3RIMG_L2B_SST")}
              </span>
            </div>

            <div className="min-w-0">
              <span className="text-ink-faint text-[9.5px] uppercase block tracking-wider">Channel / Feed</span>
              <span className="text-ink font-mono text-[10px] font-medium leading-tight break-words block">
                {sourceLabel}
              </span>
            </div>

            <div className="min-w-0">
              <span className="text-ink-faint text-[9.5px] uppercase block tracking-wider">Last Telemetry Pass</span>
              <span className="text-ink font-mono font-medium leading-tight block break-words">
                {preview
                  ? (fileMeta?.name ? new Date().toISOString().replace('T', ' ').substring(0, 19) + ' UTC' : "Recent")
                  : (meta?.lastPass ? formatDateTime(meta.lastPass) : "—")}
              </span>
            </div>

            <div className="min-w-0">
              <span className="text-ink-faint text-[9.5px] uppercase block tracking-wider">Center Vortex Detection</span>
              <span className="text-ink font-semibold break-words">{bbox ? `ML localizer ${bbox.confidence}%` : (satellite?.detected ? "Vortex detected" : "Basin clear (zero vortices detected)")}</span>
            </div>

            {state === 'failed' && errorMsg && (
              <div className="text-[10px] text-red-300 font-mono leading-relaxed border border-red-400/30 bg-red-400/5 rounded-lg p-2 break-all">
                {errorMsg}
              </div>
            )}
          </div>

          <div className="text-[10px] text-ink-faint font-mono leading-relaxed pt-2 border-t border-border/50 break-words min-w-0 overflow-hidden">
            {preview ? (
              <div className="space-y-1">
                <div>Processed by convolutional vision network.</div>
                <div className="text-ink font-sans text-[10.5px] break-all">
                  <span className="font-semibold text-ink-faint font-mono text-[9px] uppercase">File: </span>
                  {fileMeta?.name || "uploaded image"} <span className="text-ink-faint font-mono text-[9px]">({fileMeta?.size || "—"})</span>
                </div>
              </div>
            ) : (
              "Live INSAT-3DR IR telemetry via MOSDAC API."
            )}
            <div className="mt-2 text-[9px] break-words">
              STATUS: {state === 'done' ? 'USER IMAGE · REAL PREDICTION' : (preview ? 'USER IMAGE · LOCAL PREVIEW' : 'LIVE MOSDAC FEED')}
            </div>
          </div>
        </div>

      </div>
      </div>
    </CardShell>
  );
}

// Procedural placeholder rendering
function SyntheticSatelliteImage({ detected }){
  if (!detected) {
    return (
      <div className="flex flex-col items-center justify-center h-full w-full p-6 text-center select-none bg-[#090e15]">
        <div className="w-12 h-12 rounded-full border border-emerald-500/40 bg-emerald-500/10 flex items-center justify-center mb-3">
          <span className="w-3.5 h-3.5 rounded-full bg-emerald-500 animate-pulse" />
        </div>
        <div className="font-serif text-[16px] font-bold text-white tracking-wide">
          Basin Clear · No Cyclone Detected
        </div>
        <p className="text-[11.5px] text-slate-300 font-sans mt-1.5 max-w-sm leading-relaxed">
          ISRO MOSDAC INSAT-3DR telemetry indicates a quiescent atmospheric baseline across Bay of Bengal and Arabian Sea. Zero cyclonic vortices detected.
        </p>
        <span className="mt-3 text-[10px] text-slate-400 font-mono border border-white/10 rounded-full px-3 py-1 bg-white/5">
          Upload Custom Image above to evaluate with CenterNet &amp; ResNet models
        </span>
      </div>
    );
  }
  return (
    <svg viewBox="0 0 400 320" className="w-full h-full">
      <defs>
        <radialGradient id="satBg" cx="50%" cy="45%" r="75%">
          <stop offset="0%" stopColor="#1a2534" />
          <stop offset="100%" stopColor="#0a1119" />
        </radialGradient>
        <filter id="satBlur"><feGaussianBlur stdDeviation="6" /></filter>
      </defs>
      <rect width="400" height="320" fill="url(#satBg)" />
      {[70, 55, 42, 30, 20].map((r, i) => (
        <circle key={r} cx={190 + i * 4} cy={150 - i * 2} r={r} fill="none"
          stroke="#cfd8e3" strokeOpacity={0.12 + i * 0.03} strokeWidth={10} filter="url(#satBlur)" />
      ))}
      <circle cx="196" cy="146" r="14" fill="#0a1119" stroke="#cfd8e3" strokeOpacity="0.3" strokeWidth="2" />
    </svg>
  );
}
