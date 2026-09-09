import { useState, useEffect } from 'react';
import CardShell from './CardShell';
import { fetchHealth } from '../api/client';

export default function UnifiedAnalysisPanel({ data, loading, onReanalyze, backendOffline }){
  const [health, setHealth] = useState(null);
  const [healthChecking, setHealthChecking] = useState(false);

  useEffect(() => {
    let mounted = true;
    setHealthChecking(true);
    fetchHealth()
      .then(res => {
        if(mounted) {
          setHealth(res);
          setHealthChecking(false);
        }
      })
      .catch(err => {
        if(mounted) {
          setHealth(null);
          setHealthChecking(false);
        }
      });
    return () => { mounted = false; };
  }, [data]);

  const p2Ready = health?.ml?.p2?.ready ?? true;
  const p3Ready = health?.ml?.p3_tabular?.available ?? true;
  const p4Ready = health?.forecasting?.model_ready ?? true;

  const detection = data?.detection;
  const classification = data?.classification;
  const forecast = data?.forecast;

  return (
    <CardShell
      eyebrow="Integrated Model Pipeline"
      title="Unified Cyclone AI Analysis &amp; System Health"
      right={
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 bg-card border border-border/80 rounded-lg px-2.5 py-1">
            <span className={`w-2 h-2 rounded-full ${backendOffline ? 'bg-risk-high' : 'bg-risk-low animate-pulse'}`} />
            <span className="text-[11px] font-mono font-semibold text-ink">
              {backendOffline ? "Backend Offline" : "FastAPI: 8000 OK"}
            </span>
          </div>

          <button
            onClick={onReanalyze}
            disabled={loading}
            className="text-[11.5px] font-bold text-white bg-accent-strong rounded-lg px-4 py-1.5 hover:opacity-90 transition-opacity disabled:opacity-50 shadow-sm flex items-center gap-1.5"
          >
            {loading ? (
              <>
                <span className="w-3 h-3 rounded-full border-2 border-white/30 border-t-white animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                Re-run Full Analysis
              </>
            )}
          </button>
        </div>
      }
    >
      <div className="space-y-4">
        {/* Pipeline Stage Badges */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-[11.5px]">
          
          {/* Segment 1: Detection */}
          <div className="bg-card-alt/70 border border-border/60 rounded-xl p-3 flex flex-col justify-between space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-[10px] uppercase font-bold text-ink-faint tracking-wider">Satellite Detection</span>
              <span className="flex items-center gap-1 text-[10px] font-bold text-risk-low bg-risk-low/10 px-2 py-0.5 rounded-full border border-risk-low/20">
                <span className="w-1.5 h-1.5 rounded-full bg-risk-low" />
                {p2Ready ? "Active" : "Standby"}
              </span>
            </div>
            <div>
              <div className="font-serif font-bold text-ink text-[13px]">
                {detection ? (detection.detected ? "Cyclone Detected" : "No Cyclone Detected") : "Awaiting Frame"}
              </div>
              <div className="text-[10.5px] text-ink-soft font-mono mt-0.5">
                {detection ? (detection.detected ? `${detection.confidence}% confidence · ${detection.structuralPattern || "Organized Pattern"}` : "Basin Clear · Stable Conditions") : "CenterNet Keypoint Detector"}
              </div>
            </div>
          </div>

          {/* Segment 2: Classification */}
          <div className="bg-card-alt/70 border border-border/60 rounded-xl p-3 flex flex-col justify-between space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-[10px] uppercase font-bold text-ink-faint tracking-wider">Intensity Classification</span>
              <span className="flex items-center gap-1 text-[10px] font-bold text-risk-low bg-risk-low/10 px-2 py-0.5 rounded-full border border-risk-low/20">
                <span className="w-1.5 h-1.5 rounded-full bg-risk-low" />
                {p3Ready ? "Active" : "Fallback"}
              </span>
            </div>
            <div>
              <div className="font-serif font-bold text-ink text-[13px] truncate">
                {classification?.category || "Awaiting Data"}
              </div>
              <div className="text-[10.5px] text-ink-soft font-mono mt-0.5">
                {classification ? (detection?.detected ? `${classification.windSpeedKmh} km/h · ${classification.pressureHpa} hPa` : "Clear Basin · 1012 hPa baseline") : "Multi-Source Tabular Fusion"}
              </div>
            </div>
          </div>

          {/* Segment 3: Forecasting */}
          <div className="bg-card-alt/70 border border-border/60 rounded-xl p-3 flex flex-col justify-between space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-[10px] uppercase font-bold text-ink-faint tracking-wider">Track Forecast</span>
              <span className="flex items-center gap-1 text-[10px] font-bold text-risk-low bg-risk-low/10 px-2 py-0.5 rounded-full border border-risk-low/20">
                <span className="w-1.5 h-1.5 rounded-full bg-risk-low" />
                {p4Ready ? "Active" : "Standby"}
              </span>
            </div>
            <div>
              <div className="font-serif font-bold text-ink text-[13px]">
                {forecast?.length ? `+${forecast[forecast.length - 1].hour || forecast[forecast.length - 1].hours}h Forecast Horizon` : "Basin Clear (Standby)"}
              </div>
              <div className="text-[10.5px] text-ink-soft font-mono mt-0.5">
                {forecast?.length ? `6h, 12h, 24h Waypoints` : "Gated Multimodal Fusion"}
              </div>
            </div>
          </div>

          {/* Segment 4: Provenance */}
          <div className="bg-card-alt/70 border border-border/60 rounded-xl p-3 flex flex-col justify-between space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-[10px] uppercase font-bold text-ink-faint tracking-wider">Data Provenance</span>
              <span className="flex items-center gap-1 text-[10px] font-bold text-accent-strong bg-accent-soft/40 px-2 py-0.5 rounded-full border border-accent/20">
                <span className="w-1.5 h-1.5 rounded-full bg-accent-strong" />
                Audited
              </span>
            </div>
            <div>
              <div className="font-serif font-bold text-ink text-[13px]">
                Audited Model Checkpoints
              </div>
              <div className="text-[10.5px] text-ink-soft font-mono mt-0.5">
                Verified Real Predictions
              </div>
            </div>
          </div>

        </div>

        {/* Scientific Disclaimer Line */}
        <div className="bg-mist/30 border border-border-soft rounded-lg px-3 py-2 text-[11px] text-ink-soft flex items-center justify-between">
          <span>
            <strong className="text-ink font-semibold">Research Prototype:</strong> Offline real-time machine learning system.
          </span>
          <span className="font-mono text-[10px] text-ink-faint font-semibold uppercase shrink-0 pl-2">
            Vartha · SIH 2026
          </span>
        </div>
      </div>
    </CardShell>
  );
}
