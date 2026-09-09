import { useEffect, useState } from 'react';
import { formatDateTime } from '../lib/format';
import { USE_MOCK } from '../api/client';

export default function Header({ meta, loading, onExport }){
  const [time, setTime] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const formatUTC = (date) => {
    return date.toISOString().replace('T', ' ').substring(0, 19) + ' UTC';
  };

  const formatIST = (date) => {
    return date.toLocaleTimeString('en-IN', { timeZone: 'Asia/Kolkata', hour12: false }) + ' IST';
  };

  return (
    <header className="border-b border-border bg-cream/95 backdrop-blur sticky top-0 z-40 shadow-sm print:hidden">
      <div className="max-w-[1400px] mx-auto px-6 py-3.5 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        
        {/* Left branding */}
        <div>
          <div className="flex items-baseline gap-3">
            <h1 className="font-serif text-[22px] font-bold tracking-tight text-ink">Vartha</h1>
            <span className="text-[12px] font-medium tracking-wide text-ink-soft uppercase border-l border-border pl-3">
              Tropical Cyclone Intelligence &amp; Forecasting System
            </span>
          </div>
          
          <div className="text-[11.5px] text-ink-soft mt-1.5 flex flex-wrap items-center gap-3">
            <span className="flex items-center gap-1.5 bg-risk-low/10 text-risk-low px-2 py-0.5 rounded-full text-[10.5px] font-bold uppercase tracking-wider">
              <span className="w-1.5 h-1.5 rounded-full bg-risk-low animate-pulse" />
              System Operational
            </span>
            
            {USE_MOCK ? (
              <span className="flex items-center gap-1.5 bg-risk-medium/10 text-risk-medium px-2 py-0.5 rounded-full text-[10.5px] font-bold uppercase tracking-wider">
                <span className="w-1.5 h-1.5 rounded-full bg-risk-medium" />
                SIMULATION MODE
              </span>
            ) : (
              <span className="flex items-center gap-1.5 bg-accent-strong/15 text-accent-strong px-2 py-0.5 rounded-full text-[10.5px] font-bold uppercase tracking-wider">
                <span className="w-1.5 h-1.5 rounded-full bg-accent-strong animate-pulse" />
                INTEGRATED · REAL MODELS
              </span>
            )}

            {!loading && meta && (
              <span className="text-ink-faint">
                Last Satellite Pass: <span className="font-medium">{formatDateTime(meta.lastPass)}</span>
              </span>
            )}
          </div>
        </div>

        {/* Right side system metadata & clock */}
        <div className="flex flex-col sm:flex-row items-start sm:items-center gap-3 text-right self-stretch md:self-auto justify-between md:justify-end border-t md:border-t-0 border-border/50 pt-3 md:pt-0">
          <button
            onClick={onExport}
            className="text-[11.5px] font-bold text-ink border border-accent/40 bg-accent-soft/50 rounded-lg px-3.5 py-2 hover:bg-accent-soft transition-all cursor-pointer flex items-center gap-1.5 shadow-sm"
          >
            Export Report (PDF/JSON)
          </button>

          <div className="bg-card border border-border rounded-xl p-2 px-3 flex flex-col items-start sm:items-end justify-center w-full sm:w-auto shadow-xs">
            <div className="text-ink-faint text-[8.5px] uppercase tracking-wider font-sans font-semibold">Operational Clock</div>
            <div className="text-ink text-[15px] font-bold font-mono tabular-nums leading-tight mt-0.5">
              {formatIST(time)}
            </div>
            <div className="text-ink-soft text-[11px] font-mono tabular-nums leading-tight mt-0.5">
              {formatUTC(time)}
            </div>
          </div>

          {!loading && meta && (
            <div className="flex items-center gap-3 text-[11px] font-semibold text-ink-soft bg-card-alt border border-border px-3 py-1.5 rounded-lg">
              <div>
                <span className="text-ink-faint font-normal block text-[9px] uppercase tracking-wider">System</span>
                <span className="text-ink text-[12px] font-bold font-serif">{meta.systemName}</span>
              </div>
              <div className="border-l border-border pl-2.5 self-stretch flex flex-col justify-center">
                <span className="text-ink-faint font-normal block text-[9px] uppercase tracking-wider">Basin</span>
                <span className="text-ink">{meta.basin}</span>
              </div>
            </div>
          )}
        </div>

      </div>
    </header>
  );
}
