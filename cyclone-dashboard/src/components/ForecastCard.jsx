import CardShell from './CardShell';

export default function ForecastCard({ forecast, loading }) {
  if (loading) {
    return (
      <CardShell eyebrow="Stage 03 — Prediction" title="Forecast Positions Timeline">
        <div className="h-40 animate-pulse bg-card-alt rounded-lg" />
      </CardShell>
    );
  }

  if (!forecast || forecast.length === 0) {
    return (
      <CardShell eyebrow="Stage 03 — Prediction" title="Forecast Timeline">
        <div className="p-4 text-center text-[11.5px] text-ink-soft bg-card-alt/30 rounded-lg border border-border/40">
          No active forecast horizons — basin clear.
        </div>
      </CardShell>
    );
  }

  return (
    <CardShell eyebrow="Stage 03 — Prediction" title="Forecast Timeline">
      <div className="relative pl-6 space-y-6 mt-2">
        {/* Vertical Timeline bar */}
        <div className="absolute left-[7px] top-2 bottom-2 w-[2px] bg-accent-soft border-l border-accent/20" />

        {forecast.map((p, idx) => {
          const hr = p.hours !== undefined ? p.hours : p.hour;
          const lat = p.latitude !== undefined ? p.latitude : p.lat;
          const lon = p.longitude !== undefined ? p.longitude : p.lon;
          const wind = p.wind_speed_kmh !== undefined ? p.wind_speed_kmh : p.windSpeedKmh;
          const label = p.label || `+${hr}h`;
          const confidence = p.confidence !== undefined ? `${p.confidence}%` : "Awaiting data";

          return (
            <div key={idx} className="relative group">
              {/* Timeline Bullet */}
              <div className="absolute -left-[24px] top-1.5 w-3.5 h-3.5 rounded-full border-2 border-white bg-accent shadow-sm group-hover:scale-110 transition-transform" />

              <div>
                <div className="flex items-center justify-between gap-2">
                  <span className="text-[12.5px] font-bold text-ink uppercase tracking-wide">
                    {label} Horizon
                  </span>
                  {p.confidence !== undefined && (
                    <span className="text-[10px] font-semibold text-accent-strong bg-accent-soft/40 px-1.5 py-0.5 rounded">
                      Conf: {confidence}
                    </span>
                  )}
                </div>

                <div className="grid grid-cols-2 gap-2 mt-1.5 text-[12px] bg-card-alt/30 border border-border/40 rounded-lg p-2.5">
                  <div>
                    <span className="text-[10px] text-ink-faint uppercase block">Location</span>
                    <span className="text-ink font-mono font-medium block mt-0.5">
                      {lat.toFixed(2)}°N {lon.toFixed(2)}°E
                    </span>
                  </div>

                  <div>
                    <span className="text-[10px] text-ink-faint uppercase block">Wind Intensity</span>
                    <span className="text-ink font-mono font-bold block mt-0.5">
                      {wind} km/h
                    </span>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </CardShell>
  );
}
