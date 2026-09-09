import { useMemo } from 'react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, Legend } from 'recharts';

export default function ChartsPanel({ data, loading }) {
  const windTrend = useMemo(() => {
    if (!data) return [];
    const history = (data.windHistory || []).map(h => ({ t: h.t, value: h.value, label: "Observed" }));
    const forecast = (data.forecast || []).map(f => {
      const hr = f.hours !== undefined ? f.hours : f.hour;
      const wind = f.wind_speed_kmh !== undefined ? f.wind_speed_kmh : f.windSpeedKmh;
      return { t: f.label || `+${hr}h`, value: wind, label: "Forecasted" };
    });
    return [...history, ...forecast];
  }, [data]);

  const pressureTrend = useMemo(() => {
    if (!data) return [];
    const history = (data.pressureHistory || []).map(h => ({ t: h.t, value: h.value }));
    const forecast = (data.forecast || []).map(f => {
      const hr = f.hours !== undefined ? f.hours : f.hour;
      return { t: f.label || `+${hr}h`, value: f.pressureHpa || null };
    }).filter(p => p.value !== null);
    return [...history, ...forecast];
  }, [data]);

  const envTrend = useMemo(() => {
    if (!data || !data.sstHistory || !data.envWindHistory) return [];
    // Merge SST and Env Wind by time label
    return data.sstHistory.map((s, idx) => {
      const w = data.envWindHistory[idx] || { value: null };
      return {
        t: s.t,
        sst: s.value,
        envWind: w.value
      };
    });
  }, [data]);

  if (loading || !data) {
    return (
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {[0, 1, 2].map(i => (
          <div key={i} className="card h-[220px] animate-pulse bg-card-alt" />
        ))}
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
      
      {/* 1. Stitched Wind Speed Profile */}
      <div className="card p-5">
        <div className="text-[11px] uppercase tracking-wider text-ink-soft font-semibold mb-0.5">Intensity Profile</div>
        <h3 className="font-serif text-[14px] font-bold text-ink mb-3">Wind Speed Trend (History &amp; Forecast)</h3>
        <div className="h-[140px] text-[10px]">
          {windTrend.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={windTrend} margin={{ top: 5, right: 5, left: -22, bottom: 0 }}>
                <CartesianGrid stroke="#ECE0D4" vertical={false} />
                <XAxis dataKey="t" tick={{ fontSize: 9.5, fill: '#9C9187' }} axisLine={{ stroke: '#E4D6C9' }} tickLine={false} />
                <YAxis tick={{ fontSize: 9.5, fill: '#9C9187' }} axisLine={false} tickLine={false} unit=" km" width={40} />
                <Tooltip
                  contentStyle={{ fontSize: 11.5, fontFamily: 'IBM Plex Sans, sans-serif', borderRadius: 8, border: '1px solid #E4D6C9', background: '#FCF9F5' }}
                  labelStyle={{ color: '#6B615A', fontWeight: 'bold' }}
                />
                <Line type="monotone" dataKey="value" stroke="#B85C4C" strokeWidth={2.5} dot={{ r: 3, strokeWidth: 0, fill: "#B85C4C" }} name="Wind Speed (km/h)" />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-full flex items-center justify-center text-ink-faint">Data unavailable</div>
          )}
        </div>
      </div>

      {/* 2. Stitched Barometric Pressure Profile */}
      <div className="card p-5">
        <div className="text-[11px] uppercase tracking-wider text-ink-soft font-semibold mb-0.5">Barometric Profile</div>
        <h3 className="font-serif text-[14px] font-bold text-ink mb-3">Central Pressure Trend (hPa)</h3>
        <div className="h-[140px] text-[10px]">
          {pressureTrend.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={pressureTrend} margin={{ top: 5, right: 5, left: -18, bottom: 0 }}>
                <CartesianGrid stroke="#ECE0D4" vertical={false} />
                <XAxis dataKey="t" tick={{ fontSize: 9.5, fill: '#9C9187' }} axisLine={{ stroke: '#E4D6C9' }} tickLine={false} />
                <YAxis domain={['dataMin - 5', 'dataMax + 5']} tick={{ fontSize: 9.5, fill: '#9C9187' }} axisLine={false} tickLine={false} width={38} />
                <Tooltip
                  contentStyle={{ fontSize: 11.5, fontFamily: 'IBM Plex Sans, sans-serif', borderRadius: 8, border: '1px solid #E4D6C9', background: '#FCF9F5' }}
                  labelStyle={{ color: '#6B615A', fontWeight: 'bold' }}
                />
                <Line type="monotone" dataKey="value" stroke="#8B5E52" strokeWidth={2.5} dot={{ r: 3, strokeWidth: 0, fill: "#8B5E52" }} name="Pressure (hPa)" />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-full flex items-center justify-center text-ink-faint">Data unavailable</div>
          )}
        </div>
      </div>

      {/* 3. Dual Environmental Parameters (ERA5 SST & Environmental Wind Speed) */}
      <div className="card p-5">
        <div className="text-[11px] uppercase tracking-wider text-ink-soft font-semibold mb-0.5">ERA5 Atmospheric Variables</div>
        <h3 className="font-serif text-[14px] font-bold text-ink mb-3">SST &amp; Environmental Wind Speed</h3>
        <div className="h-[140px] text-[10px]">
          {envTrend.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={envTrend} margin={{ top: 5, right: 5, left: -20, bottom: 0 }}>
                <CartesianGrid stroke="#ECE0D4" vertical={false} />
                <XAxis dataKey="t" tick={{ fontSize: 9.5, fill: '#9C9187' }} axisLine={{ stroke: '#E4D6C9' }} tickLine={false} />
                <YAxis yAxisId="left" orientation="left" tick={{ fontSize: 8.5, fill: '#C7A05C' }} axisLine={false} tickLine={false} width={25} />
                <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 8.5, fill: '#7A9A7E' }} axisLine={false} tickLine={false} width={25} />
                <Tooltip
                  contentStyle={{ fontSize: 11.5, fontFamily: 'IBM Plex Sans, sans-serif', borderRadius: 8, border: '1px solid #E4D6C9', background: '#FCF9F5' }}
                  labelStyle={{ color: '#6B615A', fontWeight: 'bold' }}
                />
                <Legend iconSize={8} wrapperStyle={{ fontSize: 9.5, color: '#6B615A' }} />
                <Line yAxisId="left" type="monotone" dataKey="sst" stroke="#C7A05C" strokeWidth={2} dot={{ r: 2 }} name="SST (°C)" />
                <Line yAxisId="right" type="monotone" dataKey="envWind" stroke="#7A9A7E" strokeWidth={2} dot={{ r: 2 }} name="Env Wind (m/s)" />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-full flex items-center justify-center text-ink-faint">
              Environmental data unavailable (simulation only)
            </div>
          )}
        </div>
      </div>

    </div>
  );
}
