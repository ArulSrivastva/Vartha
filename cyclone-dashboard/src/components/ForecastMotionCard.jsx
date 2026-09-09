import CardShell from './CardShell';

function toRad(deg) {
  return (deg * Math.PI) / 180;
}

function toDeg(rad) {
  return (rad * 180) / Math.PI;
}

function haversineKm(lat1, lon1, lat2, lon2) {
  const R = 6371; // Earth radius km
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) * Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}

function bearingDeg(lat1, lon1, lat2, lon2) {
  const dLon = toRad(lon2 - lon1);
  const y = Math.sin(dLon) * Math.cos(toRad(lat2));
  const x =
    Math.cos(toRad(lat1)) * Math.sin(toRad(lat2)) -
    Math.sin(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.cos(dLon);
  const deg = (toDeg(Math.atan2(y, x)) + 360) % 360;
  return Math.round(deg);
}

function compassDirection(deg) {
  const dirs = [
    'N', 'NNE', 'NE', 'ENE',
    'E', 'ESE', 'SE', 'SSE',
    'S', 'SSW', 'SW', 'WSW',
    'W', 'WNW', 'NW', 'NNW'
  ];
  const idx = Math.round(deg / 22.5) % 16;
  return dirs[idx];
}

export default function ForecastMotionCard({ detection, forecast, loading }) {
  if (loading) {
    return (
      <CardShell eyebrow="Track Kinematics" title="Forecast Motion &amp; Track Bearing">
        <div className="h-32 animate-pulse bg-card-alt rounded-lg" />
      </CardShell>
    );
  }

  if (!forecast || !detection || !detection.detected || !detection.location || forecast.length === 0) {
    return (
      <CardShell eyebrow="Track Kinematics" title="Forecast Motion &amp; Track Bearing">
        <div className="p-4 text-center text-[11.5px] text-ink-soft bg-card-alt/30 rounded-lg border border-border/40">
          No active cyclone trajectory — track kinematics on standby.
        </div>
      </CardShell>
    );
  }

  const curLat = detection.location.lat;
  const curLon = detection.location.lon;

  // Track points: current (0h) + forecast points (+6h, +12h, +24h)
  const pts = [
    { hour: 0, lat: curLat, lon: curLon, label: "Current Fix" },
    ...forecast.map(f => ({
      hour: f.hour !== undefined ? f.hour : f.hours,
      lat: f.lat !== undefined ? f.lat : f.latitude,
      lon: f.lon !== undefined ? f.lon : f.longitude,
      label: f.label || `+${f.hour || f.hours}h`,
      wind: f.windSpeedKmh || f.wind_speed_kmh
    }))
  ];

  const intervals = [];
  for (let i = 1; i < pts.length; i++) {
    const p1 = pts[i - 1];
    const p2 = pts[i];
    const dt = p2.hour - p1.hour;
    const dist = haversineKm(p1.lat, p1.lon, p2.lat, p2.lon);
    const brg = bearingDeg(p1.lat, p1.lon, p2.lat, p2.lon);
    const speed = dt > 0 ? dist / dt : 0;

    intervals.push({
      fromLabel: p1.label,
      toLabel: p2.label,
      hours: dt,
      targetLat: p2.lat,
      targetLon: p2.lon,
      distanceKm: Math.round(dist),
      bearing: brg,
      direction: compassDirection(brg),
      speedKmh: Math.round(speed * 10) / 10,
    });
  }

  // Net 24h displacement from 0h to last point
  const lastPt = pts[pts.length - 1];
  const netDist = haversineKm(curLat, curLon, lastPt.lat, lastPt.lon);
  const netBearing = bearingDeg(curLat, curLon, lastPt.lat, lastPt.lon);
  const netSpeed = lastPt.hour > 0 ? netDist / lastPt.hour : 0;

  return (
    <CardShell eyebrow="Track Kinematics" title="Forecast Motion &amp; Track Bearing">
      <div className="space-y-3 text-[11.5px]">
        {/* Table of intervals */}
        <div className="overflow-x-auto">
          <table className="w-full text-[11px] border-collapse">
            <thead>
              <tr className="border-b border-border text-ink-faint uppercase font-bold text-[9.5px]">
                <th className="text-left py-1.5 px-3">Horizon</th>
                <th className="text-left py-1.5 px-3">Destination</th>
                <th className="text-right py-1.5 px-3 whitespace-nowrap">Heading</th>
                <th className="text-right py-1.5 px-3 whitespace-nowrap">Displacement</th>
                <th className="text-right py-1.5 px-3 whitespace-nowrap">Translation Speed</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/40 font-mono">
              {intervals.map((inv, idx) => (
                <tr key={idx} className="hover:bg-card-alt/30 transition-colors">
                  <td className="py-2 px-3 font-bold text-ink whitespace-nowrap">
                    {inv.fromLabel} → {inv.toLabel}
                  </td>
                  <td className="py-2 px-3 text-ink-soft whitespace-nowrap">
                    {inv.targetLat.toFixed(2)}°N, {inv.targetLon.toFixed(2)}°E
                  </td>
                  <td className="py-2 px-3 text-right text-accent-strong font-bold whitespace-nowrap">
                    {inv.bearing}° ({inv.direction})
                  </td>
                  <td className="py-2 px-3 text-right text-ink whitespace-nowrap">
                    {inv.distanceKm} km
                  </td>
                  <td className="py-2 px-3 text-right font-bold text-ink whitespace-nowrap">
                    {inv.speedKmh} km/h
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* 24h Net Movement Banner */}
        <div className="bg-card-alt/60 border border-border/50 rounded-xl p-2.5 flex items-center justify-between font-mono text-[10.5px]">
          <span className="text-ink-faint uppercase font-sans font-bold text-[9.5px]">
            Overall +24h Trajectory Vector:
          </span>
          <span className="text-ink font-bold">
            {Math.round(netDist)} km @ {netBearing}° ({compassDirection(netBearing)}) · Avg Speed: {Math.round(netSpeed * 10) / 10} km/h
          </span>
        </div>
      </div>
    </CardShell>
  );
}
