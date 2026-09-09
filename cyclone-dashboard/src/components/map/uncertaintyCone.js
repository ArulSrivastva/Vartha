// Builds a widening "cone of uncertainty" polygon around the forecast
// track — the further out the forecast hour, the wider the cone — using a
// simple perpendicular offset from the local track direction at each point.
// This mirrors the cone visualization used by real cyclone forecast
// products; the widening rate here is a mocked placeholder until the
// backend supplies real uncertainty radii per forecast hour.
export function buildUncertaintyCone(points){
  if(!points || points.length < 2) return [];
  const top = [];
  const bottom = [];

  for(let i = 0; i < points.length; i++){
    const p = points[i];
    const prev = points[i - 1] || p;
    const next = points[i + 1] || p;
    const dLat = next.lat - prev.lat;
    const dLon = next.lon - prev.lon;
    const len = Math.sqrt(dLat * dLat + dLon * dLon) || 1;
    const perpLat = -dLon / len;
    const perpLon = dLat / len;
    const radius = 0.12 + (p.hour || 0) * 0.028; // degrees, widens with lead time
    top.push([p.lat + perpLat * radius, p.lon + perpLon * radius]);
    bottom.push([p.lat - perpLat * radius, p.lon - perpLon * radius]);
  }

  return [...top, ...bottom.reverse()];
}
