import { useEffect, useMemo } from 'react';
import { MapContainer, TileLayer, Polyline, Polygon, CircleMarker, Marker, Popup, useMap } from 'react-leaflet';
import L from 'leaflet';
import { colorForWind } from '../../api/mockData';
import { buildUncertaintyCone } from './uncertaintyCone';

const landfallIcon = L.divIcon({
  className: '',
  html: `<div style="width:14px;height:14px;background:#8B5E52;border:2px solid white;border-radius:3px;box-shadow:0 2px 6px rgba(0,0,0,0.35);transform:rotate(45deg);"></div>`,
  iconSize: [14, 14],
  iconAnchor: [7, 7],
});

function FitBounds({ bounds }){
  const map = useMap();
  useEffect(() => {
    if(bounds && bounds.length > 1) map.fitBounds(bounds, { padding: [36, 36] });
  }, [bounds, map]);
  return null;
}

function TrackpadGestureController(){
  const map = useMap();

  useEffect(() => {
    const container = map.getContainer();

    const handleWheel = (e) => {
      // Prevent the main dashboard page from scrolling when interacting with the map
      e.preventDefault();

      if (e.ctrlKey) {
        // Two-finger pinch or expand gesture on trackpad (or Ctrl + mouse wheel) -> ZOOM
        const mousePos = map.mouseEventToContainerPoint(e);
        // deltaY is negative when expanding (zooming in), positive when pinching (zooming out)
        const rawDelta = -e.deltaY;
        const zoomDelta = Math.max(-0.5, Math.min(0.5, rawDelta * 0.012));
        const currentZoom = map.getZoom();
        const minZoom = map.getMinZoom() || 2;
        const maxZoom = map.getMaxZoom() || 18;
        const targetZoom = Math.max(minZoom, Math.min(maxZoom, currentZoom + zoomDelta));
        map.setZoomAround(mousePos, targetZoom, { animate: false });
      } else {
        // Two-finger swipe/drag gesture on trackpad -> MOVE (PAN) THE MAP
        map.panBy([e.deltaX, e.deltaY], { animate: false });
      }
    };

    container.addEventListener('wheel', handleWheel, { passive: false });
    return () => {
      container.removeEventListener('wheel', handleWheel);
    };
  }, [map]);

  return null;
}

function Legend(){
  return (
    <div className="flex flex-wrap items-center gap-4 text-[11px] text-ink-soft">
      <span className="flex items-center gap-1.5"><span className="inline-block w-2.5 h-2.5 rounded-full bg-risk-high" />Current</span>
      <span className="flex items-center gap-1.5"><span className="inline-block w-3 h-0.5 bg-ink-soft/60" />Historical</span>
      <span className="flex items-center gap-1.5"><span className="inline-block w-3 h-0.5 bg-accent-strong" />Forecast</span>
      <span className="flex items-center gap-1.5"><span className="inline-block w-2.5 h-2.5 rounded-sm bg-accent-strong rotate-45" />Landfall</span>
      <span className="flex items-center gap-1.5"><span className="inline-block w-3 h-2.5 bg-accent/20 border border-accent/40" />Illustrative cone</span>
    </div>
  );
}

export default function MapView({ data, loading }){
  // The current-position marker is sourced explicitly from `detection`
  // (per the field mapping: detection → detection card + current cyclone
  // marker), never from the forecast array — forecast is strictly the
  // projected future positions (6h/12h/24h).
  const detection = data?.detection;
  const classification = data?.classification;
  const forecast = data?.forecast || [];
  const historicalTrack = data?.historicalTrack || [];
  const landfall = data?.landfall;

  const hasLocation = Boolean(detection?.detected && detection?.location?.lat != null && detection?.location?.lon != null);

  // Combine the true current position with the forecast points so the
  // track line and uncertainty cone are continuous from "now" onward.
  const trackPoints = useMemo(() => {
    if(!hasLocation) return [];
    return [
      { hour: 0, lat: detection.location.lat, lon: detection.location.lon, windSpeedKmh: classification?.windSpeedKmh || 0 },
      ...forecast.map(p => ({
        hour: p.hours !== undefined ? p.hours : p.hour,
        lat: p.latitude !== undefined ? p.latitude : p.lat,
        lon: p.longitude !== undefined ? p.longitude : p.lon,
        windSpeedKmh: p.wind_speed_kmh !== undefined ? p.wind_speed_kmh : p.windSpeedKmh,
        pressureHpa: p.pressureHpa,
        confidence: p.confidence,
        label: p.label
      })),
    ];
  }, [hasLocation, detection, forecast, classification]);

  const cone = useMemo(() => buildUncertaintyCone(trackPoints), [trackPoints]);

  if(loading || !data){
    return <div className="card h-[440px] animate-pulse bg-card-alt" />;
  }

  const historicalLatLngs = historicalTrack.map(p => [p.lat, p.lon]);
  const forecastLatLngs = trackPoints.map(p => [p.lat, p.lon]);
  const currentBand = colorForWind(classification?.windSpeedKmh ?? 0);

  const allBoundsPoints = [...historicalLatLngs, ...forecastLatLngs];
  if(landfall?.estimated && landfall.latitude != null) allBoundsPoints.push([landfall.latitude, landfall.longitude]);

  const mapCenter = hasLocation ? [detection.location.lat, detection.location.lon] : [16.0, 82.0];
  const mapZoom = hasLocation ? 6 : 5;

  return (
    <div className="card overflow-hidden">
      <div className="flex flex-wrap items-center justify-between gap-3 px-5 pt-5 pb-3">
        <div>
          <div className="text-[11px] uppercase tracking-wider text-ink-soft font-semibold">Monitoring</div>
          <h2 className="font-serif text-lg font-semibold text-ink mt-0.5">Track Map — Historical, Current &amp; Forecast Path</h2>
        </div>
        <Legend />
      </div>
      <div className="h-[420px] w-full relative z-0" style={{ isolation: 'isolate' }}>
        {!hasLocation && (
          <div className="absolute top-3 left-14 z-[400] bg-card/95 backdrop-blur border border-border shadow-sm rounded-lg px-3 py-1.5 flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-risk-low" />
            <span className="text-[11.5px] font-semibold text-ink">Basin Clear · No Active Cyclone Track</span>
          </div>
        )}
        <MapContainer
          center={mapCenter}
          zoom={mapZoom}
          scrollWheelZoom={false}
          zoomSnap={0.1}
          touchZoom={true}
          style={{ height: '100%', width: '100%', zIndex: 0 }}
        >
          <TrackpadGestureController />
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          {allBoundsPoints.length > 1 && <FitBounds bounds={allBoundsPoints} />}
          {cone.length > 0 && <Polygon positions={cone} pathOptions={{ color: '#B98671', weight: 1, fillColor: '#B98671', fillOpacity: 0.12 }} />}

          {historicalLatLngs.length > 1 && <Polyline positions={historicalLatLngs} pathOptions={{ color: '#6B615A', weight: 2, dashArray: '4 5', opacity: 0.7 }} />}
          {forecastLatLngs.length > 1 && <Polyline positions={forecastLatLngs} pathOptions={{ color: '#8B5E52', weight: 3, opacity: 0.9 }} />}

          {/* Current position — driven by detection.location + detection.confidence */}
          {hasLocation && (
            <CircleMarker
              center={[detection.location.lat, detection.location.lon]}
              radius={10}
              pathOptions={{ color: '#fff', weight: 2, fillColor: currentBand.color, fillOpacity: 0.95 }}
            >
              <Popup>
                <div style={{ fontFamily: 'IBM Plex Sans, sans-serif', fontSize: 12.5 }} className="space-y-1">
                  <strong className="text-ink block font-bold border-b border-border pb-1">CURRENT SYSTEM POSITION</strong>
                  <div>Latitude: <span className="font-mono">{detection.location.lat.toFixed(2)}°N</span></div>
                  <div>Longitude: <span className="font-mono">{detection.location.lon.toFixed(2)}°E</span></div>
                  <div>Wind: <span className="font-mono font-semibold">{classification?.windSpeedKmh || "—"} km/h</span></div>
                  {classification && <div>Category: <span className="font-semibold" style={{ color: currentBand.color }}>{classification.category}</span></div>}
                  <div>Detection Confidence: <span className="font-mono font-semibold text-accent-strong">{detection.confidence}%</span></div>
                </div>
              </Popup>
            </CircleMarker>
          )}

          {/* Forecast positions — driven by the forecast array (6h/12h/24h) */}
          {trackPoints.slice(1).map((p, idx) => {
            const band = colorForWind(p.windSpeedKmh);
            const label = p.label || `+${p.hour}h`;
            return (
              <CircleMarker
                key={idx}
                center={[p.lat, p.lon]}
                radius={6}
                pathOptions={{ color: '#fff', weight: 2, fillColor: band.color, fillOpacity: 0.95 }}
              >
                <Popup>
                  <div style={{ fontFamily: 'IBM Plex Sans, sans-serif', fontSize: 12.5 }} className="space-y-1">
                    <strong className="text-ink block font-bold border-b border-border pb-1">{label.toUpperCase()} FORECAST</strong>
                    <div>Latitude: <span className="font-mono">{p.lat.toFixed(2)}°N</span></div>
                    <div>Longitude: <span className="font-mono">{p.lon.toFixed(2)}°E</span></div>
                    <div>Wind: <span className="font-mono font-semibold">{p.windSpeedKmh} km/h</span></div>
                    {p.pressureHpa && <div>Pressure: <span className="font-mono">{p.pressureHpa} hPa</span></div>}
                    {p.confidence !== undefined && <div>Model Confidence: <span className="font-mono font-semibold">{p.confidence}%</span></div>}
                  </div>
                </Popup>
              </CircleMarker>
            );
          })}

          {/* Landfall — driven by the landfall object */}
          {landfall?.estimated && (
            <Marker position={[landfall.latitude, landfall.longitude]} icon={landfallIcon}>
              <Popup>
                <div style={{ fontFamily: 'IBM Plex Sans, sans-serif', fontSize: 12.5 }} className="space-y-1">
                  <strong className="text-accent-strong block font-bold border-b border-border pb-1">ESTIMATED LANDFALL</strong>
                  <div>Latitude: <span className="font-mono">{landfall.latitude.toFixed(2)}°N</span></div>
                  <div>Longitude: <span className="font-mono">{landfall.longitude.toFixed(2)}°E</span></div>
                  <div>Predicted Wind: <span className="font-mono font-semibold">{landfall.predictedWindKmh} km/h</span></div>
                  {landfall.distanceToLandKm && <div>Distance to Coast: <span className="font-mono font-semibold">{landfall.distanceToLandKm} km</span></div>}
                </div>
              </Popup>
            </Marker>
          )}
        </MapContainer>
      </div>
      <div className="px-5 pb-4 text-[10px] text-ink-faint">
        Core models run fully offline; the open-street-map basemap above is an
        optional online display layer only.
      </div>
    </div>
  );
}
