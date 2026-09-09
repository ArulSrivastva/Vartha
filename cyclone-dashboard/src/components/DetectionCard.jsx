import CardShell, { Stat, CardSkeleton } from './CardShell';

export default function DetectionCard({ detection, loading }){
  if(loading || !detection){
    return <CardShell eyebrow="Satellite Detection" title="Cyclone Detection"><CardSkeleton/></CardShell>;
  }
  const { detected, confidence, location } = detection;
  return (
    <CardShell eyebrow="Satellite Detection" title="Cyclone Detection">
      <div className="flex items-center gap-2 mb-4">
        <span className={`inline-block w-2 h-2 rounded-full ${detected ? 'bg-risk-high' : 'bg-risk-low'} pulse-dot`} />
        <span className="text-sm font-semibold text-ink">{detected ? 'Cyclone Detected' : 'No System Detected'}</span>
      </div>
      <div className="grid grid-cols-2 gap-4">
        <Stat label="Confidence" value={`${confidence}%`} />
        <Stat label="Location" value={`${location.lat.toFixed(2)}°N ${location.lon.toFixed(2)}°E`} small />
      </div>
    </CardShell>
  );
}
