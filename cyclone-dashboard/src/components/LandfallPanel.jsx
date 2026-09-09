import CardShell, { Stat, CardSkeleton } from './CardShell';
import { formatDateTime } from '../lib/format';

export default function LandfallPanel({ landfall, loading }){
  if(loading || !landfall){
    return <CardShell eyebrow="Landfall Prediction" title="Estimated Landfall"><CardSkeleton/></CardShell>;
  }

  if(!landfall.estimated){
    return (
      <CardShell eyebrow="Landfall Prediction" title="Estimated Landfall">
        <p className="text-[13px] text-ink-soft">No landfall is currently estimated for this system.</p>
        <div className="mt-3 text-[10px] text-ink-faint border-t border-border/40 pt-2 font-mono">
          STATUS: HEURISTIC (NOT ML)
        </div>
      </CardShell>
    );
  }

  const distanceText = landfall.distanceToLandKm !== undefined ? `${landfall.distanceToLandKm} km` : "Not available";

  return (
    <CardShell eyebrow="Landfall Prediction" title="Estimated Landfall">
      <div className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <Stat label="Location" value={`${landfall.latitude.toFixed(2)}°N ${landfall.longitude.toFixed(2)}°E`} small />
          <Stat label="Estimated Time" value={formatDateTime(landfall.estimated_time)} small />
          <Stat label="Predicted Wind" value={`${landfall.predictedWindKmh} km/h`} />
          <Stat label="Distance to Coast" value={distanceText} />
        </div>

        <div className="text-[10px] text-ink-faint border-t border-border/40 pt-2 font-mono">
          STATUS: HEURISTIC (NOT ML)
        </div>
      </div>
    </CardShell>
  );
}
