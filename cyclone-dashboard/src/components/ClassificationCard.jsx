import CardShell, { Stat, CardSkeleton } from './CardShell';

export default function ClassificationCard({ classification, loading }){
  if(loading || !classification){
    return <CardShell eyebrow="Intensity Classification" title="Intensity Classification"><CardSkeleton/></CardShell>;
  }
  const { category, scale, windSpeedKmh, pressureHpa, confidence, structuralPattern } = classification;
  return (
    <CardShell
      eyebrow="Intensity Classification"
      title="Intensity Classification"
      right={structuralPattern && (
        <span className="text-[10.5px] font-semibold text-accent-strong bg-accent-soft border border-accent/25 rounded-full px-2.5 py-1">
          {structuralPattern}
        </span>
      )}
    >
      <div className="mb-4">
        <div className="text-[10.5px] uppercase tracking-wide text-ink-faint font-semibold mb-1">Category — {scale} Scale</div>
        <div className="text-[15.5px] font-semibold text-ink">{category}</div>
      </div>
      <div className="grid grid-cols-3 gap-4">
        <Stat label="Wind Speed" value={`${windSpeedKmh} km/h`} small />
        <Stat label="Pressure" value={`${pressureHpa} hPa`} small />
        <Stat label="Confidence" value={`${confidence}%`} small />
      </div>
    </CardShell>
  );
}
