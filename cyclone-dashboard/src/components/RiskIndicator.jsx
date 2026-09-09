const COLORS = { HIGH: '#B85C4C', MEDIUM: '#C7A05C', MODERATE: '#C7A05C', LOW: '#7A9A7E' };

function polar(cx, cy, r, angleDeg){
  const rad = angleDeg * Math.PI / 180;
  return [cx + r * Math.cos(rad), cy - r * Math.sin(rad)];
}
function arcPath(cx, cy, r, startDeg, endDeg){
  const [sx, sy] = polar(cx, cy, r, startDeg);
  const [ex, ey] = polar(cx, cy, r, endDeg);
  const largeArc = Math.abs(startDeg - endDeg) > 180 ? 1 : 0;
  return `M ${sx} ${sy} A ${r} ${r} 0 ${largeArc} 1 ${ex} ${ey}`;
}

export default function RiskIndicator({ risk, classification, landfall, loading }){
  if(loading || !risk){
    return (
      <div className="card p-5 animate-pulse bg-card-alt">
        <div className="text-[11px] uppercase tracking-wider text-ink-soft font-semibold">AI Risk Indicator</div>
        <div className="h-32 mt-4 bg-card-alt rounded" />
      </div>
    );
  }
  const { score, level } = risk;
  const color = COLORS[level] || COLORS.LOW;
  const cx = 110, cy = 100, r = 82;
  const t = Math.max(0, Math.min(100, score)) / 100;
  const bg = arcPath(cx, cy, r, 180, 0);
  const fg = arcPath(cx, cy, r, 180, 180 - t * 180);

  const wind = classification?.windSpeedKmh;
  const pressure = classification?.pressureHpa;
  const dist = landfall?.distanceToLandKm;
  const windTxt = wind != null
    ? `${wind} km/h (${wind >= 105 ? 'HIGH' : wind >= 62 ? 'ELEVATED' : 'MODERATE'})`
    : 'n/a';
  const pressureTxt = pressure != null
    ? `${pressure} hPa ${pressure <= 950 ? '(CRITICAL)' : pressure <= 985 ? '(ELEVATED)' : '(MODERATE)'}`
    : 'n/a';
  const distTxt = dist != null
    ? `${dist} km (${dist <= 50 ? 'IMPACT ZONE' : dist <= 200 ? 'CLOSE' : 'DISTANT'})`
    : 'n/a (no estimate)';

  const contributors = [
    { name: "Sustained Winds (observed)", val: windTxt },
    { name: "Central Pressure (observed)", val: pressureTxt },
    { name: "Coast Proximity", val: distTxt },
  ];

  return (
    <div className="card p-5">
      <div className="flex items-start justify-between mb-1">
        <div>
          <div className="text-[11px] uppercase tracking-wider text-ink-soft font-semibold">AI Risk Indicator</div>
          <h3 className="font-serif text-[15px] font-semibold text-ink mt-0.5">Composite Risk Score</h3>
        </div>
        <span
          className="text-[10.5px] font-semibold px-2.5 py-1 rounded-full"
          style={{ color, backgroundColor: `${color}1F` }}
        >
          {level}
        </span>
      </div>

      <div className="flex flex-col sm:flex-row items-center justify-between gap-4 mt-2">
        {/* SVG gauge */}
        <div className="w-40 sm:w-44 flex-shrink-0">
          <svg viewBox="0 0 220 120" className="w-full">
            <path d={bg} fill="none" stroke="#ECE0D4" strokeWidth="14" strokeLinecap="round" />
            <path d={fg} fill="none" stroke={color} strokeWidth="14" strokeLinecap="round" />
            <text x={cx} textAnchor="middle" y={cy - 6} className="tabular-nums"
              style={{ fontFamily: 'IBM Plex Serif, serif', fontSize: 34, fontWeight: 600, fill: '#2E2A27' }}>
              {score}
            </text>
            <text x={cx} y={cy + 16} textAnchor="middle"
              style={{ fontFamily: 'IBM Plex Sans, sans-serif', fontSize: 11, fill: '#9C9187' }}>
              out of 100
            </text>
          </svg>
        </div>

        {/* Contributing Factors */}
        <div className="flex-1 w-full text-[11px] space-y-1.5 p-2.5 bg-card-alt/60 border border-border/40 rounded-lg">
          <span className="text-[9.5px] uppercase tracking-wider text-ink-faint font-semibold block mb-1">Contributing Factors</span>
          {contributors.map((c, i) => (
            <div key={i} className="flex justify-between items-center py-0.5 border-b border-border-soft last:border-b-0">
              <span className="text-ink-soft">{c.name}</span>
              <span className="font-mono text-ink font-semibold">{c.val}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Formula breakdown */}
      <div className="mt-3 bg-card-alt/80 border border-border/60 rounded-lg p-2.5 space-y-1">
        <span className="text-[9px] uppercase tracking-wider text-ink-faint font-semibold block">
          Composite Threat Formula:
        </span>
        <div className="font-mono text-[10.5px] text-ink flex flex-wrap items-center gap-1.5 leading-tight">
          <span className="text-accent-strong bg-accent-soft/50 border border-accent/25 px-1.5 py-0.5 rounded font-bold">
            40% Wind Factor
          </span>
          <span className="text-ink-faint font-bold">+</span>
          <span className="text-accent-strong bg-accent-soft/50 border border-accent/25 px-1.5 py-0.5 rounded font-bold">
            35% Pressure Deficit
          </span>
          <span className="text-ink-faint font-bold">+</span>
          <span className="text-accent-strong bg-accent-soft/50 border border-accent/25 px-1.5 py-0.5 rounded font-bold">
            25% Coast Proximity
          </span>
        </div>
      </div>

      <div className="mt-3 pt-2 border-t border-border-soft text-[10px] text-ink-faint leading-relaxed flex flex-col gap-1">
        <p className="italic">
          Deterministic server-side heuristic, not an ML model or an official warning. Wind/pressure are the latest observed values; refer to IMD bulletins for official guidance.
        </p>
        <div className="flex justify-between items-center font-mono text-[9px]">
          <span>STATUS: HEURISTIC (NOT ML)</span>
        </div>
      </div>
    </div>
  );
}
