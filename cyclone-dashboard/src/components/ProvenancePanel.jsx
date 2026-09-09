import CardShell from './CardShell';

/**
 * Renders the backend's `provenance` block verbatim so the honesty notes the
 * integration API already attaches are visible in the dashboard (they were
 * previously computed server-side but never displayed).
 *
 * `individual` is the /api/analyze provenance object:
 *   { pipeline, sources, notes[], reference_image, tabular{...} }
 */
export default function ProvenancePanel({ provenance, loading, title }) {
  if (loading || !provenance) {
    return null;
  }
  const noteBadge = (n) => {
    const low = /not\s+displayed|not\s+a\s+live|degenerate|placeholder|NOT_RUN/i;
    const warn = /heuristic|null|no\s+calibrated|not\s+ml/i;
    if (low.test(n)) return 'bg-risk-medium/10 text-risk-medium border-risk-medium/30';
    if (warn.test(n)) return 'bg-risk-low/10 text-risk-low border-risk-low/30';
    return 'bg-card-alt text-ink-soft border-border';
  };

  const sources = Object.entries(provenance.sources || {});

  return (
    <CardShell eyebrow="Trust &amp; Provenance" title={title || "Data & Model Provenance"}>
      <div className="space-y-3 text-[11.5px]">
        <div>
          <span className="text-ink-faint text-[9.5px] uppercase block tracking-wider font-semibold mb-1">Pipeline</span>
          <p className="font-mono text-[10.5px] text-ink-soft leading-relaxed">{provenance.pipeline}</p>
        </div>

        {provenance.reference_image ? (
          <div>
            <span className="text-ink-faint text-[9.5px] uppercase block tracking-wider font-semibold mb-1">Reference Frame</span>
            <p className="font-mono text-[10.5px] text-ink-soft break-all">{provenance.reference_image}</p>
          </div>
        ) : (
          <div>
            <span className="text-ink-faint text-[9.5px] uppercase block tracking-wider font-semibold mb-1">Image Source</span>
            <p className="font-mono text-[10.5px] text-ink-soft break-all">{provenance.image_source}</p>
          </div>
        )}

        <div>
          <span className="text-ink-faint text-[9.5px] uppercase block tracking-wider font-semibold mb-1">Sources</span>
          <div className="space-y-1">
            {sources.map(([k, v]) => (
              <div key={k} className="flex flex-col gap-0.5">
                <span className="font-semibold text-ink text-[10.5px] uppercase tracking-wide">{k}</span>
                <span className="text-ink-soft font-mono text-[10.5px] break-all">{v}</span>
              </div>
            ))}
          </div>
        </div>

        <div>
          <span className="text-ink-faint text-[9.5px] uppercase block tracking-wider font-semibold mb-1">Honesty Notes</span>
          <ul className="space-y-1.5">
            {(provenance.notes || []).map((n, i) => (
              <li key={i} className="flex items-start gap-2">
                <span className={`mt-0.5 w-1.5 h-1.5 rounded-full ${noteBadge(n)} border shrink-0`} />
                <span className="text-ink-soft leading-snug">{n}</span>
              </li>
            ))}
          </ul>
        </div>

        <div className="text-[10px] text-ink-faint border-t border-border/40 pt-2 font-mono">
          STATUS: VERIFIED LOCAL INFERENCE · reference frame, not live feed
        </div>
      </div>
    </CardShell>
  );
}
