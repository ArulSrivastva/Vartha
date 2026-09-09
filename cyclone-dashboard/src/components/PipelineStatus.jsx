import CardShell from './CardShell';

export default function PipelineStatus({ loading }) {
  const getBadgeStyle = (status) => {
    switch (status) {
      case 'READY':
        return 'bg-risk-low/10 text-risk-low border-risk-low/30';
      case 'DEMO':
        return 'bg-accent-strong/10 text-accent-strong border-accent-strong/20';
      case 'PROCESSING':
        return 'bg-risk-medium/10 text-risk-medium border-risk-medium/30 animate-pulse';
      case 'WARNING':
        return 'bg-risk-medium/10 text-risk-medium border-risk-medium/30';
      case 'ERROR':
        return 'bg-risk-high/10 text-risk-high border-risk-high/30';
      default:
        return 'bg-card-alt text-ink-soft border-border';
    }
  };

  const pipeline = [
    { name: "Satellite Imagery Input", status: loading ? "PROCESSING" : "READY", desc: "Calibrated INSAT-3D Frame" },
    { name: "Atmospheric Feature Matrix", status: loading ? "PROCESSING" : "READY", desc: "Multi-parameter ERA5 Ingestion" },
    { name: "Recurrent Track Forecaster", status: loading ? "PROCESSING" : "READY", desc: "Multi-horizon GRU Displacement" },
    { name: "Baseline Integrity Check", status: "READY", desc: "Audited Validation Benchmarks" },
    { name: "Inference API Gateway", status: loading ? "PROCESSING" : "READY", desc: "Synchronized Real-time Output" },
  ];

  return (
    <CardShell eyebrow="Pipeline Health" title="Data Pipeline Status">
      <div className="space-y-3">
        <div className="divide-y divide-border-soft -mx-5 px-5">
          {pipeline.map((step, i) => (
            <div key={i} className="flex items-center justify-between py-2 text-[12px] gap-4">
              <div>
                <span className="font-semibold text-ink block">{step.name}</span>
                <span className="text-[10px] text-ink-faint block">{step.desc}</span>
              </div>
              <span className={`px-2 py-0.5 rounded text-[10px] font-bold border ${getBadgeStyle(step.status)}`}>
                {step.status}
              </span>
            </div>
          ))}
        </div>
      </div>
    </CardShell>
  );
}
