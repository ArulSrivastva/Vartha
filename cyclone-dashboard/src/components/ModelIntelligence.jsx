import CardShell from './CardShell';
import championModel from '../data/champion_model.json';
import exp005Metrics from '../data/exp005_metrics.json';
import exp005Config from '../data/exp005_config.json';

export default function ModelIntelligence() {
  const modelName = championModel.experiment_id || "EXP005";
  const architecture = (exp005Config.model?.toUpperCase() || "GRU") + " + " + (exp005Config.loss?.charAt(0).toUpperCase() + exp005Config.loss?.slice(1) || "Huber") + " Loss";
  const paramsCount = exp005Metrics.parameter_count ? exp005Metrics.parameter_count.toLocaleString() : "89,577";
  const validationLoss = exp005Metrics.best_val_loss ? exp005Metrics.best_val_loss.toFixed(6) : "0.033827";
  const validationScore = exp005Metrics.validation_primary_score ? exp005Metrics.validation_primary_score.toFixed(4) : "113.0741";
  
  const originalFeatures = [
    "latitude", "longitude", "wind speed", "pressure", "SST", "wind-u", "wind-v"
  ];
  
  const engineeredFeatures = [
    "delta_lat", "delta_lon", "movement_speed", "movement_direction", 
    "wind_change", "pressure_change", "sst_change", 
    "environmental_wind_speed", "environmental_wind_direction"
  ];

  return (
    <CardShell eyebrow="Operational ML Model" title="AI Model Intelligence">
      <div className="space-y-4">
        {/* Model Spec Grid */}
        <div className="grid grid-cols-2 gap-3 p-3 bg-card-alt rounded-xl border border-border/50 text-[12px]">
          <div>
            <span className="text-ink-faint text-[9.5px] uppercase block tracking-wider font-semibold">Champion Model</span>
            <span className="text-ink font-bold font-mono text-[13px]">{modelName}</span>
          </div>
          
          <div>
            <span className="text-ink-faint text-[9.5px] uppercase block tracking-wider font-semibold">Architecture</span>
            <span className="text-ink font-bold font-mono text-[13px]">{architecture}</span>
          </div>

          <div>
            <span className="text-ink-faint text-[9.5px] uppercase block tracking-wider font-semibold">Parameters</span>
            <span className="text-ink font-mono">{paramsCount}</span>
          </div>

          <div>
            <span className="text-ink-faint text-[9.5px] uppercase block tracking-wider font-semibold">Best Val Loss</span>
            <span className="text-ink font-mono">{validationLoss}</span>
          </div>

          <div className="col-span-2 border-t border-border/50 pt-2 mt-1">
            <span className="text-ink-faint text-[9.5px] uppercase block tracking-wider font-semibold">Selection Metric</span>
            <span className="text-ink font-mono text-[11px] leading-snug">
              Primary Validation Score: <strong className="text-accent-strong">{validationScore}</strong> (mean 6h/12h/24h track errors)
            </span>
          </div>
        </div>

        {/* Feature Contract */}
        <div className="space-y-2">
          <h4 className="text-[12.5px] font-bold text-ink">16-Feature Input Contract (per timestep)</h4>
          
          <div className="text-[11px] space-y-2">
            <div>
              <span className="font-semibold text-ink-soft block mb-1">7 Original Core Features:</span>
              <div className="flex flex-wrap gap-1">
                {originalFeatures.map(f => (
                  <span key={f} className="bg-card px-2 py-0.5 border border-border rounded font-mono text-[10px] text-ink">
                    {f}
                  </span>
                ))}
              </div>
            </div>

            <div>
              <span className="font-semibold text-ink-soft block mb-1">9 Causal Engineered Features (t - 6h):</span>
              <div className="flex flex-wrap gap-1">
                {engineeredFeatures.map(f => (
                  <span key={f} className="bg-accent-soft/40 px-2 py-0.5 border border-accent/25 rounded font-mono text-[10px] text-accent-strong">
                    {f}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Notice on Causal Flow */}
        <div className="p-3 bg-mist/30 border border-border-soft rounded-lg text-[11px] text-ink-soft leading-relaxed">
          <span className="font-semibold text-ink block mb-0.5">Causal Zero-Fill Policy:</span>
          For difference/trend features requiring a predecessor, the first available history timestep is zero-filled because no predecessor exists within the model's causal input window. This ensures no future information is leaked.
        </div>
      </div>
    </CardShell>
  );
}
