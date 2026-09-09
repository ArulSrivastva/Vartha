export default function CardShell({ eyebrow, title, right, children, className = '' }){
  return (
    <div className={`card p-5 overflow-hidden min-w-0 ${className}`}>
      <div className="flex items-start justify-between gap-3 mb-4 min-w-0">
        <div className="min-w-0">
          {eyebrow && <div className="text-[11px] uppercase tracking-wider text-ink-soft font-semibold break-words">{eyebrow}</div>}
          {title && <h3 className="font-serif text-[15px] font-semibold text-ink mt-0.5 break-words">{title}</h3>}
        </div>
        {right}
      </div>
      {children}
    </div>
  );
}

export function Stat({ label, value, small, valueClassName = '' }){
  return (
    <div>
      <div className="text-[10.5px] uppercase tracking-wide text-ink-faint font-semibold mb-1">{label}</div>
      <div className={`${small ? 'text-[13.5px]' : 'text-xl'} font-semibold text-ink tabular-nums ${valueClassName}`}>{value}</div>
    </div>
  );
}

export function CardSkeleton(){
  return (
    <div className="space-y-3 animate-pulse">
      <div className="h-4 w-2/3 bg-card-alt rounded" />
      <div className="h-8 w-1/2 bg-card-alt rounded" />
      <div className="h-4 w-full bg-card-alt rounded" />
    </div>
  );
}
