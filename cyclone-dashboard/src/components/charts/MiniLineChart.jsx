import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';

export default function MiniLineChart({ title, unit, data, color = '#B98671' }){
  return (
    <div className="card p-5">
      <div className="text-[11px] uppercase tracking-wider text-ink-soft font-semibold mb-1">{title}</div>
      <div className="text-[13px] text-ink-faint mb-3">Last 24 hours{unit ? ` · ${unit}` : ''}</div>
      <div className="h-[140px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 4, right: 8, left: -18, bottom: 0 }}>
            <CartesianGrid stroke="#ECE0D4" vertical={false} />
            <XAxis dataKey="t" tick={{ fontSize: 10.5, fill: '#9C9187' }} axisLine={{ stroke: '#E4D6C9' }} tickLine={false} />
            <YAxis tick={{ fontSize: 10.5, fill: '#9C9187' }} axisLine={false} tickLine={false} width={34} />
            <Tooltip
              contentStyle={{ fontSize: 12, fontFamily: 'IBM Plex Sans, sans-serif', borderRadius: 10, border: '1px solid #E4D6C9', background: '#FCF9F5' }}
              labelStyle={{ color: '#6B615A' }}
            />
            <Line type="monotone" dataKey="value" stroke={color} strokeWidth={2} dot={{ r: 3, fill: color, strokeWidth: 0 }} activeDot={{ r: 5 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
