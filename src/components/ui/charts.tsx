import type { ReactNode } from 'react';

export const CHART = {
  grid: '#8b93a733',
  axis: '#8b93a7',
  blue: '#4f8ef7',
  violet: '#9b59f7',
  green: '#2ecc71',
  yellow: '#f39c12',
  red: '#e74c3c',
};

export const axisProps = { stroke: CHART.axis, fontSize: 11, tickLine: false, axisLine: false } as const;

export const tooltipProps = {
  contentStyle: {
    background: 'rgb(var(--raised))',
    border: '1px solid rgb(var(--line))',
    borderRadius: 8,
    fontSize: 12,
    color: 'rgb(var(--fg))',
  },
  labelStyle: { color: 'rgb(var(--muted))' },
  cursor: { stroke: CHART.axis, strokeOpacity: 0.3 },
};

export function ChartCard({ title, right, children, className = '' }: { title: ReactNode; right?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <div className={`card flex flex-col p-4 ${className}`}>
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-[13px] font-semibold">{title}</h3>
        {right}
      </div>
      <div className="min-h-0 flex-1">{children}</div>
    </div>
  );
}

export function Legend({ items }: { items: [string, string][] }) {
  return (
    <div className="flex items-center gap-3 text-[11px] text-muted">
      {items.map(([label, color]) => (
        <span key={label} className="flex items-center gap-1.5">
          <span className="h-0.5 w-3 rounded" style={{ background: color }} /> {label}
        </span>
      ))}
    </div>
  );
}
