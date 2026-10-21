import { Circle as HelpCircle, X } from 'lucide-react';
import { useEffect, useRef, useState, type ReactNode } from 'react';
import { createPortal } from 'react-dom';

export function Tooltip({ content, children, side = 'top', disabled }: { content: ReactNode; children: ReactNode; side?: 'top' | 'right' | 'bottom'; disabled?: boolean }) {
  const ref = useRef<HTMLSpanElement>(null);
  const [pos, setPos] = useState<{ x: number; y: number } | null>(null);
  const timer = useRef<number>();
  const show = () => {
    if (disabled) return;
    timer.current = window.setTimeout(() => {
      const r = ref.current?.getBoundingClientRect();
      if (!r) return;
      if (side === 'right') setPos({ x: r.right + 10, y: r.top + r.height / 2 });
      else if (side === 'bottom') setPos({ x: r.left + r.width / 2, y: r.bottom + 8 });
      else setPos({ x: r.left + r.width / 2, y: r.top - 8 });
    }, 280);
  };
  const hide = () => {
    window.clearTimeout(timer.current);
    setPos(null);
  };
  useEffect(() => () => window.clearTimeout(timer.current), []);
  const transform = side === 'right' ? 'translate(0,-50%)' : side === 'bottom' ? 'translate(-50%,0)' : 'translate(-50%,-100%)';
  return (
    <span ref={ref} onMouseEnter={show} onMouseLeave={hide} onMouseDown={hide} className="inline-flex">
      {children}
      {pos &&
        createPortal(
          <div
            className="fade-in pointer-events-none fixed z-[100] max-w-[260px] rounded-lg border border-line bg-raised px-3 py-2 text-[12px] leading-relaxed text-fg shadow-pop"
            style={{ left: pos.x, top: pos.y, transform }}
          >
            {content}
          </div>,
          document.body,
        )}
    </span>
  );
}

export function HelpTip({ text }: { text: string }) {
  return (
    <Tooltip content={text}>
      <HelpCircle size={13} className="cursor-help text-faint transition-colors hover:text-brand-400" />
    </Tooltip>
  );
}

export function Modal({ title, subtitle, onClose, children, width = 720 }: { title: string; subtitle?: string; onClose(): void; children: ReactNode; width?: number }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);
  return createPortal(
    <div className="fade-in fixed inset-0 z-50 flex items-center justify-center bg-black/55 p-8 backdrop-blur-sm" onMouseDown={onClose}>
      <div
        className="pop-in flex max-h-full w-full flex-col overflow-hidden rounded-2xl border border-line bg-panel shadow-pop"
        style={{ maxWidth: width }}
        onMouseDown={(e) => e.stopPropagation()}
      >
        <div className="flex items-start justify-between gap-4 border-b border-line px-6 py-4">
          <div>
            <h2 className="text-[17px] font-semibold">{title}</h2>
            {subtitle && <p className="mt-1 text-[13px] text-muted">{subtitle}</p>}
          </div>
          <button className="icon-btn -mr-2" onClick={onClose} aria-label="Close">
            <X size={16} />
          </button>
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto">{children}</div>
      </div>
    </div>,
    document.body,
  );
}

export function Toggle({ checked, onChange, label }: { checked: boolean; onChange(v: boolean): void; label?: string }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      onClick={() => onChange(!checked)}
      className={`relative h-[18px] w-8 shrink-0 rounded-full transition-colors ${checked ? 'bg-brand-500' : 'bg-line'}`}
    >
      <span className={`absolute top-[2px] h-[14px] w-[14px] rounded-full bg-white shadow transition-all ${checked ? 'left-[16px]' : 'left-[2px]'}`} />
    </button>
  );
}

export function Segmented<T extends string>({ value, options, onChange, size = 'md' }: { value: T; options: { value: T; label: ReactNode; title?: string }[]; onChange(v: T): void; size?: 'sm' | 'md' }) {
  return (
    <div className="inline-flex rounded-lg border border-line bg-bg p-0.5">
      {options.map((o) => (
        <button
          key={o.value}
          title={o.title}
          onClick={() => onChange(o.value)}
          className={`inline-flex items-center gap-1.5 rounded-md font-medium transition-all ${size === 'sm' ? 'h-6 px-2 text-[11px]' : 'h-7 px-2.5 text-[12px]'} ${
            value === o.value ? 'bg-raised text-fg shadow-sm' : 'text-faint hover:text-muted'
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Section({ title, right, children, className = '' }: { title: ReactNode; right?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={`border-b border-line px-4 py-4 ${className}`}>
      <div className="mb-3 flex items-center justify-between gap-2">
        <h3 className="label">{title}</h3>
        {right}
      </div>
      {children}
    </section>
  );
}

export function Empty({ icon, title, body, action }: { icon: ReactNode; title: string; body: string; action?: ReactNode }) {
  return (
    <div className="flex h-full flex-col items-center justify-center p-10 text-center">
      <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-2xl border border-line bg-raised text-faint">{icon}</div>
      <h3 className="text-[15px] font-semibold">{title}</h3>
      <p className="mt-1.5 max-w-sm text-[13px] text-muted">{body}</p>
      {action && <div className="mt-5">{action}</div>}
    </div>
  );
}

export function Stat({ label, value, sub, tone }: { label: string; value: ReactNode; sub?: ReactNode; tone?: 'ok' | 'warn' | 'bad' | 'brand' }) {
  const color = tone === 'ok' ? 'text-ok-400' : tone === 'warn' ? 'text-warn-400' : tone === 'bad' ? 'text-bad-400' : tone === 'brand' ? 'text-brand-400' : 'text-fg';
  return (
    <div className="card px-4 py-3">
      <div className="label">{label}</div>
      <div className={`mt-1 font-mono text-[20px] font-semibold tabular-nums ${color}`}>{value}</div>
      {sub && <div className="mt-0.5 text-[11px] text-faint">{sub}</div>}
    </div>
  );
}
