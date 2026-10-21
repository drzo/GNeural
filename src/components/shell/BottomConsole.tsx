import { ChevronDown, ChevronUp, Terminal, Trash as Trash2 } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { clock } from '../../lib/format';
import { useStore, type LogLevel } from '../../store/useStore';

const LEVEL_STYLE: Record<LogLevel, string> = {
  info: 'text-brand-400',
  success: 'text-ok-400',
  warn: 'text-warn-400',
  error: 'text-bad-400',
};

const FILTERS: (LogLevel | 'all')[] = ['all', 'info', 'success', 'warn', 'error'];

export function BottomConsole() {
  const logs = useStore((s) => s.logs);
  const clear = useStore((s) => s.clearLogs);
  const [filter, setFilter] = useState<LogLevel | 'all'>('all');
  const [open, setOpen] = useState(true);
  const scroller = useRef<HTMLDivElement>(null);
  const shown = useMemo(() => (filter === 'all' ? logs : logs.filter((l) => l.level === filter)), [logs, filter]);
  const counts = useMemo(() => {
    const c: Record<string, number> = { warn: 0, error: 0 };
    for (const l of logs) c[l.level] = (c[l.level] ?? 0) + 1;
    return c;
  }, [logs]);

  useEffect(() => {
    const el = scroller.current;
    if (el && el.scrollHeight - el.scrollTop - el.clientHeight < 80) el.scrollTop = el.scrollHeight;
  }, [shown]);

  return (
    <section className={`flex shrink-0 flex-col border-t border-line bg-panel transition-[height] duration-200 ${open ? 'h-[180px]' : 'h-9'}`} data-tour="console">
      <div className="flex h-9 shrink-0 items-center gap-3 px-3">
        <button className="flex items-center gap-1.5 text-[12px] font-medium text-muted hover:text-fg" onClick={() => setOpen(!open)}>
          <Terminal size={13} /> Console
          {open ? <ChevronDown size={13} /> : <ChevronUp size={13} />}
        </button>
        {counts.error > 0 && <span className="chip bg-bad-500/15 text-bad-400">{counts.error} errors</span>}
        {counts.warn > 0 && <span className="chip bg-warn-500/15 text-warn-400">{counts.warn} warnings</span>}
        {open && (
          <div className="ml-auto flex items-center gap-1">
            {FILTERS.map((f) => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={`rounded px-2 py-0.5 text-[11px] capitalize transition-colors ${filter === f ? 'bg-raised text-fg' : 'text-faint hover:text-muted'}`}
              >
                {f}
              </button>
            ))}
            <span className="mx-1 h-4 w-px bg-line" />
            <button className="icon-btn h-6 w-6" onClick={clear} title="Clear console"><Trash2 size={13} /></button>
          </div>
        )}
      </div>
      {open && (
        <div ref={scroller} className="min-h-0 flex-1 overflow-y-auto px-3 pb-2 font-mono text-[12px] leading-[1.6]">
          {shown.map((l) => (
            <div key={l.id} className="fade-in flex gap-3 rounded px-1 hover:bg-raised/60">
              <span className="shrink-0 text-faint">{clock(l.ts)}</span>
              <span className={`w-14 shrink-0 uppercase ${LEVEL_STYLE[l.level]}`}>{l.level}</span>
              <span className="w-16 shrink-0 truncate text-faint">{l.source}</span>
              <span className={l.level === 'error' ? 'text-bad-400' : l.level === 'warn' ? 'text-warn-400' : 'text-fg/90'}>{l.msg}</span>
            </div>
          ))}
          {!shown.length && <div className="py-3 text-faint">{logs.length ? 'No messages at this level.' : 'Messages from building, training and saving appear here.'}</div>}
        </div>
      )}
    </section>
  );
}
