import { Play, SlidersHorizontal } from 'lucide-react';
import { useEffect, useState } from 'react';
import { useStore } from '../../store/useStore';
import { Inspector } from './Inspector';
import { RunPanel } from './RunPanel';

export function RightPanel() {
  const selectedId = useStore((s) => s.selectedId);
  const [tab, setTab] = useState<'inspect' | 'run'>('inspect');
  useEffect(() => {
    if (selectedId) setTab('inspect');
  }, [selectedId]);

  return (
    <aside className="flex w-[320px] shrink-0 flex-col border-l border-line bg-panel" data-tour="inspector">
      <div className="flex h-11 shrink-0 items-stretch border-b border-line px-2">
        {([
          ['inspect', 'Properties', SlidersHorizontal],
          ['run', 'Run', Play],
        ] as const).map(([id, label, Icon]) => (
          <button
            key={id}
            onClick={() => setTab(id)}
            className={`relative flex items-center gap-1.5 px-3 text-[13px] font-medium transition-colors ${tab === id ? 'text-fg' : 'text-faint hover:text-muted'}`}
          >
            <Icon size={13} /> {label}
            {tab === id && <span className="absolute inset-x-2 -bottom-px h-0.5 rounded-full bg-brand-500" />}
          </button>
        ))}
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto">{tab === 'inspect' ? <Inspector /> : <RunPanel />}</div>
    </aside>
  );
}
