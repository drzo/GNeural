import { ChevronDown, GripVertical, Plus, Search } from 'lucide-react';
import { useMemo, useState } from 'react';
import { CATEGORIES, LAYERS, type LayerDef } from '../../engine/catalog';
import { fmtShape } from '../../engine/shapes';
import { compact, layerColor } from '../../lib/format';
import { useNetwork } from '../../store/useNetwork';
import { useStore } from '../../store/useStore';
import { LayerBadge } from '../ui/LayerIcon';
import { Tooltip } from '../ui/primitives';

export const DRAG_TYPE = 'application/x-gneural-layer';

function PaletteItem({ def, beginner }: { def: LayerDef; beginner: boolean }) {
  const appendLayer = useStore((s) => s.appendLayer);
  const item = (
    <div
      draggable
      onDragStart={(e) => {
        e.dataTransfer.setData(DRAG_TYPE, def.type);
        e.dataTransfer.effectAllowed = 'copy';
      }}
      onDoubleClick={() => appendLayer(def.type)}
      className="group flex w-full cursor-grab items-center gap-2.5 rounded-lg px-2 py-1.5 transition-colors hover:bg-raised active:cursor-grabbing"
    >
      <LayerBadge type={def.type} color={def.type === 'Input' ? layerColor('Input', 'input') : def.color} size={26} />
      <div className="min-w-0 flex-1">
        <div className="truncate text-[13px] font-medium">{def.label}</div>
        {beginner && <div className="truncate text-[11px] text-faint">{def.description}</div>}
      </div>
      <button
        className="rounded p-0.5 text-faint opacity-0 transition hover:bg-line hover:text-fg group-hover:opacity-100"
        onClick={() => appendLayer(def.type)}
        title="Append to the end of the network"
      >
        <Plus size={14} />
      </button>
      <GripVertical size={13} className="text-faint/50" />
    </div>
  );
  return beginner ? (
    <Tooltip content={<><b className="font-semibold">{def.label}</b><br />{def.help}</>} side="right">
      <div className="w-full">{item}</div>
    </Tooltip>
  ) : (
    item
  );
}

function Palette() {
  const beginner = useStore((s) => s.mode === 'beginner');
  const [q, setQ] = useState('');
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});
  const filtered = useMemo(() => {
    const t = q.trim().toLowerCase();
    return t ? LAYERS.filter((l) => `${l.label} ${l.type} ${l.description}`.toLowerCase().includes(t)) : LAYERS;
  }, [q]);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="px-3 pb-2 pt-3">
        <div className="relative">
          <Search size={13} className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-faint" />
          <input className="input pl-8" placeholder="Search layers" value={q} onChange={(e) => setQ(e.target.value)} />
        </div>
        {beginner && <p className="mt-2 text-[11px] text-faint">Drag a layer onto the canvas, or double-click to append it.</p>}
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto px-2 pb-3">
        {CATEGORIES.map((cat) => {
          const items = filtered.filter((l) => l.category === cat);
          if (!items.length) return null;
          const closed = collapsed[cat] && !q;
          return (
            <div key={cat} className="mb-1">
              <button
                className="flex w-full items-center gap-1 px-2 py-1.5 text-[11px] font-medium uppercase tracking-wider text-faint hover:text-muted"
                onClick={() => setCollapsed((c) => ({ ...c, [cat]: !c[cat] }))}
              >
                <ChevronDown size={12} className={`transition-transform ${closed ? '-rotate-90' : ''}`} />
                {cat}
                <span className="ml-auto font-mono normal-case">{items.length}</span>
              </button>
              {!closed && items.map((def) => <PaletteItem key={def.type} def={def} beginner={beginner} />)}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function LayerTree() {
  const net = useNetwork();
  const nodes = useStore((s) => s.nodes);
  const selectedId = useStore((s) => s.selectedId);
  const select = useStore((s) => s.select);
  const orphanNodes = nodes.filter((n) => net.orphans.includes(n.id));

  return (
    <div className="flex max-h-[42%] min-h-[160px] flex-col border-t border-line">
      <div className="flex items-center justify-between px-4 pb-1.5 pt-3">
        <span className="label">Layers</span>
        <span className="font-mono text-[11px] text-faint">{compact(net.totalParams)} params</span>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto px-2 pb-2">
        {net.chain.map((l, i) => {
          const a = net.byId.get(l.id);
          const role = i === 0 ? 'input' : i === net.chain.length - 1 ? 'output' : 'hidden';
          return (
            <button
              key={l.id}
              onClick={() => select(l.id)}
              className={`group flex w-full items-center gap-2 rounded-md px-2 py-1 text-left transition-colors ${selectedId === l.id ? 'bg-brand-500/15' : 'hover:bg-raised'}`}
            >
              <span className="w-4 text-right font-mono text-[10px] text-faint">{i}</span>
              <span className="h-2 w-2 shrink-0 rounded-sm" style={{ background: layerColor(l.type, role) }} />
              <span className={`min-w-0 flex-1 truncate text-[12px] ${a?.error ? 'text-bad-400' : ''}`}>{l.name}</span>
              <span className="font-mono text-[10px] text-faint">{fmtShape(a?.outShape ?? null)}</span>
            </button>
          );
        })}
        {orphanNodes.length > 0 && (
          <div className="mt-2 border-t border-dashed border-line pt-2">
            <div className="px-2 pb-1 text-[10px] uppercase tracking-wider text-warn-400">Not connected</div>
            {orphanNodes.map((n) => (
              <button key={n.id} onClick={() => select(n.id)} className="flex w-full items-center gap-2 rounded-md px-2 py-1 text-left text-[12px] text-muted hover:bg-raised">
                <span className="h-2 w-2 rounded-sm bg-warn-500/70" />
                <span className="truncate">{n.data.name}</span>
              </button>
            ))}
          </div>
        )}
        {!net.chain.length && <p className="px-2 py-3 text-[12px] text-faint">Add an Input layer to start the network.</p>}
      </div>
    </div>
  );
}

export function LeftSidebar() {
  return (
    <aside className="flex w-[260px] shrink-0 flex-col border-r border-line bg-panel" data-tour="palette">
      <Palette />
      <LayerTree />
    </aside>
  );
}
