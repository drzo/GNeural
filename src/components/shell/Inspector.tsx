import { TriangleAlert as AlertTriangle, Code as Code2, Copy, MousePointerClick, ShieldAlert, Trash as Trash2, Circle as XCircle } from 'lucide-react';
import { layerDef, type ParamDef } from '../../engine/catalog';
import { fmtShape } from '../../engine/shapes';
import type { ParamValue } from '../../engine/types';
import { compact } from '../../lib/format';
import { useNetwork } from '../../store/useNetwork';
import { useStore } from '../../store/useStore';
import { LayerBadge } from '../ui/LayerIcon';
import { Empty, HelpTip, Section, Toggle } from '../ui/primitives';

function ParamField({ def, value, onChange }: { def: ParamDef; value: ParamValue | undefined; onChange(v: ParamValue): void }) {
  const label = (
    <div className="mb-1.5 flex items-center gap-1.5">
      <span className="text-[12px] font-medium text-muted">{def.label}</span>
      <HelpTip text={def.help} />
    </div>
  );
  switch (def.kind) {
    case 'bool':
      return (
        <label className="flex items-center justify-between gap-2 py-1">
          <span className="flex items-center gap-1.5 text-[12px] font-medium text-muted">{def.label}<HelpTip text={def.help} /></span>
          <Toggle checked={value === true} onChange={onChange} label={def.label} />
        </label>
      );
    case 'select':
      return (
        <div>
          {label}
          <select className="input" value={String(value ?? def.default)} onChange={(e) => onChange(e.target.value)}>
            {def.options!.map((o) => <option key={o} value={o}>{o}</option>)}
          </select>
        </div>
      );
    case 'int':
    case 'float': {
      const n = typeof value === 'number' ? value : Number(def.default);
      const slider = def.min !== undefined && def.max !== undefined && def.max - def.min <= 1;
      return (
        <div>
          {label}
          <div className="flex items-center gap-2">
            {slider && (
              <input type="range" className="flex-1 accent-brand-500" min={def.min} max={def.max} step={def.step ?? 0.01} value={n} onChange={(e) => onChange(Number(e.target.value))} />
            )}
            <input
              type="number"
              className={`input font-mono ${slider ? 'w-24' : ''}`}
              min={def.min}
              max={def.max}
              step={def.step ?? (def.kind === 'int' ? 1 : 0.01)}
              value={n}
              onChange={(e) => {
                const v = def.kind === 'int' ? Math.round(Number(e.target.value)) : Number(e.target.value);
                if (Number.isFinite(v)) onChange(v);
              }}
            />
          </div>
        </div>
      );
    }
    case 'shape':
      return (
        <div>
          {label}
          <input className="input font-mono" value={String(value ?? '')} onChange={(e) => onChange(e.target.value)} placeholder="e.g. 8,8,1" />
        </div>
      );
    case 'code':
      return (
        <div>
          {label}
          <textarea
            className="input h-28 resize-y py-2 font-mono text-[12px] leading-relaxed"
            spellCheck={false}
            value={String(value ?? '')}
            onChange={(e) => onChange(e.target.value)}
          />
        </div>
      );
  }
}

export function Inspector() {
  const selectedId = useStore((s) => s.selectedId);
  const node = useStore((s) => s.nodes.find((n) => n.id === s.selectedId));
  const update = useStore((s) => s.updateLayer);
  const remove = useStore((s) => s.removeNode);
  const addLayer = useStore((s) => s.addLayer);
  const expert = useStore((s) => s.mode === 'expert');
  const net = useNetwork();

  if (!selectedId || !node) {
    return (
      <Empty
        icon={<MousePointerClick size={22} />}
        title="Nothing selected"
        body="Click a layer on the canvas or in the layer list to see and edit its settings."
      />
    );
  }

  const def = layerDef(node.data.layerType);
  const a = net.byId.get(node.id);
  const p = node.data.params;
  const set = (key: string, v: ParamValue) => update(node.id, { params: { [key]: v } });
  const basic = def.params.filter((d) => !d.advanced);
  const advanced = def.params.filter((d) => d.advanced);

  return (
    <div className="fade-in" key={node.id}>
      <div className="flex items-start gap-3 border-b border-line px-4 py-4">
        <LayerBadge type={def.type} size={36} />
        <div className="min-w-0 flex-1">
          <input
            className="w-full rounded bg-transparent text-[15px] font-semibold outline-none ring-brand-500/30 focus:ring-2"
            value={node.data.name}
            onChange={(e) => update(node.id, { name: e.target.value })}
            aria-label="Layer name"
          />
          <div className="mt-0.5 flex items-center gap-1.5 text-[12px] text-faint">
            {def.label} layer <HelpTip text={def.help} />
          </div>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-px border-b border-line bg-line">
        {[
          ['In', fmtShape(a?.inShape ?? null)],
          ['Out', fmtShape(a?.outShape ?? null)],
          ['Params', a ? compact(a.params) : '-'],
        ].map(([k, v]) => (
          <div key={k} className="bg-panel px-3 py-2.5">
            <div className="label">{k}</div>
            <div className="mt-0.5 truncate font-mono text-[12px]">{a ? v : 'not connected'}</div>
          </div>
        ))}
      </div>

      {(a?.error || a?.warnings.length) && (
        <div className="space-y-2 border-b border-line px-4 py-3">
          {a.error && (
            <div className="flex gap-2 rounded-lg bg-bad-500/10 px-3 py-2 text-[12px] text-bad-400"><XCircle size={14} className="mt-0.5 shrink-0" />{a.error}</div>
          )}
          {a.warnings.map((w) => (
            <div key={w} className="flex gap-2 rounded-lg bg-warn-500/10 px-3 py-2 text-[12px] text-warn-400"><AlertTriangle size={14} className="mt-0.5 shrink-0" />{w}</div>
          ))}
        </div>
      )}

      {basic.length > 0 && (
        <Section title="Settings">
          <div className="space-y-3.5">{basic.map((d) => <ParamField key={d.key} def={d} value={p[d.key]} onChange={(v) => set(d.key, v)} />)}</div>
        </Section>
      )}

      {def.type === 'Custom' && (
        <Section title={<span className="flex items-center gap-1.5"><Code2 size={12} /> Code execution</span>}>
          <div className={`flex items-start gap-3 rounded-lg px-3 py-2.5 ${p.trusted ? 'bg-ok-500/10' : 'bg-warn-500/10'}`}>
            <ShieldAlert size={16} className={`mt-0.5 shrink-0 ${p.trusted ? 'text-ok-400' : 'text-warn-400'}`} />
            <div className="flex-1 text-[12px] text-muted">
              {p.trusted ? 'This code runs in your browser during training and inference.' : 'Disabled: the layer passes values through unchanged until you enable it. Only enable code you wrote or trust.'}
            </div>
            <Toggle checked={p.trusted === true} onChange={(v) => set('trusted', v)} label="Enable custom code" />
          </div>
        </Section>
      )}

      {expert && (
        <>
          {advanced.length > 0 && (
            <Section title="Regularization and initialization">
              <div className="space-y-3.5">{advanced.map((d) => <ParamField key={d.key} def={d} value={p[d.key]} onChange={(v) => set(d.key, v)} />)}</div>
            </Section>
          )}
          {def.type !== 'Input' && (
            <Section title="Shape override">
              <ParamField
                def={{ key: 'declaredShape', label: 'Expected output shape', kind: 'shape', default: '', help: 'Assert the output shape. Leave empty to infer. A mismatch is flagged as an error so mistakes surface early.' }}
                value={p.declaredShape}
                onChange={(v) => set('declaredShape', v)}
              />
            </Section>
          )}
        </>
      )}
      {!expert && advanced.length > 0 && (
        <p className="px-4 py-3 text-[11px] text-faint">Switch to Expert mode to edit regularization and initialization.</p>
      )}

      <div className="flex gap-2 px-4 py-4">
        <button className="btn-outline flex-1" onClick={() => addLayer(def.type, { x: node.position.x + 40, y: node.position.y + 140 }, p, `${node.data.name} copy`)}>
          <Copy size={13} /> Duplicate
        </button>
        <button className="btn-danger flex-1" onClick={() => remove(node.id)}>
          <Trash2 size={13} /> Delete
        </button>
      </div>
    </div>
  );
}
