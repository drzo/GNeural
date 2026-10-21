import { Handle, Position, type NodeProps } from '@xyflow/react';
import { TriangleAlert as AlertTriangle, Circle as XCircle } from 'lucide-react';
import { memo } from 'react';
import { layerDef } from '../../../engine/catalog';
import { fmtShape } from '../../../engine/shapes';
import { compact, layerColor } from '../../../lib/format';
import type { LayerNode } from '../../../store/graph';
import { useStore } from '../../../store/useStore';
import { LayerIcon } from '../../ui/LayerIcon';
import { useNetworkContext } from './context';

const SUMMARY_KEYS = ['units', 'filters', 'kernel', 'heads', 'rate', 'pool', 'shape', 'target', 'activation', 'cell'];

function summary(params: LayerNode['data']['params']) {
  return SUMMARY_KEYS.filter((k) => params[k] !== undefined && params[k] !== '')
    .slice(0, 3)
    .map((k) => `${k} ${params[k]}`);
}

function LayerNodeView({ id, data, selected }: NodeProps<LayerNode>) {
  const net = useNetworkContext();
  const training = useStore((s) => s.training.status === 'running');
  const beginner = useStore((s) => s.mode === 'beginner');
  const def = layerDef(data.layerType);
  const idx = net.chain.findIndex((c) => c.id === id);
  const inChain = idx >= 0;
  const role = idx === 0 ? 'input' : idx === net.chain.length - 1 ? 'output' : 'hidden';
  const color = layerColor(data.layerType, role);
  const a = net.byId.get(id);
  const pulse = training && inChain && a && a.params > 0;

  return (
    <div
      className={`group relative w-[210px] rounded-xl border bg-panel shadow-node transition-[border-color,transform] duration-150 ${
        selected ? 'border-brand-500 ring-2 ring-brand-500/25' : 'border-line hover:border-faint/60'
      } ${inChain ? '' : 'opacity-60'} ${pulse ? 'node-training' : ''}`}
      style={{ ['--pulse' as string]: `${color}66` }}
    >
      <div className="absolute inset-y-0 left-0 w-[4px] rounded-l-xl" style={{ background: color }} />
      {data.layerType !== 'Input' && <Handle type="target" position={Position.Left} className="!h-3 !w-3 !border-2" />}
      <Handle type="source" position={Position.Right} className="!h-3 !w-3 !border-2" />

      <div className="flex items-center gap-2.5 py-2.5 pl-4 pr-3">
        <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg" style={{ background: `${color}1f`, color }}>
          <LayerIcon type={data.layerType} size={15} />
        </span>
        <div className="min-w-0 flex-1">
          <div className="truncate text-[13px] font-semibold leading-tight">{data.name}</div>
          <div className="truncate text-[11px] text-faint">{def.label}{role === 'output' && idx > 0 ? ' / output' : ''}</div>
        </div>
        {a?.error ? (
          <XCircle size={15} className="shrink-0 text-bad-500" />
        ) : a?.warnings.length ? (
          <AlertTriangle size={14} className="shrink-0 text-warn-500" />
        ) : null}
      </div>

      <div className="flex items-center justify-between gap-2 border-t border-line/70 py-1.5 pl-4 pr-3 font-mono text-[11px]">
        <span className={a?.error ? 'text-bad-400' : 'text-muted'}>{inChain ? fmtShape(a?.outShape ?? null) : 'disconnected'}</span>
        {a && a.params > 0 && <span className="text-faint">{compact(a.params)}p</span>}
      </div>
      {!beginner && summary(data.params).length > 0 && (
        <div className="truncate border-t border-line/70 py-1 pl-4 pr-3 font-mono text-[10px] text-faint">{summary(data.params).join(' / ')}</div>
      )}
      {a?.error && selected && (
        <div className="absolute left-0 right-0 top-full z-10 mt-2 rounded-lg border border-bad-500/40 bg-panel px-2.5 py-1.5 text-[11px] text-bad-400 shadow-node">{a.error}</div>
      )}
    </div>
  );
}

export const LayerNodeComponent = memo(LayerNodeView);
