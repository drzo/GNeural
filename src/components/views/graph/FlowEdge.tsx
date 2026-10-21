import { BaseEdge, EdgeLabelRenderer, getBezierPath, type EdgeProps } from '@xyflow/react';
import { memo } from 'react';
import { fmtShape } from '../../../engine/shapes';
import { useStore } from '../../../store/useStore';
import { useNetworkContext } from './context';

function FlowEdgeView({ id, source, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, selected, markerEnd }: EdgeProps) {
  const net = useNetworkContext();
  const flowing = useStore((s) => s.flowing || s.training.status === 'running');
  const [path, lx, ly] = getBezierPath({ sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition });
  const shape = net.byId.get(source)?.outShape ?? null;
  const live = flowing && net.byId.has(source);

  return (
    <>
      <BaseEdge id={id} path={path} markerEnd={markerEnd} style={{ stroke: selected ? '#4f8ef7' : undefined, strokeWidth: selected ? 2.5 : 1.75 }} />
      {live && <path d={path} fill="none" stroke="#4f8ef7" strokeWidth={2.5} className="edge-flow" style={{ filter: 'drop-shadow(0 0 4px rgb(79 142 247 / 0.7))' }} />}
      {shape && (
        <EdgeLabelRenderer>
          <div
            className="nodrag nopan pointer-events-none absolute rounded-md border border-line bg-panel px-1.5 py-px font-mono text-[10px] text-faint"
            style={{ transform: `translate(-50%, -50%) translate(${lx}px, ${ly}px)` }}
          >
            {fmtShape(shape)}
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  );
}

export const FlowEdge = memo(FlowEdgeView);
