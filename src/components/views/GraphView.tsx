import {
  Background, BackgroundVariant, Controls, MarkerType, MiniMap, ReactFlow, ReactFlowProvider, useReactFlow,
  type Connection, type Edge, type IsValidConnection,
} from '@xyflow/react';
import { LayoutGrid, Maximize2, MousePointer2, Sparkles, Circle as XCircle } from 'lucide-react';
import { useCallback, useMemo, type DragEvent } from 'react';
import { layerDef } from '../../engine/catalog';
import type { LayerType } from '../../engine/types';
import { TEMPLATES } from '../../data/templates';
import { compact, layerColor } from '../../lib/format';
import type { LayerNode } from '../../store/graph';
import { useNetwork } from '../../store/useNetwork';
import { useStore } from '../../store/useStore';
import { DRAG_TYPE } from '../shell/LeftSidebar';
import { NetworkContext } from './graph/context';
import { FlowEdge } from './graph/FlowEdge';
import { LayerNodeComponent } from './graph/LayerNode';

const nodeTypes = { layer: LayerNodeComponent };
const edgeTypes = { flow: FlowEdge };
const defaultEdgeOptions = { type: 'flow', markerEnd: { type: MarkerType.ArrowClosed, width: 16, height: 16 } };

function Suggestions() {
  const load = useStore((s) => s.loadTemplate);
  const setModal = useStore((s) => s.setModal);
  return (
    <div className="pop-in pointer-events-auto absolute bottom-6 left-1/2 w-[560px] -translate-x-1/2 rounded-2xl border border-line bg-panel/95 p-4 shadow-pop backdrop-blur">
      <div className="mb-3 flex items-center gap-2 text-[13px] font-medium">
        <Sparkles size={14} className="text-brand-400" /> Not sure where to start? Try a ready-made network
      </div>
      <div className="flex flex-wrap gap-2">
        {TEMPLATES.slice(0, 4).map((t) => (
          <button key={t.id} className="btn-outline h-7 text-[12px]" onClick={() => load(t)}>{t.name}</button>
        ))}
        <button className="btn-ghost h-7 text-[12px]" onClick={() => setModal('templates')}>All templates</button>
      </div>
    </div>
  );
}

function Canvas() {
  const nodes = useStore((s) => s.nodes);
  const edges = useStore((s) => s.edges);
  const selectedId = useStore((s) => s.selectedId);
  const beginner = useStore((s) => s.mode === 'beginner');
  const { onNodesChange, onEdgesChange, onConnect, addLayer, select, checkpoint, arrange } = useStore.getState();
  const net = useNetwork();
  const { screenToFlowPosition, fitView } = useReactFlow();

  const shown = useMemo(() => nodes.map((n) => (n.selected === (n.id === selectedId) ? n : { ...n, selected: n.id === selectedId })), [nodes, selectedId]);

  const onDrop = useCallback(
    (e: DragEvent) => {
      const type = e.dataTransfer.getData(DRAG_TYPE) as LayerType;
      if (!type) return;
      e.preventDefault();
      const p = screenToFlowPosition({ x: e.clientX, y: e.clientY });
      addLayer(type, { x: p.x - 105, y: p.y - 30 });
    },
    [screenToFlowPosition, addLayer],
  );

  const isValidConnection: IsValidConnection = useCallback((c: Connection | Edge) => c.source !== c.target, []);
  const errorCount = net.errors.length + net.issues.length;

  return (
    <NetworkContext.Provider value={net}>
      <div className="relative h-full w-full" onDragOver={(e) => { e.preventDefault(); e.dataTransfer.dropEffect = 'copy'; }} onDrop={onDrop}>
        <ReactFlow<LayerNode>
          nodes={shown}
          edges={edges}
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          defaultEdgeOptions={defaultEdgeOptions}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          isValidConnection={isValidConnection}
          onNodeClick={(_, n) => select(n.id)}
          onPaneClick={() => select(null)}
          onNodeDragStart={checkpoint}
          deleteKeyCode={null}
          fitView
          fitViewOptions={{ padding: 0.25, maxZoom: 1.1 }}
          minZoom={0.2}
          maxZoom={2}
          proOptions={{ hideAttribution: true }}
        >
          <Background variant={BackgroundVariant.Dots} gap={22} size={1.2} color="rgb(var(--line))" />
          <Controls showInteractive={false} position="bottom-left" />
          <MiniMap
            position="bottom-right"
            pannable
            zoomable
            maskColor="rgb(var(--bg) / 0.7)"
            nodeBorderRadius={6}
            nodeColor={(n) => {
              const i = net.chain.findIndex((c) => c.id === n.id);
              const t = (n as LayerNode).data.layerType;
              return layerColor(t, i === 0 ? 'input' : i === net.chain.length - 1 ? 'output' : 'hidden');
            }}
          />
        </ReactFlow>

        <div className="pointer-events-none absolute left-4 top-4 flex items-center gap-2">
          <div className="pointer-events-auto flex items-center gap-3 rounded-xl border border-line bg-panel/90 px-3 py-1.5 text-[12px] shadow-node backdrop-blur">
            <span><b className="font-semibold">{net.chain.length}</b> <span className="text-faint">layers</span></span>
            <span className="h-3 w-px bg-line" />
            <span><b className="font-mono font-semibold">{compact(net.totalParams)}</b> <span className="text-faint">parameters</span></span>
            {errorCount > 0 && (
              <>
                <span className="h-3 w-px bg-line" />
                <span className="flex items-center gap-1 text-bad-400"><XCircle size={12} /> {errorCount} issue{errorCount > 1 ? 's' : ''}</span>
              </>
            )}
          </div>
          <div className="pointer-events-auto flex rounded-xl border border-line bg-panel/90 shadow-node backdrop-blur">
            <button className="icon-btn" title="Auto-arrange layers" onClick={() => { arrange(); window.setTimeout(() => fitView({ padding: 0.25, duration: 400 }), 30); }}><LayoutGrid size={15} /></button>
            <button className="icon-btn" title="Fit to screen" onClick={() => fitView({ padding: 0.25, duration: 400 })}><Maximize2 size={15} /></button>
          </div>
        </div>

        {(net.issues.length > 0 || net.errors.length > 0) && (
          <div className="pointer-events-none absolute right-4 top-4 w-[300px] space-y-1.5">
            {[...net.issues, ...net.errors.map((e) => `${e.name}: ${e.message}`)].slice(0, 4).map((m) => (
              <div key={m} className="fade-in rounded-lg border border-bad-500/30 bg-panel/95 px-3 py-2 text-[12px] text-bad-400 shadow-node backdrop-blur">{m}</div>
            ))}
          </div>
        )}

        {!nodes.length && (
          <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center text-center">
            <MousePointer2 size={28} className="mb-3 text-faint" />
            <div className="text-[15px] font-semibold">Drag an {layerDef('Input').label} layer here to begin</div>
            <p className="mt-1 text-[13px] text-muted">Then add layers from the palette and connect them left to right.</p>
          </div>
        )}
        {beginner && net.chain.length <= 1 && <Suggestions />}
      </div>
    </NetworkContext.Provider>
  );
}

export function GraphView() {
  return (
    <ReactFlowProvider>
      <Canvas />
    </ReactFlowProvider>
  );
}
