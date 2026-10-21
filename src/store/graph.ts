import type { Edge, Node } from '@xyflow/react';
import { layerDef } from '../engine/catalog';
import type { LayerParams, LayerSpec, LayerType } from '../engine/types';

export interface LayerNodeData extends Record<string, unknown> {
  layerType: LayerType;
  name: string;
  params: LayerParams;
}

export type LayerNode = Node<LayerNodeData, 'layer'>;

export interface ChainResult {
  chain: LayerSpec[];
  issues: string[];
  orphans: string[];
}

export const newId = () => crypto.randomUUID().slice(0, 8);

export function makeNode(type: LayerType, position: { x: number; y: number }, params: LayerParams, name?: string): LayerNode {
  return {
    id: newId(),
    type: 'layer',
    position,
    data: { layerType: type, name: name ?? layerDef(type).label, params },
  };
}

export const makeEdge = (source: string, target: string): Edge => ({ id: `e-${source}-${target}`, source, target, type: 'flow' });

/** Follows edges from the single Input node to produce the executable layer order. */
export function extractChain(nodes: LayerNode[], edges: Edge[]): ChainResult {
  const issues: string[] = [];
  const inputs = nodes.filter((n) => n.data.layerType === 'Input');
  if (!inputs.length) return { chain: [], issues: nodes.length ? ['Add an Input layer to start the network.'] : [], orphans: nodes.map((n) => n.id) };
  if (inputs.length > 1) issues.push('Only one Input layer is supported; using the first.');
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const out = new Map<string, string[]>();
  for (const e of edges) out.set(e.source, [...(out.get(e.source) ?? []), e.target]);
  const chain: LayerSpec[] = [];
  const seen = new Set<string>();
  let cur: LayerNode | undefined = inputs[0];
  while (cur && !seen.has(cur.id)) {
    seen.add(cur.id);
    chain.push({ id: cur.id, type: cur.data.layerType, name: cur.data.name, params: cur.data.params });
    const next: string[] = out.get(cur.id) ?? [];
    if (next.length > 1) issues.push(`"${cur.data.name}" has ${next.length} outgoing connections; branching is not executable, following the first.`);
    cur = next.length ? byId.get(next[0]) : undefined;
    if (cur && seen.has(cur.id)) issues.push('The graph contains a loop; it was cut where it repeats.');
  }
  const orphans = nodes.filter((n) => !seen.has(n.id)).map((n) => n.id);
  if (orphans.length) issues.push(`${orphans.length} layer${orphans.length > 1 ? 's are' : ' is'} not connected to the Input chain and will be ignored.`);
  return { chain, issues, orphans };
}

/** Lays layers out left-to-right in rows so long networks stay readable. */
export function autoLayout(count: number, perRow = 5) {
  return Array.from({ length: count }, (_, i) => ({ x: (i % perRow) * 270, y: Math.floor(i / perRow) * 190 }));
}
