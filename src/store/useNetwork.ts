import { useMemo } from 'react';
import { loadDataset, type Dataset } from '../data/datasets';
import { analyzeChain, type LayerAnalysis } from '../engine/shapes';
import type { LayerSpec } from '../engine/types';
import { extractChain } from './graph';
import { useStore, type NetworkConfig } from './useStore';
import type { LayerNode } from './graph';
import type { Edge } from '@xyflow/react';

export interface NetworkInfo {
  chain: LayerSpec[];
  analysis: LayerAnalysis[];
  byId: Map<string, LayerAnalysis>;
  issues: string[];
  orphans: string[];
  dataset: Dataset | null;
  datasetError: string | null;
  totalParams: number;
  errors: { id: string; name: string; message: string }[];
  valid: boolean;
}

export function computeNetwork(nodes: LayerNode[], edges: Edge[], config: NetworkConfig): NetworkInfo {
  const { chain, issues, orphans } = extractChain(nodes, edges);
  let dataset: Dataset | null = null;
  let datasetError: string | null = null;
  try {
    dataset = loadDataset(config.datasetId, config.csv);
  } catch (e) {
    datasetError = (e as Error).message;
  }
  const analysis = analyzeChain(chain, dataset?.targetShape);
  if (dataset && chain[0]?.type === 'Input' && analysis[0]?.outShape && analysis[0].outShape.join() !== dataset.inputShape.join()) {
    analysis[0].error = `Input shape [${analysis[0].outShape.join(', ')}] does not match dataset samples [${dataset.inputShape.join(', ')}]`;
  }
  const byId = new Map(analysis.map((a) => [a.id, a]));
  const names = new Map(chain.map((c) => [c.id, c.name]));
  const errors = analysis.filter((a) => a.error).map((a) => ({ id: a.id, name: names.get(a.id) ?? a.id, message: a.error! }));
  return {
    chain,
    analysis,
    byId,
    issues,
    orphans,
    dataset,
    datasetError,
    totalParams: analysis.reduce((s, a) => s + a.params, 0),
    errors,
    valid: chain.length > 1 && !errors.length && !!dataset,
  };
}

export function useNetwork(): NetworkInfo {
  const nodes = useStore((s) => s.nodes);
  const edges = useStore((s) => s.edges);
  const config = useStore((s) => s.config);
  return useMemo(() => computeNetwork(nodes, edges, config), [nodes, edges, config]);
}
