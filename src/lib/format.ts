import { LAYER_COLORS, layerDef } from '../engine/catalog';
import type { LayerType } from '../engine/types';

export function compact(n: number) {
  if (!Number.isFinite(n)) return '-';
  const a = Math.abs(n);
  if (a >= 1e6) return `${(n / 1e6).toFixed(a >= 1e7 ? 1 : 2)}M`;
  if (a >= 1e3) return `${(n / 1e3).toFixed(a >= 1e4 ? 1 : 2)}k`;
  return String(n);
}

export const pct = (v: number | null | undefined, digits = 1) => (v === null || v === undefined || !Number.isFinite(v) ? '-' : `${(v * 100).toFixed(digits)}%`);

export const num = (v: number | null | undefined, digits = 4) => (v === null || v === undefined || !Number.isFinite(v) ? '-' : v.toFixed(digits));

export const clock = (ts: number) => new Date(ts).toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });

/** Input layers are blue, the final trainable layer green, everything else uses its category color. */
export function layerColor(type: LayerType, role: 'input' | 'output' | 'hidden') {
  if (type === 'Input') return LAYER_COLORS.input;
  if (role === 'output' && type === 'Dense') return LAYER_COLORS.output;
  return layerDef(type).color;
}
