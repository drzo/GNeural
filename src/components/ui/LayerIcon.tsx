import {
  Binary, Boxes, Brain, Code2, Combine, Eye, Grid3x3, Layers, LogIn, Minimize2, Repeat, Repeat2, Rows3, Shrink,
  Sigma, Sparkles, SplitSquareHorizontal, Waves, Zap, Activity, TrendingUp, Spline, Scaling, CircleDot, type LucideIcon,
} from 'lucide-react';
import { layerDef } from '../../engine/catalog';
import type { LayerType } from '../../engine/types';

const ICONS: Record<LayerType, LucideIcon> = {
  Input: LogIn,
  Dense: Layers,
  Dropout: CircleDot,
  Flatten: Minimize2,
  Reshape: Scaling,
  Conv2D: Grid3x3,
  Conv1D: Rows3,
  MaxPooling: Shrink,
  AveragePooling: Combine,
  BatchNorm: Sigma,
  LSTM: Repeat,
  GRU: Repeat2,
  SimpleRNN: Waves,
  Bidirectional: SplitSquareHorizontal,
  MultiHeadAttention: Eye,
  SelfAttention: Sparkles,
  TransformerBlock: Brain,
  ReLU: Zap,
  Sigmoid: Spline,
  Softmax: Binary,
  Tanh: Activity,
  GELU: TrendingUp,
  Custom: Code2,
};

export function LayerIcon({ type, size = 14, className = '' }: { type: LayerType; size?: number; className?: string }) {
  const Icon = ICONS[type] ?? Boxes;
  return <Icon size={size} className={className} />;
}

export function LayerBadge({ type, color, size = 28 }: { type: LayerType; color?: string; size?: number }) {
  const c = color ?? layerDef(type).color;
  return (
    <span
      className="inline-flex shrink-0 items-center justify-center rounded-md"
      style={{ width: size, height: size, background: `${c}22`, color: c, boxShadow: `inset 0 0 0 1px ${c}40` }}
    >
      <LayerIcon type={type} size={Math.round(size * 0.52)} />
    </span>
  );
}
