import type { LayerType, ParamValue } from './types';

export type Category = 'Core' | 'Convolutional' | 'Recurrent' | 'Attention' | 'Activation' | 'Custom';

export interface ParamDef {
  key: string;
  label: string;
  kind: 'int' | 'float' | 'select' | 'bool' | 'shape' | 'code';
  default: ParamValue;
  options?: string[];
  min?: number;
  max?: number;
  step?: number;
  advanced?: boolean;
  help: string;
}

export interface LayerDef {
  type: LayerType;
  label: string;
  category: Category;
  color: string;
  short: string;
  description: string;
  help: string;
  params: ParamDef[];
}

export const ACTIVATIONS = ['linear', 'relu', 'sigmoid', 'tanh', 'softmax', 'gelu', 'leaky_relu'];
const INITIALIZERS = ['glorot_uniform', 'he_normal', 'lecun_normal', 'gneural', 'zeros'];

export const LAYER_COLORS = {
  input: '#4f8ef7',
  hidden: '#9b59f7',
  output: '#2ecc71',
  conv: '#f5873b',
  recurrent: '#1fc7b6',
  attention: '#f06292',
  activation: '#f1c40f',
  utility: '#8b93a7',
  custom: '#e74c3c',
};

const activation = (def = 'relu'): ParamDef => ({
  key: 'activation',
  label: 'Activation',
  kind: 'select',
  default: def,
  options: ACTIVATIONS,
  help: 'The function applied to each output value. ReLU is a strong default for hidden layers; Softmax turns outputs into class probabilities.',
});

const regularization: ParamDef[] = [
  { key: 'l1', label: 'L1 penalty', kind: 'float', default: 0, min: 0, max: 0.1, step: 0.0001, advanced: true, help: 'Pushes weights toward exactly zero, producing sparse connections.' },
  { key: 'l2', label: 'L2 penalty', kind: 'float', default: 0, min: 0, max: 0.1, step: 0.0001, advanced: true, help: 'Keeps weights small to reduce overfitting (also called weight decay).' },
  { key: 'initializer', label: 'Initialization', kind: 'select', default: 'glorot_uniform', options: INITIALIZERS, advanced: true, help: 'How starting weights are chosen. Glorot suits tanh/sigmoid, He suits ReLU. "gneural" matches the Geometry Dash library.' },
];

const units = (def: number, label = 'Units'): ParamDef => ({
  key: 'units',
  label,
  kind: 'int',
  default: def,
  min: 1,
  max: 1024,
  help: 'How many neurons this layer has. More units can learn more complex patterns but train slower.',
});

const returnSeq: ParamDef = {
  key: 'returnSequences',
  label: 'Return sequences',
  kind: 'bool',
  default: false,
  help: 'When on, outputs a value for every time step (needed when stacking recurrent layers). When off, only the final summary is passed on.',
};

const heads: ParamDef = { key: 'heads', label: 'Heads', kind: 'int', default: 2, min: 1, max: 16, help: 'Number of parallel attention patterns. The feature size must divide evenly by this.' };

export const LAYERS: LayerDef[] = [
  {
    type: 'Input', label: 'Input', category: 'Core', color: LAYER_COLORS.input, short: 'IN',
    description: 'Where data enters the network. Its shape must match your dataset.',
    help: 'Every network starts with exactly one Input layer. Its shape describes one sample: [8, 8, 1] is an 8x8 grayscale image, [16, 1] is a sequence of 16 numbers, [2] is two plain numbers.',
    params: [{ key: 'shape', label: 'Shape', kind: 'shape', default: '2', help: 'Comma-separated size of one sample, e.g. "8,8,1" for a small image.' }],
  },
  {
    type: 'Dense', label: 'Dense', category: 'Core', color: LAYER_COLORS.hidden, short: 'FC',
    description: 'A fully connected layer: every input connects to every neuron.',
    help: 'Dense layers are the workhorse of neural networks. Each neuron computes a weighted sum of all inputs plus a bias, then applies the activation. Applied to the last dimension of the input.',
    params: [units(32), activation(), { key: 'useBias', label: 'Use bias', kind: 'bool', default: true, advanced: true, help: 'Adds a learnable offset to each neuron.' }, ...regularization],
  },
  {
    type: 'Dropout', label: 'Dropout', category: 'Core', color: LAYER_COLORS.utility, short: 'DO',
    description: 'Randomly switches off neurons during training to prevent memorizing.',
    help: 'During training, each value is zeroed with the given probability and the rest are scaled up. At prediction time Dropout does nothing. It is one of the simplest ways to reduce overfitting.',
    params: [{ key: 'rate', label: 'Rate', kind: 'float', default: 0.2, min: 0, max: 0.9, step: 0.05, help: 'Fraction of values dropped during training.' }],
  },
  {
    type: 'Flatten', label: 'Flatten', category: 'Core', color: LAYER_COLORS.utility, short: 'FL',
    description: 'Unrolls a grid or sequence into one long list of numbers.',
    help: 'Converts multi-dimensional data such as [4, 4, 16] into a flat vector of 256 values so Dense layers can treat each value as a separate input.',
    params: [],
  },
  {
    type: 'Reshape', label: 'Reshape', category: 'Core', color: LAYER_COLORS.utility, short: 'RS',
    description: 'Rearranges values into a new shape without changing them.',
    help: 'The total number of values must stay the same. Useful for turning a flat output back into an image, as in an autoencoder.',
    params: [{ key: 'target', label: 'Target shape', kind: 'shape', default: '8,8,1', help: 'Comma-separated new shape.' }],
  },
  {
    type: 'Conv2D', label: 'Conv2D', category: 'Convolutional', color: LAYER_COLORS.conv, short: 'C2',
    description: 'Slides small learnable filters over an image to detect local patterns.',
    help: 'Each filter scans the image and lights up where it finds its pattern (edges, corners, strokes). Stacking convolutions builds up from simple to complex features. Expects [height, width, channels].',
    params: [
      { key: 'filters', label: 'Filters', kind: 'int', default: 8, min: 1, max: 256, help: 'Number of pattern detectors; each produces one feature map.' },
      { key: 'kernel', label: 'Kernel size', kind: 'int', default: 3, min: 1, max: 9, help: 'Width and height of each filter window.' },
      { key: 'stride', label: 'Stride', kind: 'int', default: 1, min: 1, max: 4, advanced: true, help: 'Step size of the sliding window. 2 halves the output size.' },
      { key: 'padding', label: 'Padding', kind: 'select', default: 'same', options: ['same', 'valid'], help: '"same" keeps the size by padding the borders, "valid" shrinks it.' },
      activation(), ...regularization,
    ],
  },
  {
    type: 'Conv1D', label: 'Conv1D', category: 'Convolutional', color: LAYER_COLORS.conv, short: 'C1',
    description: 'Slides filters along a sequence to find local patterns over time.',
    help: 'Like Conv2D but along one axis. Great for signals, audio and time series. Expects [steps, features].',
    params: [
      { key: 'filters', label: 'Filters', kind: 'int', default: 8, min: 1, max: 256, help: 'Number of pattern detectors.' },
      { key: 'kernel', label: 'Kernel size', kind: 'int', default: 3, min: 1, max: 15, help: 'How many time steps each filter looks at.' },
      { key: 'stride', label: 'Stride', kind: 'int', default: 1, min: 1, max: 4, advanced: true, help: 'Step size of the window.' },
      { key: 'padding', label: 'Padding', kind: 'select', default: 'same', options: ['same', 'valid'], help: '"same" keeps the length, "valid" shrinks it.' },
      activation(), ...regularization,
    ],
  },
  {
    type: 'MaxPooling', label: 'MaxPooling', category: 'Convolutional', color: LAYER_COLORS.conv, short: 'MP',
    description: 'Shrinks feature maps by keeping the strongest value in each window.',
    help: 'Downsamples the data, making the network faster and more tolerant to small shifts. Works on images [H, W, C] and sequences [T, C].',
    params: [{ key: 'pool', label: 'Pool size', kind: 'int', default: 2, min: 1, max: 4, help: 'Window size; the output is this many times smaller.' }],
  },
  {
    type: 'AveragePooling', label: 'AvgPooling', category: 'Convolutional', color: LAYER_COLORS.conv, short: 'AP',
    description: 'Shrinks feature maps by averaging each window.',
    help: 'A smoother alternative to max pooling that keeps the average response in each region.',
    params: [{ key: 'pool', label: 'Pool size', kind: 'int', default: 2, min: 1, max: 4, help: 'Window size.' }],
  },
  {
    type: 'BatchNorm', label: 'BatchNorm', category: 'Convolutional', color: LAYER_COLORS.utility, short: 'BN',
    description: 'Re-centers and re-scales values to keep training stable.',
    help: 'Normalizes each channel using running statistics gathered during training, then applies a learnable scale and shift. Helps deeper networks train faster.',
    params: [{ key: 'momentum', label: 'Momentum', kind: 'float', default: 0.9, min: 0.5, max: 0.999, step: 0.01, advanced: true, help: 'How slowly the running statistics update.' }],
  },
  {
    type: 'LSTM', label: 'LSTM', category: 'Recurrent', color: LAYER_COLORS.recurrent, short: 'LS',
    description: 'Reads sequences step by step with a long-term memory cell.',
    help: 'Long Short-Term Memory layers use gates to decide what to remember and forget, letting them capture patterns across many time steps. Expects [steps, features].',
    params: [units(16), returnSeq, ...regularization],
  },
  {
    type: 'GRU', label: 'GRU', category: 'Recurrent', color: LAYER_COLORS.recurrent, short: 'GR',
    description: 'A lighter recurrent layer with gated memory.',
    help: 'Gated Recurrent Units are a simpler cousin of LSTM with fewer parameters and often similar accuracy.',
    params: [units(16), returnSeq, ...regularization],
  },
  {
    type: 'SimpleRNN', label: 'SimpleRNN', category: 'Recurrent', color: LAYER_COLORS.recurrent, short: 'RN',
    description: 'The most basic recurrent layer: carries a hidden state through time.',
    help: 'Each step mixes the new input with the previous state. Simple and fast, but struggles with long-range patterns (vanishing gradients).',
    params: [units(16), returnSeq, ...regularization],
  },
  {
    type: 'Bidirectional', label: 'Bidirectional', category: 'Recurrent', color: LAYER_COLORS.recurrent, short: 'BI',
    description: 'Runs a recurrent layer forwards and backwards and combines both.',
    help: 'Lets the network use both past and future context at every step. The output size is twice the units.',
    params: [{ key: 'cell', label: 'Cell', kind: 'select', default: 'LSTM', options: ['LSTM', 'GRU', 'SimpleRNN'], help: 'Which recurrent layer to wrap.' }, units(8), returnSeq, ...regularization],
  },
  {
    type: 'MultiHeadAttention', label: 'MultiHeadAttn', category: 'Attention', color: LAYER_COLORS.attention, short: 'MH',
    description: 'Lets every position look at every other position, in several ways at once.',
    help: 'Each head learns which tokens should attend to which. Outputs keep the input shape [steps, features]. The feature size must be divisible by the number of heads.',
    params: [heads, ...regularization],
  },
  {
    type: 'SelfAttention', label: 'SelfAttention', category: 'Attention', color: LAYER_COLORS.attention, short: 'SA',
    description: 'Single-head attention: each step gathers information from relevant steps.',
    help: 'Computes queries, keys and values from the same sequence and mixes values using attention weights.',
    params: [...regularization],
  },
  {
    type: 'TransformerBlock', label: 'Transformer', category: 'Attention', color: LAYER_COLORS.attention, short: 'TF',
    description: 'Attention + feed-forward network with residuals and normalization.',
    help: 'The building block of modern language models: multi-head self-attention, then a small two-layer network, each wrapped with a skip connection and layer normalization.',
    params: [heads, { key: 'ffDim', label: 'Feed-forward size', kind: 'int', default: 32, min: 4, max: 1024, help: 'Width of the inner feed-forward layer.' }, ...regularization],
  },
  ...(['ReLU', 'Sigmoid', 'Softmax', 'Tanh', 'GELU'] as const).map(
    (t): LayerDef => ({
      type: t, label: t, category: 'Activation', color: LAYER_COLORS.activation, short: t.slice(0, 2).toUpperCase(),
      description: {
        ReLU: 'Keeps positive values, zeroes out negatives.',
        Sigmoid: 'Squashes values into the 0 to 1 range.',
        Softmax: 'Turns scores into probabilities that sum to 1.',
        Tanh: 'Squashes values into the -1 to 1 range.',
        GELU: 'A smooth ReLU used by Transformers.',
      }[t],
      help: 'A standalone activation layer applies its function to every value without adding any parameters.',
      params: [],
    }),
  ),
  {
    type: 'Custom', label: 'Custom', category: 'Custom', color: LAYER_COLORS.custom, short: 'JS',
    description: 'Write your own element-wise function in JavaScript.',
    help: 'The expression receives each value as x and returns the new value. Gradients are estimated numerically, so any smooth function trains. Custom code only runs after you enable it in the inspector.',
    params: [{ key: 'code', label: 'Expression', kind: 'code', default: 'return x * (1 / (1 + Math.exp(-x)));', help: 'JavaScript function body. Receives x, returns a number.' }],
  },
];

export const CATEGORIES: Category[] = ['Core', 'Convolutional', 'Recurrent', 'Attention', 'Activation', 'Custom'];

const byType = new Map(LAYERS.map((l) => [l.type, l]));
export const layerDef = (t: LayerType): LayerDef => byType.get(t)!;

export function defaultParams(t: LayerType) {
  return Object.fromEntries(layerDef(t).params.map((p) => [p.key, p.default]));
}

export function parseShape(v: ParamValue | undefined): number[] | null {
  if (typeof v !== 'string' && typeof v !== 'number') return null;
  const parts = String(v).split(/[,x\s]+/).filter(Boolean).map(Number);
  if (!parts.length || parts.some((n) => !Number.isInteger(n) || n <= 0)) return null;
  return parts;
}

export const TRAINABLE: LayerType[] = ['Dense', 'Conv2D', 'Conv1D', 'LSTM', 'GRU', 'SimpleRNN', 'Bidirectional', 'MultiHeadAttention', 'SelfAttention', 'TransformerBlock', 'BatchNorm'];
