export type LayerType =
  | 'Input'
  | 'Dense'
  | 'Dropout'
  | 'Flatten'
  | 'Reshape'
  | 'Conv2D'
  | 'Conv1D'
  | 'MaxPooling'
  | 'AveragePooling'
  | 'BatchNorm'
  | 'LSTM'
  | 'GRU'
  | 'SimpleRNN'
  | 'Bidirectional'
  | 'MultiHeadAttention'
  | 'SelfAttention'
  | 'TransformerBlock'
  | 'ReLU'
  | 'Sigmoid'
  | 'Softmax'
  | 'Tanh'
  | 'GELU'
  | 'Custom';

export type Activation = 'linear' | 'relu' | 'sigmoid' | 'tanh' | 'softmax' | 'gelu' | 'leaky_relu';
export type Initializer = 'glorot_uniform' | 'he_normal' | 'lecun_normal' | 'gneural' | 'zeros';
export type ParamValue = number | string | boolean;
export type LayerParams = Record<string, ParamValue>;

export type Shape = number[];

export interface LayerSpec {
  id: string;
  type: LayerType;
  name: string;
  params: LayerParams;
}

export type OptimizerName = 'adam' | 'sgd' | 'rmsprop' | 'adagrad';
export type LossName = 'crossentropy' | 'bce' | 'mse' | 'mae' | 'huber';
export type MetricName = 'accuracy' | 'precision' | 'recall' | 'f1';
export type ScheduleName = 'constant' | 'step' | 'exponential' | 'cosine' | 'warmup_cosine';
export type EngineName = 'workbench' | 'gneural';

export interface TrainConfig {
  engine: EngineName;
  optimizer: OptimizerName;
  loss: LossName;
  metrics: MetricName[];
  epochs: number;
  batchSize: number;
  learningRate: number;
  schedule: ScheduleName;
  momentum: number;
  beta1: number;
  beta2: number;
  epsilon: number;
  clipNorm: number;
  seed: number;
}

/** A learnable tensor and its accumulated gradient. */
export interface Param {
  name: string;
  shape: Shape;
  value: Float32Array;
  grad: Float32Array;
  l1: number;
  l2: number;
  /** Updated by the layer itself (e.g. running statistics), never by the optimizer. */
  frozen?: boolean;
}

export interface LayerTrace {
  /** Attention weights per head, each [T, T] */
  attention?: Float32Array[];
  /** Feature maps shape for conv layers [H, W, C] */
  featureShape?: Shape;
}

export interface RuntimeLayer {
  spec: LayerSpec;
  inShape: Shape;
  outShape: Shape;
  params: Param[];
  forward(x: Float32Array, training: boolean): Float32Array;
  backward(dy: Float32Array): Float32Array;
  trace?(): LayerTrace;
}

export interface Sample {
  x: Float32Array;
  y: Float32Array;
  /** Class index for classification tasks, -1 otherwise */
  label: number;
  text?: string;
}

export type TaskKind = 'classification' | 'regression' | 'reconstruction';
