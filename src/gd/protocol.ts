export interface GneuralArch {
  inputs: number;
  hidden: number[];
  outputs: number;
  /** 'mlp' = one hidden layer with linear output; 'dense' = MLPDense with ReLU output */
  kind: 'mlp' | 'dense';
}

/** One entry per hidden layer, then the output layer. W is [in][out]. */
export interface GneuralWeights {
  layers: { W: number[][]; b: number[] }[];
}

export interface Sample {
  input: number[];
  output: number[];
}

export type WorkerRequest =
  | { type: 'export'; arch: GneuralArch; weights: GneuralWeights; input: number[]; spacing: number }
  | {
      type: 'train';
      arch: GneuralArch;
      weights: GneuralWeights | null;
      train: Sample[];
      val: Sample[];
      epochs: number;
      learningRate: number;
      reportEvery: number;
    };

export type WorkerResponse =
  | { type: 'exported'; levelString: string; objects: number; groups: number; outputs: number[] }
  | { type: 'progress'; epoch: number; trainLoss: number; valLoss: number; weights: GneuralWeights }
  | { type: 'trained'; weights: GneuralWeights }
  | { type: 'error'; message: string };
