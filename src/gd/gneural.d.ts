declare module 'gneural' {
  interface Counter {
    item: number;
    display(x: number, y: number): void;
  }
  export class MLP {
    constructor(inputNeurons: number, hiddenNeurons: number, outputNeurons: number);
    inputNeurons: number;
    outputNeurons: number;
    weightsInputHidden: number[][];
    weightsHiddenOutput: number[][];
    biasHidden: number[] | number[][];
    biasOutput: number[];
    outputOutputs: number[];
    feedForward(input: number[]): void;
    calculateError(target: number[]): number;
    train(data: { input: number[]; output: number[] }[], iterations: number, learningRate?: number): void;
    predict(...input: number[]): { input: Counter[]; hidden: Counter[] | Counter[][]; output: Counter[] };
  }
  export class MLPDense extends MLP {
    constructor(inputNeurons: number, hiddenNeurons: number[] | number, outputNeurons: number);
    weightsHidden: number[][][];
    hiddenLayersConfig: number[];
  }
}
declare module '@g-js-api/g.js';
