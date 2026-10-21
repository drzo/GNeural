import { defaultParams } from '../engine/catalog';
import type { LayerParams, LayerType, TrainConfig } from '../engine/types';

export interface TemplateDef {
  id: string;
  name: string;
  tagline: string;
  description: string;
  datasetId: string;
  difficulty: 'Beginner' | 'Intermediate' | 'Advanced';
  layers: [LayerType, LayerParams?, string?][];
  train: Partial<TrainConfig>;
  gd?: boolean;
}

export const TEMPLATES: TemplateDef[] = [
  {
    id: 'mlp', name: 'Simple MLP', tagline: 'MNIST-style digit classifier', difficulty: 'Beginner', datasetId: 'digits',
    description: 'Flattens 8x8 digit images and classifies them with two dense layers. The best place to start.',
    layers: [
      ['Input', { shape: '8,8,1' }, 'Pixels'],
      ['Flatten'],
      ['Dense', { units: 64 }, 'Hidden 1'],
      ['Dropout', { rate: 0.2 }],
      ['Dense', { units: 32 }, 'Hidden 2'],
      ['Dense', { units: 10, activation: 'softmax' }, 'Digit'],
    ],
    train: { optimizer: 'adam', loss: 'crossentropy', learningRate: 0.005, epochs: 15, batchSize: 32 },
  },
  {
    id: 'cnn', name: 'CNN', tagline: 'Convolutional image classifier', difficulty: 'Intermediate', datasetId: 'digits',
    description: 'Two convolution + pooling stages extract strokes and shapes before a dense classifier head.',
    layers: [
      ['Input', { shape: '8,8,1' }, 'Image'],
      ['Conv2D', { filters: 8, kernel: 3 }, 'Conv 1'],
      ['MaxPooling', { pool: 2 }],
      ['Conv2D', { filters: 16, kernel: 3 }, 'Conv 2'],
      ['MaxPooling', { pool: 2 }],
      ['Flatten'],
      ['Dense', { units: 32 }, 'Features'],
      ['Dense', { units: 10, activation: 'softmax' }, 'Digit'],
    ],
    train: { optimizer: 'adam', loss: 'crossentropy', learningRate: 0.004, epochs: 12, batchSize: 16 },
  },
  {
    id: 'lstm', name: 'LSTM', tagline: 'Sequence classifier', difficulty: 'Intermediate', datasetId: 'waves',
    description: 'An LSTM reads a 16-step signal and recognizes whether it is a sine, square, sawtooth or triangle wave.',
    layers: [
      ['Input', { shape: '16,1' }, 'Signal'],
      ['LSTM', { units: 16 }, 'Memory'],
      ['Dense', { units: 4, activation: 'softmax' }, 'Waveform'],
    ],
    train: { optimizer: 'adam', loss: 'crossentropy', learningRate: 0.01, epochs: 15, batchSize: 16, clipNorm: 5 },
  },
  {
    id: 'transformer', name: 'Transformer', tagline: 'Text sentiment classifier', difficulty: 'Advanced', datasetId: 'sentiment',
    description: 'A Transformer encoder block attends across tokens of a short phrase to decide if it is positive or negative.',
    layers: [
      ['Input', { shape: '8,16' }, 'Tokens'],
      ['TransformerBlock', { heads: 2, ffDim: 32 }, 'Encoder'],
      ['Flatten'],
      ['Dense', { units: 2, activation: 'softmax' }, 'Sentiment'],
    ],
    train: { optimizer: 'adam', loss: 'crossentropy', learningRate: 0.003, epochs: 10, batchSize: 16, schedule: 'warmup_cosine' },
  },
  {
    id: 'autoencoder', name: 'Autoencoder', tagline: 'Compress and rebuild images', difficulty: 'Intermediate', datasetId: 'digits-ae',
    description: 'Squeezes each digit through an 8-number bottleneck and learns to reconstruct the original image.',
    layers: [
      ['Input', { shape: '8,8,1' }, 'Image'],
      ['Flatten'],
      ['Dense', { units: 32 }, 'Encoder'],
      ['Dense', { units: 8 }, 'Code'],
      ['Dense', { units: 32 }, 'Decoder'],
      ['Dense', { units: 64, activation: 'sigmoid' }, 'Pixels'],
      ['Reshape', { target: '8,8,1' }, 'Image out'],
    ],
    train: { optimizer: 'adam', loss: 'mse', learningRate: 0.005, epochs: 15, batchSize: 32 },
  },
  {
    id: 'gd-mlp', name: 'Geometry Dash MLP', tagline: 'Runs inside Geometry Dash', difficulty: 'Beginner', datasetId: 'next-number', gd: true,
    description: 'The Gneural README network: predicts the next number in a sequence and exports to a Geometry Dash level.',
    layers: [
      ['Input', { shape: '2' }, 'Inputs'],
      ['Dense', { units: 4, initializer: 'gneural' }, 'Hidden 1'],
      ['Dense', { units: 3, initializer: 'gneural' }, 'Hidden 2'],
      ['Dense', { units: 1, activation: 'relu', initializer: 'gneural' }, 'Output'],
    ],
    train: { engine: 'gneural', optimizer: 'sgd', loss: 'mse', learningRate: 0.01, epochs: 300, batchSize: 1, schedule: 'constant' },
  },
];

export function templateParams(type: LayerType, overrides?: LayerParams) {
  return { ...defaultParams(type), ...overrides };
}
