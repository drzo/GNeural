import { TEMPLATES } from './templates';
import { useStore, type AppState } from '../store/useStore';

export interface TutorialStep {
  /** data-tour attribute of the element to spotlight; omit for a centered card */
  target?: string;
  title: string;
  body: string;
  /** Leaves the app usable and parks the card in a corner instead of dimming the screen */
  corner?: boolean;
  /** When set, the learner must perform an action; the step advances automatically once done */
  task?: { hint: string; done: (s: AppState) => boolean };
}

export interface Tutorial {
  id: string;
  title: string;
  summary: string;
  level: 'Beginner' | 'Intermediate' | 'Advanced';
  setup?: () => void;
  steps: TutorialStep[];
}

const loadTemplate = (id: string) => () => {
  const t = TEMPLATES.find((x) => x.id === id);
  const s = useStore.getState();
  s.run?.stop();
  if (t) s.loadTemplate(t);
  s.setView('graph');
  s.setRightTab('inspect');
  s.setModal(null);
};

const denseCount = (s: AppState) => s.nodes.filter((n) => n.data.layerType === 'Dense').length;

export const TUTORIALS: Tutorial[] = [
  {
    id: 'tour',
    title: 'Tour of the workbench',
    summary: 'A quick look at every part of the screen and what it is for.',
    level: 'Beginner',
    setup: () => useStore.getState().setView('graph'),
    steps: [
      { title: 'Welcome to Gneural', body: 'A visual workbench for designing, training and understanding neural networks right in your browser. This short tour shows you around.' },
      { target: 'palette', title: 'Layer palette', body: 'Every building block lives here. Drag a layer onto the canvas, or double-click it to append it to the end of your network. Hover any layer to learn what it does.' },
      { target: 'canvas', title: 'The canvas', body: 'Layers appear as cards. Drag from the dot on the right of one card to the left of another to connect them. Data flows from the Input layer along the arrows.' },
      { target: 'inspector', title: 'Properties and Run', body: 'Click a layer to tune it here. The Run tab holds the optimizer, loss, epochs and the Train button.' },
      { target: 'views', title: 'Six views of one network', body: 'Switch between building, architecture shapes, live training charts, trying predictions, layer-by-layer debugging and dataset setup.' },
      { target: 'console', title: 'Console', body: 'Training progress, warnings and errors are logged here with timestamps. Press Space at any time to start training.' },
      { target: 'tutorials-button', title: 'More tutorials', body: 'Open this menu any time to replay this tour or follow hands-on lessons that walk you through training, debugging and more.' },
    ],
  },
  {
    id: 'first-training',
    title: 'Train your first network',
    summary: 'Load a digit classifier, train it and test it on your own drawing.',
    level: 'Beginner',
    setup: loadTemplate('mlp'),
    steps: [
      { target: 'canvas', title: 'A ready-made network', body: 'This network reads a small 7 by 5 picture of a digit and guesses which digit it is. Pixels enter on the left and the Digit layer on the right gives one score per digit.' },
      { target: 'run-tab', title: 'Open the Run settings', body: 'Training settings live in the Run tab.', task: { hint: 'Click the Run tab', done: (s) => s.rightTab === 'run' } },
      { target: 'inspector', title: 'Training settings', body: 'Epochs is how many times the network studies the whole dataset. The learning rate controls how big each correction is. The defaults work well here.' },
      { target: 'train-button', title: 'Start training', body: 'Press Train and the network starts learning from thousands of example digits.', task: { hint: 'Click Train', done: (s) => s.training.status !== 'idle' || s.training.history.length > 0 } },
      { target: 'canvas', corner: true, title: 'Watch it learn', body: 'Loss going down means fewer mistakes. Accuracy going up means more correct guesses. Validation numbers show how it does on digits it has never studied.', task: { hint: 'Wait for 3 epochs to finish', done: (s) => s.training.history.length >= 3 } },
      { target: 'view-inference', title: 'Try it yourself', body: 'Let\'s see if it can read your handwriting.', task: { hint: 'Open the Inference view', done: (s) => s.view === 'inference' } },
      { target: 'canvas', corner: true, title: 'Draw a digit', body: 'Paint on the grid with the left mouse button, erase with the right. The bars show how confident the network is in each digit, and the cards below show what each layer saw.' },
    ],
  },
  {
    id: 'build-from-scratch',
    title: 'Build a network from scratch',
    summary: 'Start from an empty canvas, add layers and check their shapes.',
    level: 'Beginner',
    setup: () => {
      const s = useStore.getState();
      s.newNetwork();
      s.setView('graph');
      s.setRightTab('inspect');
    },
    steps: [
      { target: 'canvas', title: 'An empty network', body: 'You start with only an Input layer. It is set up for the XOR dataset: two numbers in, one of two answers out.' },
      { target: 'palette', corner: true, title: 'Add a hidden layer', body: 'Dense layers connect every input to every output. Double-click Dense in the palette, or drag it onto the canvas next to Input.', task: { hint: 'Add one Dense layer', done: (s) => denseCount(s) >= 1 } },
      { target: 'palette', corner: true, title: 'Add an output layer', body: 'Add a second Dense layer. The last Dense layer becomes the output and gives one score per answer.', task: { hint: 'Add a second Dense layer', done: (s) => denseCount(s) >= 2 } },
      { target: 'canvas', corner: true, title: 'Connect the layers', body: 'Layers you double-clicked are connected for you. If you dragged them, draw a line from the right dot of one card to the left dot of the next so data flows Input to Dense to Dense.', task: { hint: 'Connect all three layers', done: (s) => s.edges.length >= 2 } },
      { target: 'inspector', title: 'Tune a layer', body: 'Click any layer card to see its settings here. Units is how many neurons the layer has, and more units can learn more complex patterns. The ? icons explain every option in plain words.' },
      { target: 'view-architecture', title: 'Check the shapes', body: 'The Architecture view shows the shape of the data between every layer and warns about mistakes.', task: { hint: 'Open the Architecture view', done: (s) => s.view === 'architecture' } },
      { target: 'canvas', corner: true, title: 'Ready to train', body: 'If there are no red warnings, press Space or the Run button at the top to train your network.' },
    ],
  },
  {
    id: 'debugging',
    title: 'Debug a network layer by layer',
    summary: 'Step through a forward pass, set breakpoints and read the warning signs.',
    level: 'Intermediate',
    setup: loadTemplate('mlp'),
    steps: [
      { target: 'view-debug', title: 'Open the debugger', body: 'The Debug view runs one sample through the network one layer at a time so you can see exactly what happens inside.', task: { hint: 'Open the Debug view', done: (s) => s.view === 'debug' } },
      { target: 'debug-controls', title: 'Step through', body: 'Use the arrows to move one layer forward or back. The fast-forward button runs until the next breakpoint. Shuffle picks a different sample.' },
      { target: 'debug-layers', title: 'Breakpoints', body: 'Click the dot next to a layer to set a breakpoint. Layers past the current one are faded because they have not run yet. A warning triangle marks layers with gradient problems.' },
      { target: 'canvas', corner: true, title: 'Read the charts', body: 'Activations show what this layer outputs. Weights show what it has learned. Tiny gradients mean the layer has stopped learning, and huge ones mean training is unstable. A skull warning means some neurons never switch on.' },
    ],
  },
  {
    id: 'datasets',
    title: 'Work with your data',
    summary: 'Choose a dataset, set the train and test split and add augmentation.',
    level: 'Intermediate',
    setup: () => useStore.getState().setRightTab('inspect'),
    steps: [
      { target: 'view-dataset', title: 'Open the Dataset view', body: 'Everything about the data the network learns from lives here.', task: { hint: 'Open the Dataset view', done: (s) => s.view === 'dataset' } },
      { target: 'dataset-list', title: 'Pick a dataset', body: 'Choose a built-in dataset or upload a CSV file of your own where the last column is the answer. The preview on the right updates instantly.' },
      { target: 'dataset-splits', title: 'Split your data', body: 'The network studies the training set, is checked against the validation set after every epoch, and the test set is kept aside for the Inference view.' },
      { target: 'dataset-augmentation', title: 'Augmentation', body: 'These options add small random changes to training samples, like noise or a slight shift, so the network learns the idea instead of memorizing the examples.' },
    ],
  },
  {
    id: 'geometry-dash',
    title: 'Run a network inside Geometry Dash',
    summary: 'Train with the Gneural library and export a playable level.',
    level: 'Advanced',
    setup: loadTemplate('gd-mlp'),
    steps: [
      { target: 'canvas', title: 'A Geometry Dash network', body: 'This small network learns to predict the next number in a sequence. It only uses Dense layers with ReLU, which is exactly what Gneural can build out of Geometry Dash triggers.' },
      { target: 'run-tab', title: 'Open the Run settings', body: 'The engine choice is in the Run tab.', task: { hint: 'Click the Run tab', done: (s) => s.rightTab === 'run' } },
      { target: 'engine', title: 'Choose the Gneural engine', body: 'The Gneural engine trains with the same library that generates the level, so what you train is what runs in the game.', task: { hint: 'Select Gneural', done: (s) => s.config.train.engine === 'gneural' } },
      { target: 'train-button', title: 'Train it', body: 'Training runs in the background so the app stays responsive.', task: { hint: 'Click Train and let it finish', done: (s) => s.training.status === 'idle' && s.training.history.length > 0 } },
      { target: 'export-button', title: 'Export the level', body: 'Open the Export menu and switch to the Geometry Dash tab.', task: { hint: 'Open Export', done: (s) => s.modal === 'export' } },
      { corner: true, title: 'Generate and play', body: 'Enter some input values, generate the level, then copy or download it. Paste it into Geometry Dash with a level-sharing tool to watch the network compute in-game.' },
    ],
  },
];

const DONE_KEY = 'gneural.tutorials.done';

export function completedTutorials(): string[] {
  try {
    const v = JSON.parse(localStorage.getItem(DONE_KEY) ?? '[]');
    return Array.isArray(v) ? v.filter((x): x is string => typeof x === 'string') : [];
  } catch {
    return [];
  }
}

export function markTutorialDone(id: string) {
  const done = new Set(completedTutorials());
  done.add(id);
  localStorage.setItem(DONE_KEY, JSON.stringify([...done]));
}

export function startTutorial(id: string) {
  const t = TUTORIALS.find((x) => x.id === id);
  if (!t) return;
  const s = useStore.getState();
  s.setModal(null);
  t.setup?.();
  s.setTutorial({ id, step: 0 });
}
