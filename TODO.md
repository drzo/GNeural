Build a complete, production-ready frontend for **Gneural** — a visual neural network workbench — using React as the frontend framework and integrating with the `@g-js-api/g.js` library.

---

## Application Overview

Create a full end-to-end neural network development environment called **Gneural** that serves users of all skill levels (beginners to experts). The application should function as a visual, interactive neural network IDE with multiple switchable views, real-time computation, and deep debugging capabilities.

---

## Core Layout & Shell

Build a multi-panel IDE-style layout:
- **Top navigation bar**: App logo/name "Gneural", global actions (New Network, Save, Load, Export), view switcher tabs, and a run/stop control
- **Left sidebar**: Layer palette (drag-and-drop components), network tree/layer list, and a properties inspector
- **Main canvas area**: The primary workspace that changes based on the active view
- **Right panel**: Configuration panel, hyperparameter controls, and contextual help
- **Bottom panel**: Console/log output, training metrics, and debug output

---

## Multiple Views (Switchable)

Implement a view switcher in the top nav that toggles between these modes:

### 1. Graph View (Default)
- Interactive node-graph canvas for visually building neural networks
- Drag layers from the palette onto the canvas
- Connect layers by drawing edges between nodes
- Each node displays: layer type, shape, activation function, parameter count
- Zoom, pan, and minimap navigation
- Color-code layers by type (input=blue, hidden=purple, output=green, conv=orange, etc.)
- Click a node to inspect/edit its properties in the right panel

### 2. Architecture View
- Clean, top-down flowchart showing the full network architecture
- Display tensor shapes flowing between layers
- Highlight bottlenecks, shape mismatches, or incompatible connections with visual warnings
- Show parameter counts per layer and totals

### 3. Training View
- Real-time loss and accuracy charts (line graphs) that update during training
- Epoch/batch progress bars
- Side-by-side train vs. validation metrics
- Learning rate schedule visualization
- Ability to pause, resume, or stop training mid-run

### 4. Inference View
- Input panel: upload an image, paste text, or enter structured data depending on network type
- Run inference and display output predictions with confidence scores
- Visualize intermediate activations layer-by-layer
- Show attention maps or feature maps where applicable

### 5. Debug View
- Step through the network layer-by-layer
- Display activation distributions (histograms) per layer
- Highlight vanishing/exploding gradient warnings
- Show weight distributions and dead neurons
- Breakpoint system: pause forward pass at a selected layer

### 6. Dataset View
- Upload or connect a dataset
- Preview samples in a grid
- Show class distribution charts
- Configure train/val/test splits with sliders
- Data augmentation toggle options

---

## Layer Palette (Left Sidebar)

Organize draggable layer components into categories:
- **Core**: Dense, Dropout, Flatten, Reshape
- **Convolutional**: Conv2D, Conv1D, MaxPooling, AveragePooling, BatchNorm
- **Recurrent**: LSTM, GRU, SimpleRNN, Bidirectional
- **Attention**: MultiHeadAttention, SelfAttention, TransformerBlock
- **Activation**: ReLU, Sigmoid, Softmax, Tanh, GELU
- **Custom**: Code block for writing custom layer logic

Each palette item should show an icon, name, and a tooltip with a beginner-friendly description on hover.

---

## Properties Inspector

When a layer node is selected, show an editable form in the right panel:
- Layer name (editable)
- All hyperparameters for that layer type (units, kernel size, filters, dropout rate, etc.)
- Activation function selector
- Regularization options (L1, L2)
- Initialization strategy
- A "?" help button that shows a plain-language explanation for beginners

---

## Computational Environment

Integrate with `@g-js-api/g.js` to power:
- **Model compilation**: Define optimizer, loss function, and metrics
- **Training execution**: Run training loops with real-time metric streaming back to the Training View
- **Inference execution**: Pass inputs through the model and retrieve outputs
- **Layer-by-layer debugging**: Extract intermediate activations during a forward pass
- **Model serialization**: Save and load model configurations and weights

Expose a **Run Panel** with:
- Optimizer selector (Adam, SGD, RMSProp, AdaGrad)
- Loss function selector (CrossEntropy, MSE, MAE, etc.)
- Metrics checkboxes (Accuracy, Precision, Recall, F1)
- Epochs and batch size inputs
- A prominent **Train** button and **Run Inference** button

---

## Console & Logs (Bottom Panel)

- Scrollable log output with timestamps
- Color-coded messages: info (white), warning (yellow), error (red), success (green)
- Display epoch logs, shape validation errors, gradient warnings, and API responses
- A clear button and a filter by log level

---

## Beginner vs. Expert Mode

Add a toggle in the top bar for **Beginner / Expert mode**:
- **Beginner mode**: Show tooltips everywhere, hide advanced options, display a "Getting Started" guided walkthrough overlay, suggest common architectures from a template gallery
- **Expert mode**: Show all raw configuration options, enable custom code blocks, expose full API controls, allow manual tensor shape overrides

---

## Template Gallery

Include a "New from Template" modal with pre-built network templates:
- Simple MLP (MNIST classifier)
- CNN (Image classifier)
- LSTM (Sequence model)
- Transformer (Text classification)
- Autoencoder

Each template loads a pre-configured graph into the canvas instantly.

---

## Design System & Styling

- **Theme**: Dark mode by default (deep navy/charcoal background, neon accent colors for nodes and highlights), with a light mode toggle
- **Typography**: Monospace font for code/logs, clean sans-serif for UI
- **Color palette**: Dark background `#0f1117`, panel background `#1a1d27`, accent blue `#4f8ef7`, accent purple `#9b59f7`, success green `#2ecc71`, warning yellow `#f39c12`, error red `#e74c3c`
- **Nodes on canvas**: Rounded rectangles with a colored left border indicating layer type, subtle drop shadows
- **Animations**: Smooth transitions when switching views, pulse animation on active training nodes, animated data flow along edges during inference

---

## Technical Requirements

- Use **React** with functional components and hooks throughout
- Use **React Context** or a state management solution (Zustand or Redux) for global network state
- Use **React Flow** or a custom SVG/Canvas solution for the graph canvas
- Use **Recharts** or **Chart.js** for training metric charts
- Integrate `@g-js-api/g.js` for all neural network computation
- Fully responsive layout that works on large desktop screens (primary target: 1440px+)
- Keyboard shortcuts: `Space` to run, `Ctrl+S` to save, `Delete` to remove selected node, `Ctrl+Z` to undo
- Undo/redo history for graph edits