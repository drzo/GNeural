import { useEffect } from 'react';
import { BottomConsole } from './components/shell/BottomConsole';
import { ExportModal } from './components/shell/ExportModal';
import { LeftSidebar } from './components/shell/LeftSidebar';
import { LoadModal } from './components/shell/LoadModal';
import { RightPanel } from './components/shell/RightPanel';
import { TemplateModal } from './components/shell/TemplateModal';
import { TopBar } from './components/shell/TopBar';
import { useShortcuts } from './components/shell/useShortcuts';
import { TOUR_KEY, Walkthrough } from './components/shell/Walkthrough';
import { ArchitectureView } from './components/views/ArchitectureView';
import { DatasetView } from './components/views/DatasetView';
import { DebugView } from './components/views/DebugView';
import { GraphView } from './components/views/GraphView';
import { InferenceView } from './components/views/InferenceView';
import { TrainingView } from './components/views/TrainingView';
import { useStore } from './store/useStore';

const VIEWS = {
  graph: GraphView,
  architecture: ArchitectureView,
  training: TrainingView,
  inference: InferenceView,
  debug: DebugView,
  dataset: DatasetView,
};

export default function App() {
  const view = useStore((s) => s.view);
  const theme = useStore((s) => s.theme);
  const modal = useStore((s) => s.modal);
  useShortcuts();

  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark');
  }, [theme]);

  useEffect(() => {
    const s = useStore.getState();
    s.log('info', 'Gneural workbench ready. Press Space to train the current network.', 'app');
    if (!localStorage.getItem(TOUR_KEY)) s.setWalkthrough(0);
    const warn = (e: BeforeUnloadEvent) => {
      if (useStore.getState().dirty) e.preventDefault();
    };
    window.addEventListener('beforeunload', warn);
    return () => window.removeEventListener('beforeunload', warn);
  }, []);

  const View = VIEWS[view];

  return (
    <div className="flex h-full min-w-[1280px] flex-col">
      <TopBar />
      <div className="flex min-h-0 flex-1">
        <LeftSidebar />
        <div className="flex min-w-0 flex-1 flex-col">
          <main className="relative min-h-0 flex-1 overflow-hidden bg-bg" data-tour="canvas">
            <div key={view} className="view-enter absolute inset-0">
              <View />
            </div>
          </main>
          <BottomConsole />
        </div>
        <RightPanel />
      </div>
      {modal === 'templates' && <TemplateModal />}
      {modal === 'load' && <LoadModal />}
      {modal === 'export' && <ExportModal />}
      <Walkthrough />
    </div>
  );
}
