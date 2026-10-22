import { Boxes, Bug, ChartLine, Database, Download, FilePlus as FilePlus2, FolderOpen, Gauge, GitBranch, GraduationCap, LayoutTemplate, Loader as Loader2, Moon, Network, Pause, Play, Redo2, Save, Square, Sun, Undo2 } from 'lucide-react';
import { useState } from 'react';
import { saveNetwork, startTraining, stopTraining, pauseTraining } from '../../store/actions';
import { useStore, type ViewName } from '../../store/useStore';
import { Segmented, Tooltip } from '../ui/primitives';

const VIEWS: { id: ViewName; label: string; icon: typeof Network; hint: string }[] = [
  { id: 'graph', label: 'Graph', icon: Network, hint: 'Build your network by dragging and connecting layers' },
  { id: 'architecture', label: 'Architecture', icon: GitBranch, hint: 'See tensor shapes and parameter counts flow top to bottom' },
  { id: 'training', label: 'Training', icon: ChartLine, hint: 'Watch loss and accuracy improve live' },
  { id: 'inference', label: 'Inference', icon: Gauge, hint: 'Try the trained network on your own input' },
  { id: 'debug', label: 'Debug', icon: Bug, hint: 'Step through layers and inspect activations and gradients' },
  { id: 'dataset', label: 'Dataset', icon: Database, hint: 'Pick data, set splits and augmentation' },
];

export function TopBar() {
  const s = useStore();
  const [saving, setSaving] = useState(false);
  const beginner = s.mode === 'beginner';
  const running = s.training.status === 'running';
  const paused = s.training.status === 'paused';

  const save = async () => {
    setSaving(true);
    await saveNetwork();
    setSaving(false);
  };

  return (
    <header className="relative z-20 flex h-14 shrink-0 items-center gap-3 border-b border-line bg-panel px-3">
      <div className="flex items-center gap-2.5 pr-2">
        <div className="relative flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-brand-500 to-ok-500 shadow-[0_6px_18px_-6px_rgb(79_142_247/0.8)]">
          <Boxes size={17} className="text-white" />
        </div>
        <div className="leading-tight">
          <div className="text-[14px] font-semibold tracking-tight">Gneural</div>
          <input
            value={s.networkName}
            onChange={(e) => s.setName(e.target.value)}
            className="w-36 truncate bg-transparent text-[11px] text-faint outline-none hover:text-muted focus:text-fg"
            aria-label="Network name"
          />
        </div>
        {s.dirty && <span className="h-1.5 w-1.5 rounded-full bg-warn-500" title="Unsaved changes" />}
      </div>

      <div className="flex items-center gap-0.5 border-l border-line pl-2">
        <Tooltip content="New empty network" side="bottom"><button className="icon-btn" onClick={() => s.newNetwork()}><FilePlus2 size={16} /></button></Tooltip>
        <Tooltip content="Start from a template" side="bottom"><button className="icon-btn" onClick={() => s.setModal('templates')}><LayoutTemplate size={16} /></button></Tooltip>
        <Tooltip content="Save to the cloud (Ctrl+S)" side="bottom">
          <button className="icon-btn" onClick={save} disabled={saving}>{saving ? <Loader2 size={16} className="animate-spin" /> : <Save size={16} />}</button>
        </Tooltip>
        <Tooltip content="Open a saved network" side="bottom"><button className="icon-btn" onClick={() => s.setModal('load')}><FolderOpen size={16} /></button></Tooltip>
        <Tooltip content="Export as JSON or a Geometry Dash level" side="bottom"><button className="icon-btn" data-tour="export-button" onClick={() => s.setModal('export')}><Download size={16} /></button></Tooltip>
        <span className="mx-1 h-5 w-px bg-line" />
        <Tooltip content="Undo (Ctrl+Z)" side="bottom"><button className="icon-btn" onClick={s.undo} disabled={!s.past.length}><Undo2 size={16} /></button></Tooltip>
        <Tooltip content="Redo (Ctrl+Shift+Z)" side="bottom"><button className="icon-btn" onClick={s.redo} disabled={!s.future.length}><Redo2 size={16} /></button></Tooltip>
      </div>

      <nav className="mx-auto flex items-center gap-0.5 rounded-xl border border-line bg-bg p-1" data-tour="views">
        {VIEWS.map((v) => {
          const active = s.view === v.id;
          return (
            <Tooltip key={v.id} content={v.hint} side="bottom" disabled={!beginner}>
              <button
                onClick={() => s.setView(v.id)}
                data-tour={`view-${v.id}`}
                className={`relative flex h-8 items-center gap-1.5 rounded-lg px-3 text-[13px] font-medium transition-all ${
                  active ? 'bg-raised text-fg shadow-sm' : 'text-faint hover:text-muted'
                }`}
              >
                <v.icon size={14} className={active ? 'text-brand-400' : ''} />
                {v.label}
                {v.id === 'training' && (running || paused) && <span className={`h-1.5 w-1.5 rounded-full ${running ? 'animate-pulse bg-ok-500' : 'bg-warn-500'}`} />}
              </button>
            </Tooltip>
          );
        })}
      </nav>

      <div className="flex items-center gap-2">
        {running ? (
          <div className="flex items-center gap-1">
            {s.training.engine === 'workbench' && (
              <button className="btn-outline" onClick={pauseTraining}><Pause size={14} /> Pause</button>
            )}
            <button className="btn-danger" onClick={stopTraining}><Square size={13} /> Stop</button>
          </div>
        ) : (
          <div className="flex items-center gap-1">
            <Tooltip content="Train the network (Space)" side="bottom">
              <button className="btn-primary" onClick={startTraining}><Play size={14} fill="currentColor" /> {paused ? 'Resume' : 'Run'}</button>
            </Tooltip>
            {paused && <button className="btn-danger" onClick={stopTraining}><Square size={13} /></button>}
          </div>
        )}
        <Tooltip content="Tutorials and the guided tour" side="bottom">
          <button className="btn-ghost h-8 px-2.5 text-[13px]" data-tour="tutorials-button" onClick={() => s.setModal('tutorials')}>
            <GraduationCap size={15} /> Learn
          </button>
        </Tooltip>
        <Segmented
          size="sm"
          value={s.mode}
          onChange={s.setMode}
          options={[
            { value: 'beginner', label: 'Beginner', title: 'Guided mode with tooltips and simplified options' },
            { value: 'expert', label: 'Expert', title: 'All options, custom code and shape overrides' },
          ]}
        />
        <Tooltip content={s.theme === 'dark' ? 'Light theme' : 'Dark theme'} side="bottom">
          <button className="icon-btn" onClick={s.toggleTheme}>{s.theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}</button>
        </Tooltip>
      </div>
    </header>
  );
}
