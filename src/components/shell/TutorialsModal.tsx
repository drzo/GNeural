import { CircleCheck as CheckCircle2, GraduationCap, Play, RotateCcw } from 'lucide-react';
import { useState } from 'react';
import { completedTutorials, startTutorial, TUTORIALS } from '../../data/tutorials';
import { useStore } from '../../store/useStore';
import { Modal } from '../ui/primitives';

const LEVEL_STYLE = {
  Beginner: 'bg-ok-500/12 text-ok-400',
  Intermediate: 'bg-warn-500/12 text-warn-400',
  Advanced: 'bg-bad-500/12 text-bad-400',
} as const;

export function TutorialsModal() {
  const setModal = useStore((s) => s.setModal);
  const dirty = useStore((s) => s.dirty);
  const [done] = useState(() => new Set(completedTutorials()));

  const begin = (id: string, replacesNetwork: boolean) => {
    if (replacesNetwork && dirty && !window.confirm('This tutorial loads its own network and replaces your unsaved changes. Continue?')) return;
    startTutorial(id);
  };

  return (
    <Modal title="Tutorials" subtitle={`${done.size} of ${TUTORIALS.length} completed. Lessons are hands-on: the guide waits for you to do each step.`} onClose={() => setModal(null)} width={760}>
      <div className="grid grid-cols-2 gap-3 p-6">
        {TUTORIALS.map((t, i) => {
          const finished = done.has(t.id);
          const replaces = !!t.setup && !['tour', 'datasets'].includes(t.id);
          return (
            <button
              key={t.id}
              onClick={() => begin(t.id, replaces)}
              className="group relative flex flex-col rounded-xl border border-line bg-bg p-4 text-left transition-all hover:-translate-y-0.5 hover:border-brand-500/50 hover:shadow-pop"
            >
              <div className="mb-3 flex items-center justify-between">
                <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-raised font-mono text-[12px] font-semibold text-muted group-hover:bg-brand-500/15 group-hover:text-brand-400">
                  {finished ? <CheckCircle2 size={16} className="text-ok-400" /> : i + 1}
                </span>
                <span className={`chip ${LEVEL_STYLE[t.level]}`}>{t.level}</span>
              </div>
              <div className="text-[14px] font-semibold">{t.title}</div>
              <p className="mt-1 flex-1 text-[12px] text-muted">{t.summary}</p>
              <div className="mt-3 flex items-center justify-between text-[11px] text-faint">
                <span>{t.steps.length} steps{replaces ? ' / loads a network' : ''}</span>
                <span className="flex items-center gap-1 font-medium text-brand-400 opacity-0 transition-opacity group-hover:opacity-100">
                  {finished ? <><RotateCcw size={11} /> Replay</> : <><Play size={11} fill="currentColor" /> Start</>}
                </span>
              </div>
            </button>
          );
        })}
      </div>
      <div className="flex items-center gap-2 border-t border-line px-6 py-3 text-[12px] text-faint">
        <GraduationCap size={14} /> Press Esc to leave a tutorial at any time, or Skip a step you have already done.
      </div>
    </Modal>
  );
}
