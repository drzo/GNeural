import { ArrowLeft, ArrowRight, CircleCheck as CheckCircle2, Circle, GraduationCap, X } from 'lucide-react';
import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { markTutorialDone, TUTORIALS } from '../../data/tutorials';
import { useStore } from '../../store/useStore';

export const TOUR_KEY = 'gneural.tour.v1';

const sameRect = (a: DOMRect | null, b: DOMRect | null) =>
  a === b || (!!a && !!b && a.left === b.left && a.top === b.top && a.width === b.width && a.height === b.height);

export function Walkthrough() {
  const progress = useStore((s) => s.tutorial);
  const setTutorial = useStore((s) => s.setTutorial);
  const setModal = useStore((s) => s.setModal);
  const tutorial = progress ? TUTORIALS.find((t) => t.id === progress.id) ?? null : null;
  const step = progress?.step ?? 0;
  const cur = tutorial?.steps[step] ?? null;
  const taskDone = useStore((s) => (cur?.task ? cur.task.done(s) : true));
  const [rect, setRect] = useState<DOMRect | null>(null);
  const [cardH, setCardH] = useState(220);
  const cardRef = useRef<HTMLDivElement>(null);

  const go = (n: number) => progress && setTutorial({ id: progress.id, step: n });
  const close = (completed: boolean) => {
    if (!progress) return;
    if (completed) markTutorialDone(progress.id);
    if (progress.id === 'tour') localStorage.setItem(TOUR_KEY, '1');
    setTutorial(null);
  };
  const last = !!tutorial && step === tutorial.steps.length - 1;
  const next = () => (last ? close(true) : go(step + 1));

  useLayoutEffect(() => {
    if (!cur?.target) return setRect(null);
    const measure = () => {
      const r = document.querySelector(`[data-tour="${cur.target}"]`)?.getBoundingClientRect() ?? null;
      setRect((prev) => (sameRect(prev, r) ? prev : r));
    };
    measure();
    const id = window.setInterval(measure, 300);
    window.addEventListener('resize', measure);
    return () => {
      window.clearInterval(id);
      window.removeEventListener('resize', measure);
    };
  }, [cur]);

  useLayoutEffect(() => {
    if (cardRef.current) setCardH(cardRef.current.offsetHeight);
  }, [cur, rect, taskDone]);

  useEffect(() => {
    if (!cur?.task || !taskDone) return;
    const id = window.setTimeout(next, 650);
    return () => window.clearTimeout(id);
  });

  useEffect(() => {
    if (!progress) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') close(false);
      const typing = e.target instanceof HTMLElement && ['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName);
      if (typing) return;
      if (e.key === 'ArrowRight' && taskDone) next();
      if (e.key === 'ArrowLeft' && step > 0) go(step - 1);
    };
    window.addEventListener('keydown', onKey, true);
    return () => window.removeEventListener('keydown', onKey, true);
  });

  if (!tutorial || !cur) return null;

  const interactive = !!cur.task || !!cur.corner;
  const card = { w: 352, h: cardH };
  const vw = window.innerWidth, vh = window.innerHeight, gap = 16;
  let pos: React.CSSProperties = { left: '50%', top: '50%', transform: 'translate(-50%,-50%)' };
  if (cur.corner) {
    const r = rect ?? new DOMRect(0, 0, vw, vh);
    pos = { left: Math.max(gap, r.right - card.w - gap * 1.5), top: Math.max(gap, Math.min(vh - card.h - gap, r.bottom - card.h - gap * 1.5)) };
  } else if (rect) {
    const centerX = rect.left + rect.width / 2 - card.w / 2;
    let left: number, top: number;
    if (rect.right + card.w + gap * 2 < vw) [left, top] = [rect.right + gap, rect.top + 24];
    else if (rect.left - card.w - gap * 2 > 0) [left, top] = [rect.left - card.w - gap, rect.top + 24];
    else if (rect.bottom + card.h + gap * 2 < vh) [left, top] = [centerX, rect.bottom + gap];
    else if (rect.top - card.h - gap * 2 > 0) [left, top] = [centerX, rect.top - card.h - gap];
    else [left, top] = [rect.right - card.w - gap * 2, rect.bottom - card.h - gap * 2];
    pos = { left: Math.min(vw - card.w - gap, Math.max(gap, left)), top: Math.min(vh - card.h - gap, Math.max(gap, top)) };
  }

  const pad = 6;
  return createPortal(
    <div className={`fixed inset-0 z-[90] ${interactive ? 'pointer-events-none' : ''}`}>
      {rect ? (
        <div
          className="pointer-events-none absolute rounded-xl border-2 border-brand-500 transition-all duration-300"
          style={{
            left: rect.left - pad, top: rect.top - pad, width: rect.width + pad * 2, height: rect.height + pad * 2,
            boxShadow: cur.corner ? '0 0 0 4px rgb(79 142 247 / 0.18)' : `0 0 0 9999px rgb(0 0 0 / ${interactive ? 0.45 : 0.62})`,
          }}
        />
      ) : (
        !cur.corner && <div className="absolute inset-0 bg-black/60 backdrop-blur-[2px]" />
      )}
      <div key={`${tutorial.id}-${step}`} ref={cardRef} className="pop-in pointer-events-auto absolute rounded-2xl border border-line bg-panel p-5 shadow-pop" style={{ width: card.w, ...pos }}>
        <div className="mb-2 flex items-center justify-between gap-2">
          <span className="flex min-w-0 items-center gap-1.5 text-[11px] font-medium uppercase tracking-wider text-brand-400">
            <GraduationCap size={13} className="shrink-0" /> <span className="truncate">{tutorial.title}</span>
          </span>
          <button className="icon-btn -mr-2 -mt-1 h-7 w-7 shrink-0" onClick={() => close(false)} aria-label="Close tutorial"><X size={14} /></button>
        </div>
        <h3 className="text-[16px] font-semibold leading-tight">{cur.title}</h3>
        <p className="mt-1.5 text-[13px] text-muted">{cur.body}</p>
        {cur.task && (
          <div className={`mt-3 flex items-center gap-2 rounded-lg px-3 py-2 text-[12px] font-medium transition-colors ${taskDone ? 'bg-ok-500/12 text-ok-400' : 'bg-brand-500/10 text-brand-300'}`}>
            {taskDone ? <CheckCircle2 size={14} /> : <Circle size={14} className="animate-pulse" />}
            {taskDone ? 'Nice, moving on...' : cur.task.hint}
          </div>
        )}
        <div className="mt-4 flex items-center justify-between">
          <div className="flex items-center gap-1">
            {tutorial.steps.map((_, i) => <span key={i} className={`h-1.5 rounded-full transition-all ${i === step ? 'w-5 bg-brand-500' : i < step ? 'w-1.5 bg-brand-500/50' : 'w-1.5 bg-line'}`} />)}
            <span className="ml-2 font-mono text-[10px] text-faint">{step + 1}/{tutorial.steps.length}</span>
          </div>
          <div className="flex gap-1.5">
            {step > 0 && <button className="btn-ghost h-7 px-2" onClick={() => go(step - 1)} aria-label="Previous step"><ArrowLeft size={13} /></button>}
            {last ? (
              <>
                <button className="btn-outline h-7" onClick={() => { close(true); setModal('tutorials'); }}>More</button>
                <button className="btn-primary h-7" onClick={() => close(true)}>Finish</button>
              </>
            ) : cur.task && !taskDone ? (
              <button className="btn-ghost h-7 text-[12px] text-faint" onClick={next}>Skip</button>
            ) : (
              <button className="btn-primary h-7" onClick={next}>Next <ArrowRight size={13} /></button>
            )}
          </div>
        </div>
      </div>
    </div>,
    document.body,
  );
}
