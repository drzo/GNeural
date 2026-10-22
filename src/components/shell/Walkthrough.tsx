import { ArrowLeft, ArrowRight, Sparkles, X } from 'lucide-react';
import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { useStore } from '../../store/useStore';

const STEPS = [
  { target: null, title: 'Welcome to Gneural', body: 'A visual workbench for designing, training and understanding neural networks right in your browser. This short tour shows you around.' },
  { target: 'palette', title: 'Layer palette', body: 'Every building block lives here. Drag a layer onto the canvas, or double-click it to append it to the end of your network. Hover any layer to learn what it does.' },
  { target: 'canvas', title: 'The canvas', body: 'Layers appear as cards. Drag from the dot on the right of one card to the left of another to connect them. Data flows from the Input layer along the arrows.' },
  { target: 'inspector', title: 'Properties and Run', body: 'Click a layer to tune it here. The Run tab holds the optimizer, loss, epochs and the Train button.' },
  { target: 'views', title: 'Six views of one network', body: 'Switch between building, architecture shapes, live training charts, trying predictions, layer-by-layer debugging and dataset setup.' },
  { target: 'console', title: 'Console', body: 'Training progress, warnings and errors are logged here with timestamps. Press Space at any time to start training.' },
] as const;

export const TOUR_KEY = 'gneural.tour.v1';

export function Walkthrough() {
  const step = useStore((s) => s.walkthrough);
  const setStep = useStore((s) => s.setWalkthrough);
  const setModal = useStore((s) => s.setModal);
  const [rect, setRect] = useState<DOMRect | null>(null);
  const [cardH, setCardH] = useState(200);
  const cardRef = useRef<HTMLDivElement>(null);
  const cur = step === null ? null : STEPS[step];

  useLayoutEffect(() => {
    if (cardRef.current) setCardH(cardRef.current.offsetHeight);
  }, [step, rect]);

  useLayoutEffect(() => {
    if (!cur?.target) return setRect(null);
    const measure = () => setRect(document.querySelector(`[data-tour="${cur.target}"]`)?.getBoundingClientRect() ?? null);
    measure();
    window.addEventListener('resize', measure);
    return () => window.removeEventListener('resize', measure);
  }, [cur]);

  useEffect(() => {
    if (step === null) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') finish();
      if (e.key === 'ArrowRight') next();
      if (e.key === 'ArrowLeft' && step > 0) setStep(step - 1);
    };
    window.addEventListener('keydown', onKey, true);
    return () => window.removeEventListener('keydown', onKey, true);
  });

  if (step === null || !cur) return null;

  const finish = () => {
    localStorage.setItem(TOUR_KEY, '1');
    setStep(null);
  };
  const next = () => (step < STEPS.length - 1 ? setStep(step + 1) : finish());
  const last = step === STEPS.length - 1;

  const pad = 6;
  const card = { w: 340, h: cardH };
  let pos: React.CSSProperties = { left: '50%', top: '50%', transform: 'translate(-50%,-50%)' };
  if (rect) {
    const vw = window.innerWidth, vh = window.innerHeight, gap = 16;
    const centerX = rect.left + rect.width / 2 - card.w / 2;
    let left: number, top: number;
    if (rect.right + card.w + gap * 2 < vw) [left, top] = [rect.right + gap, rect.top + 24];
    else if (rect.left - card.w - gap * 2 > 0) [left, top] = [rect.left - card.w - gap, rect.top + 24];
    else if (rect.top - card.h - gap * 2 > 0) [left, top] = [centerX, rect.top - card.h - gap];
    else if (rect.bottom + card.h + gap * 2 < vh) [left, top] = [centerX, rect.bottom + gap];
    else [left, top] = [rect.right - card.w - gap * 2, rect.bottom - card.h - gap * 2];
    pos = {
      left: Math.min(vw - card.w - gap, Math.max(gap, left)),
      top: Math.min(vh - card.h - gap, Math.max(gap, top)),
    };
  }

  return createPortal(
    <div className="fixed inset-0 z-[90]">
      {rect ? (
        <div
          className="pointer-events-none absolute rounded-xl border-2 border-brand-500 transition-all duration-300"
          style={{ left: rect.left - pad, top: rect.top - pad, width: rect.width + pad * 2, height: rect.height + pad * 2, boxShadow: '0 0 0 9999px rgb(0 0 0 / 0.62)' }}
        />
      ) : (
        <div className="absolute inset-0 bg-black/60 backdrop-blur-[2px]" />
      )}
      <div key={step} ref={cardRef} className="pop-in absolute rounded-2xl border border-line bg-panel p-5 shadow-pop" style={{ width: card.w, ...pos }}>
        <div className="mb-2 flex items-center justify-between">
          <span className="flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wider text-brand-400"><Sparkles size={12} /> Getting started</span>
          <button className="icon-btn -mr-2 -mt-1 h-7 w-7" onClick={finish} aria-label="Close tour"><X size={14} /></button>
        </div>
        <h3 className="text-[16px] font-semibold">{cur.title}</h3>
        <p className="mt-1.5 text-[13px] text-muted">{cur.body}</p>
        <div className="mt-5 flex items-center justify-between">
          <div className="flex gap-1">
            {STEPS.map((_, i) => <span key={i} className={`h-1.5 rounded-full transition-all ${i === step ? 'w-5 bg-brand-500' : 'w-1.5 bg-line'}`} />)}
          </div>
          <div className="flex gap-1.5">
            {step > 0 && <button className="btn-ghost h-7 px-2" onClick={() => setStep(step - 1)}><ArrowLeft size={13} /></button>}
            {last ? (
              <>
                <button className="btn-outline h-7" onClick={finish}>Done</button>
                <button className="btn-primary h-7" onClick={() => { finish(); setModal('templates'); }}>Pick a template</button>
              </>
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
