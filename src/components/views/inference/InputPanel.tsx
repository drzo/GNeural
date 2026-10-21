import { Eraser, Shuffle } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { encodeText, waveform, WAVE_CLASSES, type Dataset } from '../../../data/datasets';

export interface InputValue {
  x: Float32Array;
  label: number | null;
  note?: string;
}

function DrawPad({ x, h, w, onChange }: { x: Float32Array; h: number; w: number; onChange(x: Float32Array): void }) {
  const painting = useRef<0 | 1 | -1>(0);
  const paint = (i: number, mode: 1 | -1) => {
    const next = x.slice();
    next[i] = mode === 1 ? 1 : 0;
    const r = Math.floor(i / w), c = i % w;
    if (mode === 1) {
      for (const [dr, dc] of [[0, 1], [1, 0], [0, -1], [-1, 0]]) {
        const rr = r + dr, cc = c + dc;
        if (rr >= 0 && rr < h && cc >= 0 && cc < w) next[rr * w + cc] = Math.max(next[rr * w + cc], 0.35);
      }
    }
    onChange(next);
  };
  useEffect(() => {
    const up = () => (painting.current = 0);
    window.addEventListener('mouseup', up);
    return () => window.removeEventListener('mouseup', up);
  }, []);
  return (
    <div className="inline-grid select-none gap-[2px] rounded-xl border border-line bg-bg p-2" style={{ gridTemplateColumns: `repeat(${w}, 26px)` }} onContextMenu={(e) => e.preventDefault()}>
      {Array.from({ length: h * w }, (_, i) => (
        <div
          key={i}
          onMouseDown={(e) => {
            painting.current = e.button === 2 ? -1 : 1;
            paint(i, painting.current);
          }}
          onMouseEnter={() => painting.current && paint(i, painting.current)}
          className="h-[26px] w-[26px] cursor-crosshair rounded-[4px] transition-colors"
          style={{ background: `rgb(79 142 247 / ${0.06 + x[i] * 0.94})` }}
        />
      ))}
    </div>
  );
}

function WavePlot({ x }: { x: Float32Array }) {
  const pts = Array.from(x, (v, i) => `${(i / (x.length - 1)) * 300},${60 - v * 50}`).join(' ');
  return (
    <svg viewBox="0 -2 300 124" className="h-32 w-full rounded-xl border border-line bg-bg">
      <line x1="0" x2="300" y1="60" y2="60" stroke="#8b93a733" />
      <polyline points={pts} fill="none" stroke="#1fc7b6" strokeWidth="2.5" strokeLinejoin="round" />
      {Array.from(x, (v, i) => <circle key={i} cx={(i / (x.length - 1)) * 300} cy={60 - v * 50} r="3" fill="#1fc7b6" />)}
    </svg>
  );
}

const EXAMPLES = ['this movie was really great and fun', 'the level felt boring and the music was awful', 'i love this beautiful game', 'quite a disappointing show'];

export function InputPanel({ ds, pool, value, onChange }: { ds: Dataset; pool: Dataset['samples']; value: InputValue | null; onChange(v: InputValue): void }) {
  const [text, setText] = useState(EXAMPLES[0]);
  const [wave, setWave] = useState({ kind: 0, freq: 1.5, phase: 0.1, amp: 0.9 });

  const random = () => {
    const s = pool[Math.floor(Math.random() * pool.length)];
    if (!s) return;
    if (s.text) setText(s.text);
    onChange({ x: s.x, label: ds.task === 'classification' ? s.label : null, note: s.text });
  };

  useEffect(() => {
    if (ds.kind === 'text') onChange({ x: encodeText(text), label: null, note: text });
  }, [text, ds.kind]);

  useEffect(() => {
    if (ds.kind === 'sequence' && ds.id === 'waves') onChange({ x: waveform(wave.kind, wave.freq, wave.phase, wave.amp, ds.inputShape[0]), label: wave.kind });
  }, [wave, ds.id]);

  useEffect(() => {
    if (!value) random();
  }, [ds.id]);

  const x = value?.x;
  const header = (
    <div className="mb-3 flex items-center justify-between">
      <span className="label">Input</span>
      <div className="flex gap-1">
        {ds.kind === 'image' && x && <button className="btn-ghost h-7 px-2 text-[12px]" onClick={() => onChange({ x: new Float32Array(x.length), label: null })}><Eraser size={13} /> Clear</button>}
        <button className="btn-outline h-7 px-2 text-[12px]" onClick={random}><Shuffle size={13} /> Test sample</button>
      </div>
    </div>
  );

  if (ds.kind === 'image' && ds.inputShape.length === 3) {
    const [h, w] = ds.inputShape;
    return (
      <div>
        {header}
        {x && <DrawPad x={x} h={h} w={w} onChange={(nx) => onChange({ x: nx, label: null })} />}
        <p className="mt-2 text-[11px] text-faint">Draw with the left mouse button, erase with the right.</p>
      </div>
    );
  }

  if (ds.kind === 'text') {
    return (
      <div>
        {header}
        <textarea className="input h-24 resize-none py-2 text-[13px]" value={text} onChange={(e) => setText(e.target.value)} placeholder="Type a short review" />
        <div className="mt-2 flex flex-wrap gap-1.5">
          {EXAMPLES.map((e) => <button key={e} className="chip bg-raised text-muted hover:text-fg" onClick={() => setText(e)}>{e.slice(0, 28)}...</button>)}
        </div>
      </div>
    );
  }

  if (ds.id === 'waves') {
    return (
      <div className="space-y-3">
        {header}
        {x && <WavePlot x={x} />}
        <div className="flex gap-1">
          {WAVE_CLASSES.map((w, k) => (
            <button key={w} className={`btn h-7 flex-1 text-[12px] ${wave.kind === k ? 'bg-brand-500/15 text-brand-400' : 'bg-raised text-muted'}`} onClick={() => setWave({ ...wave, kind: k })}>{w}</button>
          ))}
        </div>
        {([['freq', 'Frequency', 1, 2.5], ['phase', 'Phase', 0, 1], ['amp', 'Amplitude', 0.3, 1]] as const).map(([k, label, min, max]) => (
          <label key={k} className="flex items-center gap-3 text-[12px] text-muted">
            <span className="w-20">{label}</span>
            <input type="range" className="flex-1 accent-brand-500" min={min} max={max} step={0.01} value={wave[k]} onChange={(e) => setWave({ ...wave, [k]: Number(e.target.value) })} />
            <span className="w-10 text-right font-mono">{wave[k].toFixed(2)}</span>
          </label>
        ))}
      </div>
    );
  }

  const n = x?.length ?? 0;
  const norm = ds.normalize;
  const raw = (i: number) => (x ? (norm ? x[i] * norm.std[i] + norm.mean[i] : x[i]) : 0);
  return (
    <div>
      {header}
      {ds.kind === 'sequence' && x && <div className="mb-3"><WavePlot x={x} /></div>}
      <div className="grid grid-cols-2 gap-2">
        {Array.from({ length: Math.min(n, 24) }, (_, i) => (
          <label key={i} className="block">
            <span className="mb-1 block truncate text-[11px] text-faint">{ds.featureNames[i] ?? `x${i + 1}`}</span>
            <input
              type="number"
              step={0.1}
              className="input font-mono"
              value={Number(raw(i).toFixed(4))}
              onChange={(e) => {
                if (!x) return;
                const next = x.slice();
                const v = Number(e.target.value);
                next[i] = norm ? (v - norm.mean[i]) / (norm.std[i] || 1) : v;
                onChange({ x: next, label: null });
              }}
            />
          </label>
        ))}
      </div>
      {n > 24 && <p className="mt-2 text-[11px] text-faint">Showing the first 24 of {n} values.</p>}
    </div>
  );
}
