import { Check, Copy, Download, File as FileJson, Gamepad2, Loader as Loader2, Circle as XCircle } from 'lucide-react';
import { useMemo, useState } from 'react';
import { gneuralCompatibility } from '../../gd/bridge';
import { exportGeometryDash, exportJson, type GdExport } from '../../store/actions';
import { useNetwork } from '../../store/useNetwork';
import { useStore } from '../../store/useStore';
import { Modal, Segmented } from '../ui/primitives';

function download(name: string, text: string, type: string) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

function CopyButton({ text }: { text: string }) {
  const [done, setDone] = useState(false);
  return (
    <button
      className="btn-outline"
      onClick={async () => {
        await navigator.clipboard.writeText(text).catch(() => undefined);
        setDone(true);
        window.setTimeout(() => setDone(false), 1400);
      }}
    >
      {done ? <Check size={14} className="text-ok-400" /> : <Copy size={14} />} {done ? 'Copied' : 'Copy'}
    </button>
  );
}

const slug = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'network';

function GdPanel() {
  const net = useNetwork();
  const name = useStore((s) => s.networkName);
  const compat = useMemo(() => gneuralCompatibility(net.chain, net.analysis), [net]);
  const inputs = compat.ok ? compat.arch.inputs : 0;
  const sample = net.dataset?.samples[0];
  const [values, setValues] = useState<number[]>(() => Array.from({ length: inputs }, (_, i) => Number((sample?.x[i] ?? 1).toFixed(3))));
  const [spacing, setSpacing] = useState(30);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<GdExport | null>(null);
  const [error, setError] = useState<string | null>(null);

  if (!compat.ok) {
    return (
      <div className="p-6">
        <div className="flex gap-3 rounded-xl bg-bad-500/10 p-4 text-[13px] text-bad-400">
          <XCircle size={18} className="shrink-0" />
          <div>
            <div className="font-semibold">This network cannot be exported to Geometry Dash</div>
            <p className="mt-1 text-bad-400/90">{compat.reason}</p>
            <p className="mt-2 text-muted">Geometry Dash export supports flat inputs, ReLU Dense hidden layers and a ReLU output (or a Linear output with one hidden layer). The "Geometry Dash MLP" template is a ready-made example.</p>
          </div>
        </div>
      </div>
    );
  }

  const generate = async () => {
    setBusy(true);
    setError(null);
    try {
      setResult(await exportGeometryDash(values.slice(0, inputs), spacing));
    } catch (e) {
      setError((e as Error).message);
    }
    setBusy(false);
  };

  return (
    <div className="space-y-5 p-6">
      <p className="text-[13px] text-muted">
        Compiles your trained {compat.arch.inputs}-{compat.arch.hidden.join('-')}-{compat.arch.outputs} network into Geometry Dash objects using the Gneural library. Counters display every neuron so you can watch it compute inside the level.
      </p>
      <div className="grid grid-cols-[1fr_140px] gap-4">
        <div>
          <div className="label mb-2">Starting input values</div>
          <div className="flex flex-wrap gap-2">
            {Array.from({ length: inputs }, (_, i) => (
              <input
                key={i}
                type="number"
                className="input w-24 font-mono"
                value={values[i] ?? 0}
                step={0.1}
                onChange={(e) => setValues((v) => Object.assign([...v], { [i]: Number(e.target.value) }))}
                aria-label={`Input ${i + 1}`}
              />
            ))}
          </div>
        </div>
        <div>
          <div className="label mb-2">Counter spacing</div>
          <input type="number" className="input font-mono" min={10} max={120} value={spacing} onChange={(e) => setSpacing(Math.max(10, Math.min(120, Number(e.target.value) || 30)))} />
        </div>
      </div>
      <button className="btn-primary" onClick={generate} disabled={busy}>
        {busy ? <Loader2 size={14} className="animate-spin" /> : <Gamepad2 size={14} />} {busy ? 'Compiling level' : 'Generate level string'}
      </button>
      {error && <div className="rounded-lg bg-bad-500/10 px-3 py-2 text-[13px] text-bad-400">{error}</div>}
      {result && (
        <div className="pop-in space-y-3">
          <div className="grid grid-cols-3 gap-3">
            {[
              ['Objects', result.objects],
              ['Groups', result.groups],
              ['Outputs', result.outputs.map((o) => o.toFixed(3)).join(', ')],
            ].map(([k, v]) => (
              <div key={k} className="rounded-lg border border-line bg-bg px-3 py-2">
                <div className="label">{k}</div>
                <div className="mt-0.5 truncate font-mono text-[14px]">{v}</div>
              </div>
            ))}
          </div>
          <textarea readOnly className="input h-32 resize-none py-2 font-mono text-[11px] text-muted" value={result.levelString} />
          <div className="flex gap-2">
            <CopyButton text={result.levelString} />
            <button className="btn-outline" onClick={() => download(`${slug(name)}.gd.txt`, result.levelString, 'text/plain')}><Download size={14} /> Download</button>
          </div>
          <p className="text-[12px] text-faint">Paste the level string into a level editor tool such as GDShare, or import it with the g.js toolchain.</p>
        </div>
      )}
    </div>
  );
}

function JsonPanel() {
  const name = useStore((s) => s.networkName);
  const json = useMemo(() => exportJson(), []);
  const kb = (json.length / 1024).toFixed(1);
  return (
    <div className="space-y-3 p-6">
      <p className="text-[13px] text-muted">The full architecture with inferred shapes, training settings and, if the network has been built, its weights.</p>
      <pre className="max-h-[360px] overflow-auto rounded-lg border border-line bg-bg p-3 font-mono text-[11px] leading-relaxed text-muted">{json.length > 60000 ? `${json.slice(0, 60000)}\n...` : json}</pre>
      <div className="flex items-center gap-2">
        <CopyButton text={json} />
        <button className="btn-primary" onClick={() => download(`${slug(name)}.json`, json, 'application/json')}><Download size={14} /> Download JSON</button>
        <span className="ml-auto font-mono text-[11px] text-faint">{kb} KB</span>
      </div>
    </div>
  );
}

export function ExportModal() {
  const [tab, setTab] = useState<'json' | 'gd'>('json');
  return (
    <Modal title="Export network" onClose={() => useStore.getState().setModal(null)} width={720}>
      <div className="border-b border-line px-6 py-3">
        <Segmented
          value={tab}
          onChange={(v) => setTab(v === 'gd' ? 'gd' : 'json')}
          options={[
            { value: 'json', label: <><FileJson size={13} /> JSON</> },
            { value: 'gd', label: <><Gamepad2 size={13} /> Geometry Dash level</> },
          ]}
        />
      </div>
      {tab === 'json' ? <JsonPanel /> : <GdPanel />}
    </Modal>
  );
}
