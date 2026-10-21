import { ArrowRight, Gamepad2 } from 'lucide-react';
import { layerDef } from '../../engine/catalog';
import { TEMPLATES } from '../../data/templates';
import { BUILTIN_DATASETS } from '../../data/datasets';
import { layerColor } from '../../lib/format';
import { useStore } from '../../store/useStore';
import { LayerIcon } from '../ui/LayerIcon';
import { Modal } from '../ui/primitives';

const DIFFICULTY = { Beginner: 'bg-ok-500/15 text-ok-400', Intermediate: 'bg-warn-500/15 text-warn-400', Advanced: 'bg-bad-500/15 text-bad-400' };

export function TemplateModal() {
  const close = () => useStore.getState().setModal(null);
  const load = useStore((s) => s.loadTemplate);
  return (
    <Modal title="Start from a template" subtitle="Each template comes with a matching dataset and tuned training settings." onClose={close} width={920}>
      <div className="grid grid-cols-3 gap-4 p-6">
        {TEMPLATES.map((t) => {
          const ds = BUILTIN_DATASETS.find((d) => d.id === t.datasetId);
          return (
            <button
              key={t.id}
              onClick={() => load(t)}
              className="group flex flex-col rounded-xl border border-line bg-bg p-4 text-left transition-all hover:-translate-y-0.5 hover:border-brand-500/60 hover:shadow-pop"
            >
              <div className="mb-3 flex items-center gap-1">
                {t.layers.map(([type], i) => {
                  const role = i === 0 ? 'input' : i === t.layers.length - 1 ? 'output' : 'hidden';
                  const c = layerColor(type, role);
                  return (
                    <span key={i} title={layerDef(type).label} className="flex h-6 w-6 items-center justify-center rounded" style={{ background: `${c}22`, color: c }}>
                      <LayerIcon type={type} size={12} />
                    </span>
                  );
                })}
              </div>
              <div className="flex items-center gap-2">
                <h3 className="text-[15px] font-semibold">{t.name}</h3>
                {t.gd && <Gamepad2 size={14} className="text-ok-400" />}
              </div>
              <p className="text-[12px] font-medium text-brand-400">{t.tagline}</p>
              <p className="mt-2 flex-1 text-[12px] text-muted">{t.description}</p>
              <div className="mt-4 flex items-center justify-between">
                <div className="flex items-center gap-1.5">
                  <span className={`chip ${DIFFICULTY[t.difficulty]}`}>{t.difficulty}</span>
                  {ds && <span className="chip bg-raised text-faint">{ds.name}</span>}
                </div>
                <ArrowRight size={15} className="text-faint transition-transform group-hover:translate-x-0.5 group-hover:text-brand-400" />
              </div>
            </button>
          );
        })}
      </div>
    </Modal>
  );
}
