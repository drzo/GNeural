import { useEffect } from 'react';
import { saveNetwork, toggleRun } from '../../store/actions';
import { useStore } from '../../store/useStore';

const isTyping = (t: EventTarget | null) =>
  t instanceof HTMLElement && (t.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(t.tagName));

export function useShortcuts() {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const s = useStore.getState();
      const mod = e.ctrlKey || e.metaKey;
      if (mod && e.key.toLowerCase() === 's') {
        e.preventDefault();
        void saveNetwork();
        return;
      }
      if (isTyping(e.target) || s.modal) return;
      if (mod && e.key.toLowerCase() === 'z') {
        e.preventDefault();
        if (e.shiftKey) s.redo();
        else s.undo();
      } else if (mod && e.key.toLowerCase() === 'y') {
        e.preventDefault();
        s.redo();
      } else if (e.key === ' ' && !mod) {
        e.preventDefault();
        toggleRun();
      } else if (e.key === 'Delete' || e.key === 'Backspace') {
        if (s.view !== 'graph' && s.view !== 'architecture') return;
        e.preventDefault();
        s.removeSelected();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);
}
