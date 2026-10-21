import { createContext, useContext } from 'react';
import type { NetworkInfo } from '../../../store/useNetwork';

export const NetworkContext = createContext<NetworkInfo | null>(null);

export function useNetworkContext() {
  const ctx = useContext(NetworkContext);
  if (!ctx) throw new Error('NetworkContext missing');
  return ctx;
}
