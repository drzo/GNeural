import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath } from 'node:url';

const shim = (name: string) => fileURLToPath(new URL(`./src/gd/shims/${name}.js`, import.meta.url));

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      gneural: fileURLToPath(new URL('./index.js', import.meta.url)),
      ws: shim('ws'),
      crypto: shim('crypto'),
      fs: shim('empty'),
      zlib: shim('empty'),
      stream: shim('stream'),
    },
  },
  worker: { format: 'es' },
  build: { chunkSizeWarningLimit: 2000 },
});
