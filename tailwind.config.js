/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        bg: 'rgb(var(--bg) / <alpha-value>)',
        panel: 'rgb(var(--panel) / <alpha-value>)',
        raised: 'rgb(var(--raised) / <alpha-value>)',
        line: 'rgb(var(--line) / <alpha-value>)',
        fg: 'rgb(var(--fg) / <alpha-value>)',
        muted: 'rgb(var(--muted) / <alpha-value>)',
        faint: 'rgb(var(--faint) / <alpha-value>)',
        brand: { 300: '#8fb5fa', 400: '#6a9ff8', 500: '#4f8ef7', 600: '#3a74d8', 700: '#2c5bb0' },
        violet: { 400: '#b384f9', 500: '#9b59f7', 600: '#7f3fe0' },
        ok: { 400: '#58d68d', 500: '#2ecc71', 600: '#25a75c' },
        warn: { 400: '#f5b041', 500: '#f39c12', 600: '#d68910' },
        bad: { 400: '#ec7063', 500: '#e74c3c', 600: '#c0392b' },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'monospace'],
      },
      boxShadow: {
        node: '0 4px 18px -6px rgb(0 0 0 / 0.45), 0 1px 0 0 rgb(255 255 255 / 0.03) inset',
        pop: '0 24px 60px -12px rgb(0 0 0 / 0.55)',
      },
    },
  },
  plugins: [],
};
