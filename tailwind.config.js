/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'Plus Jakarta Sans', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'SFMono-Regular', 'Menlo', 'monospace'],
        editorial: ['Plus Jakarta Sans', 'Inter', 'sans-serif'],
      },
      colors: {
        paper: {
          DEFAULT: '#FAFAF8',
          50: '#FFFFFF',
          100: '#F5F5F1',
          200: '#EBEBE6',
          300: '#DFDFD8',
        },
        ink: {
          DEFAULT: '#111215',
          light: '#27272A',
          muted: '#52525B',
          faint: '#71717A',
        },
        border: {
          DEFAULT: '#E4E4E7',
          subtle: '#EFEFEA',
          strong: '#D4D4D8',
        },
        accent: {
          DEFAULT: '#0F172A',
          blue: '#2563EB',
          emerald: '#059669',
          amber: '#D97706',
        },
      },
      boxShadow: {
        'subtle': '0 1px 2px 0 rgba(0, 0, 0, 0.04)',
        'elevated': '0 4px 20px -2px rgba(0, 0, 0, 0.05), 0 2px 6px -1px rgba(0, 0, 0, 0.03)',
        'card': '0 1px 3px 0 rgba(0, 0, 0, 0.04), 0 1px 2px -1px rgba(0, 0, 0, 0.02)',
        'modal': '0 25px 50px -12px rgba(0, 0, 0, 0.15)',
      },
      animation: {
        'pulse-subtle': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
    },
  },
  plugins: [],
};