/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: {
        sans: ['Plus Jakarta Sans', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'SFMono-Regular', 'Menlo', 'monospace'],
        editorial: ['Space Grotesk', 'Plus Jakarta Sans', 'sans-serif'],
        display: ['Space Grotesk', 'Plus Jakarta Sans', 'sans-serif'],
      },
      colors: {
        paper: {
          DEFAULT: '#F5F5F2',
          50: '#FFFFFF',
          100: '#EBEBE5',
          200: '#DEDED6',
          300: '#CECEPE',
        },
        ink: {
          DEFAULT: '#0E1015',
          light: '#252830',
          muted: '#4A4E5A',
          faint: '#767B89',
        },
        border: {
          DEFAULT: '#DFDFD7',
          subtle: '#E9E9E2',
          strong: '#C4C4B8',
        },
        accent: {
          DEFAULT: '#0E1015',
          cobalt: '#1849D6',
          emerald: '#0D7A4E',
        },
      },
      boxShadow: {
        'subtle': '0 1px 2px 0 rgba(14, 16, 21, 0.04)',
        'elevated': '0 10px 30px -6px rgba(14, 16, 21, 0.07), 0 2px 6px -1px rgba(14, 16, 21, 0.03)',
        'card': '0 2px 8px -2px rgba(14, 16, 21, 0.05), 0 1px 2px -1px rgba(14, 16, 21, 0.03)',
        'modal': '0 25px 50px -12px rgba(14, 16, 21, 0.25)',
      },
      animation: {
        'pulse-subtle': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
    },
  },
  plugins: [],
};
