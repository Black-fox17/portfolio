/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: {
        sans: ['Plus Jakarta Sans', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'SFMono-Regular', 'Menlo', 'monospace'],
        editorial: ['Plus Jakarta Sans', '-apple-system', 'sans-serif'],
        display: ['Plus Jakarta Sans', '-apple-system', 'sans-serif'],
      },
      colors: {
        paper: {
          DEFAULT: '#FFFFFF',
          50: '#FAFAFA',
          100: '#F5F5F5',
          200: '#EBEBEB',
          300: '#E0E0E0',
        },
        ink: {
          DEFAULT: '#171717',
          light: '#262626',
          muted: '#525252',
          faint: '#737373',
        },
        border: {
          DEFAULT: '#E5E7EB',
          subtle: '#F0F0F0',
          strong: '#D4D4D8',
        },
        accent: {
          DEFAULT: '#171717',
          cobalt: '#171717',
          emerald: '#525252',
        },
      },
      boxShadow: {
        'subtle': '0 1px 2px 0 rgba(0, 0, 0, 0.03)',
        'elevated': '0 8px 24px -4px rgba(0, 0, 0, 0.05), 0 2px 6px -1px rgba(0, 0, 0, 0.02)',
        'card': '0 1px 4px 0 rgba(0, 0, 0, 0.04)',
        'modal': '0 20px 40px -15px rgba(0, 0, 0, 0.1)',
      },
    },
  },
  plugins: [],
};
