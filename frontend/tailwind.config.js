/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'black': '#000000',
        'white': '#FFFFFF',
        'gray': {
          300: '#333333',
          500: '#555555',
          700: '#777777',
          900: '#999999',
        },
        'positive': '#00FF00', // Green for positive status
        'negative': '#FF0000', // Red for negative status
      },
      fontFamily: {
        'serif': ['"Source Serif 4"', 'Georgia', 'serif'],
        'sans': ['"DM Sans"', 'sans-serif'],
        'mono': ['"JetBrains Mono"', 'monospace'],
      },
      borderRadius: {
        'none': '0',
        'sm': '0',
        'md': '0',
        'lg': '0',
        'xl': '0',
        'full': '0',
      },
    },
  },
  plugins: [],
}