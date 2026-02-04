/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'poker-felt': '#0f5132',
        'poker-felt-dark': '#0a3721',
        'poker-rail': '#8b4513',
        'poker-chip': '#ffd700',
      },
      animation: {
        'deal-card': 'dealCard 0.3s ease-out',
        'chip-move': 'chipMove 0.5s ease-in-out',
      },
      keyframes: {
        dealCard: {
          '0%': { transform: 'translateY(-100px) scale(0.5)', opacity: '0' },
          '100%': { transform: 'translateY(0) scale(1)', opacity: '1' },
        },
        chipMove: {
          '0%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-20px)' },
          '100%': { transform: 'translateY(0)' },
        },
      },
    },
  },
  plugins: [],
}
