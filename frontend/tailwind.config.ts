import type { Config } from "tailwindcss"

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ["var(--font-sora)", "sans-serif"],
        mono: ["var(--font-fira)", "monospace"],
      },
      colors: {
        bg: "#f8fafc",
        surface: "#ffffff",
        "surface-2": "#f1f5f9",
        "surface-3": "#e2e8f0",
        border: "#cbd5e1",
        "border-2": "#94a3b8",
        text: "#0f172a",
        "text-2": "#334155",
        "text-3": "#64748b",
        accent: "#0f766e",
        "accent-dim": "#dbeafe",
        amber: "#d97706",
        "amber-dim": "#fef3c7",
        blue: "#2563eb",
        "blue-dim": "#dbeafe",
        red: "#dc2626",
        "red-dim": "#fee2e2",
      },
      animation: {
        "fade-in": "fadeIn 0.3s ease-out",
        "slide-up": "slideUp 0.3s ease-out",
        "pulse-dot": "pulseDot 1.4s ease-in-out infinite",
      },
      keyframes: {
        fadeIn: { from: { opacity: "0" }, to: { opacity: "1" } },
        slideUp: { from: { opacity: "0", transform: "translateY(8px)" }, to: { opacity: "1", transform: "translateY(0)" } },
        pulseDot: { "0%, 80%, 100%": { transform: "scale(0.6)", opacity: "0.4" }, "40%": { transform: "scale(1)", opacity: "1" } },
      },
    },
  },
  plugins: [],
}
export default config
