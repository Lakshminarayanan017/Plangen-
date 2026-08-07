import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  // Assets are served by FastAPI from /static/, which maps to frontend/dist.
  base: "/static/",
  build: {
    outDir: "dist",
    emptyOutDir: true,
    // The 3D chunk is lazy-loaded and is legitimately large; everything else
    // must stay small, so warn at a threshold that would catch regressions in
    // the app code rather than firing on three.js every build.
    chunkSizeWarningLimit: 700,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes("node_modules")) {
            if (id.includes("three") || id.includes("@react-three")) return "three";
            if (id.includes("gsap")) return "gsap";
            if (id.includes("framer-motion")) return "motion";
            return "vendor";
          }
        },
      },
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://127.0.0.1:8000", changeOrigin: true },
    },
  },
});
