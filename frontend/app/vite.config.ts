import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

// The Python WSGI server serves static assets directly out of frontend/static/.
// We point Vite's build output there so no changes to the existing serving
// layer are required: workflow_dashboard renders <script src="static/dist/dashboard.js">
// and the file is just on disk after `npm run build`.
//
// During local development, `npm run dev` hosts a Vite dev server that talks to
// the Python backend via the same JSON state endpoints. The server-rendered
// HTML shell embeds a global `__DASHBOARD_STATE__` so the migrated React app
// can hydrate against the same payload as the legacy bundle.

const ROOT = path.resolve(__dirname);
const PYTHON_STATIC = path.resolve(ROOT, "..", "static");

export default defineConfig({
  plugins: [react()],
  root: ROOT,
  base: "/assets/static/dist/",
  build: {
    outDir: path.resolve(PYTHON_STATIC, "dist"),
    emptyOutDir: true,
    sourcemap: true,
    // Single-bundle deliberately - matches the existing dashboard_app.js
    // shipping pattern and keeps the Python server's <script> tag stable.
    rollupOptions: {
      input: path.resolve(ROOT, "src", "main.tsx"),
      output: {
        entryFileNames: "dashboard.js",
        chunkFileNames: "chunks/[name]-[hash].js",
        assetFileNames: "assets/[name]-[hash][extname]",
      },
    },
  },
  server: {
    port: 5173,
    strictPort: false,
    proxy: {
      // Forward API calls to the Python WSGI server during dev.
      "/api": "http://127.0.0.1:8000",
      "/files": "http://127.0.0.1:8000",
      "/actions": "http://127.0.0.1:8000",
    },
  },
});
