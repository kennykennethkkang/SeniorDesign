# Vite + React + TypeScript dashboard (Phase 6 scaffold)

This directory hosts the modernized dashboard frontend. The legacy bundle at
`frontend/static/js/dashboard_app.js` still ships with every page; the Vite
build is a parallel pipeline that produces `frontend/static/dist/dashboard.js`
which the Python WSGI server serves as a static asset.

## Status

- ✅ Pipeline scaffolded (Vite, React 18, TypeScript strict, JSX)
- ✅ Two components ported as types-tightened reference (`ClusterQueuePanel`,
      `FileSearchPanel`)
- ⬜ Per-page migration (Overview, MediaLibrary, YouTube, Diarization,
      FineTuning, TrainingLabels)
- ⬜ Cut-over from legacy bundle in the Python rendering layer

## One-time install

```bash
cd frontend/app
npm install
```

`node_modules/` is gitignored. CI / fresh checkouts must run `npm install`
before the first build.

## Build

```bash
cd frontend/app
npm run build
```

Outputs:
- `frontend/static/dist/dashboard.js`
- `frontend/static/dist/dashboard.js.map`
- `frontend/static/dist/chunks/*.js` (lazy-loaded routes)
- `frontend/static/dist/assets/*` (CSS, images, fonts)

## Dev server

```bash
# Terminal A — Python backend
python3 workflow_cli.py local-web start

# Terminal B — Vite dev server (proxies /api /files /actions to the Python server)
cd frontend/app
npm run dev
```

The dev server listens on `5173` by default; the Python server on `8000`. Visit
`http://127.0.0.1:5173/` for HMR; the Python-served pages continue to use the
legacy bundle until a page is explicitly migrated.

## How to migrate a page

1. Pick a tab in `frontend/static/js/dashboard_app.js` (e.g., `OverviewPage`).
2. Create `frontend/app/src/pages/Overview.tsx`. Copy the structure, swap
   `React.createElement(...)` for JSX, type the props using the types in
   `src/types.ts`.
3. If the page needs new types, add them to `src/types.ts` so other migrated
   pages can share them.
4. Wire the new page into `App.tsx`'s router (currently a single demo
   page — the proper router lands as part of the second migration).
5. Run `npm run typecheck` then `npm run build`. Open the dashboard and
   verify the page works against real backend data.

## Cut-over plan

When all six pages are migrated:

1. Update `dashboard/app/rendering.py::frontend_state` to point the
   `assets.app` URL at `static/dist/dashboard.js` (and drop the
   `assets.react` / `assets.reactDom` script tags — they're bundled now).
2. Move `frontend/static/js/dashboard_app.js` and the `frontend/vendor/`
   React UMD bundles into `archive/frontend_legacy_<date>/`.
3. Update `frontend/package.json::scripts.check` to call
   `cd app && npm run typecheck`.

## Why a parallel bundle?

The user explicitly asked for "no logic changes." Running both bundles in
parallel until each page is verified means we can roll back any migration
by reverting one file. The Python serving layer never has to switch modes
mid-migration.
