# Dashboard Frontend

This folder holds the browser UI for the local workflow site.

- `html/dashboard/index.html`: HTML shell that loads the React app and receives backend state.
- `static/js/dashboard_app.js`: React components for the tabbed dashboard pages.
- `static/css/dashboard.css`: visual styling for the dashboard.
- `vendor/react.production.min.js`: local React browser runtime.
- `vendor/react-dom.production.min.js`: local ReactDOM browser runtime.

The Python backend still owns the workflow logic, file scans, queue state, and POST routes. It sends JSON page state into the browser, and this React app renders the tabs, forms, tables, progress cards, and dialogs from that state.

There is no separate dev server or build step. The backend serves these files under `/assets/...`.

Quick check:

```bash
npm run check --prefix frontend
```
