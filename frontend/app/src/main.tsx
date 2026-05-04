import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./styles.css";

const stateNode = document.getElementById("dashboard-state");
const initialState = stateNode ? JSON.parse(stateNode.textContent || "{}") : {};
declare global {
  interface Window {
    __DASHBOARD_STATE__?: typeof initialState;
  }
}
window.__DASHBOARD_STATE__ = initialState;

const rootEl = document.getElementById("dashboard-root");
if (rootEl) {
  createRoot(rootEl).render(
    <StrictMode>
      <App initialState={initialState} />
    </StrictMode>
  );
}
