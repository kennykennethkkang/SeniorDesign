/* global React, ReactDOM */
/*
 * dashboard_app.js — senior-design dashboard frontend.
 *
 * One hand-written React bundle (no JSX, no build step) that renders every
 * page of the local dashboard. Boots from a JSON blob the Python backend
 * stamps into <script id="dashboard-state">, then talks to the same Python
 * server over plain form posts and JSON endpoints — no separate API tier.
 *
 * Why one file: the app is small enough that the cost of a build pipeline
 * outweighed the cleanliness of splitting per-page. A Vite/TS scaffold
 * exists at frontend/app/ but is intentionally not live yet — see
 * frontend/app/README.md for the cut-over plan.
 *
 * Structure (top → bottom):
 *   1. Module-scope constants, theme handling, IndexedDB audio cache.
 *   2. Small primitives (h, formatters, fetch helpers, polling).
 *   3. Per-page renderers (Overview, MediaLibrary, YouTube, Diarization,
 *      Stitching, TrainingLabels, FineTuning).
 *   4. Shell components (tab nav, hero card, command card, mount + boot).
 *
 * If you're adding a feature: state lives in module-level `state` (the
 * server-stamped blob, refreshed by polling) plus the per-page `uiState`
 * for ephemeral UI toggles. Keep behavior server-driven where possible.
 */
(function () {
  "use strict";

  const h = React.createElement;
  const initialState = JSON.parse(document.getElementById("dashboard-state").textContent);
  const APP_TITLE = "ML Speech Diarization";
  const LIVE_TRACKING_INTERVAL_MS = 1000;
  const IDLE_TRACKING_INTERVAL_MS = 5000;
  const ACTIVE_RUN_STATUSES = ["running", "submitted", "pending", "waiting", "configuring"];
  const THEME_STORAGE_KEY = "ml-speech-diarization-theme";
  const AUDIO_CACHE_DB_NAME = "ml-speech-audio-cache";
  const AUDIO_CACHE_STORE = "blobs";
  const AUDIO_CACHE_DB_VERSION = 1;
  const AUDIO_CACHE_ALL_CONCURRENCY = 2;
  const DEFAULT_FILE_VIEW_LIMIT = 250;
  const FILE_VIEW_LIMIT_OPTIONS = [100, 250, 500, 1000];
  const root = ReactDOM.createRoot(document.getElementById("dashboard-root"));
  let state = initialState;
  let ctx = state.context || {};
  let routes = state.routes || {};
  let defaults = state.defaults || {};
  let currentPage = null;
  let navItems = [];
  let refreshTimer = null;
  let trackingStopped = false;
  let currentFingerprint = "";
  const uiState = {
    uploadsModelFilter: "all",
    showCompletedTrainingLabels: false,
  };

  function safeStoredTheme() {
    try {
      const stored = window.localStorage.getItem(THEME_STORAGE_KEY);
      return stored === "dark" || stored === "light" ? stored : "";
    } catch (_error) {
      return "";
    }
  }

  function preferredTheme() {
    const stored = safeStoredTheme();
    if (stored) {
      return stored;
    }
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  function applyTheme(theme, persist = true) {
    const normalized = theme === "dark" ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", normalized);
    if (!persist) {
      return normalized;
    }
    try {
      window.localStorage.setItem(THEME_STORAGE_KEY, normalized);
    } catch (_error) {
      // Theme persistence is optional; the UI still updates for this page.
    }
    return normalized;
  }

  applyTheme(preferredTheme(), false);
  const PAGE_CONFIG = {
    "/": {
      label: "Overview",
      title: APP_TITLE,
      summary: "Recommended step, workspace counts, and recent activity.",
      intro: "Start here to see what is ready, what ran last, and where to go next.",
    },
    "/uploads": {
      label: "Media Library",
      title: "Media Library",
      summary: "Source audio, diarized outputs, reviews, flags, and logs.",
      intro: "Manage source audio and inspect the outputs each file has produced.",
    },
    "/stitching": {
      label: "Audio Stitching",
      title: "Audio Stitching",
      summary: "Randomized stitched WAVs, RTTM timing, speaker assignment, and training export.",
      intro: "Build a randomized stitched training sample from selected clips and verify the generated timing artifacts.",
    },
    "/training-labels": {
      label: "Training Labels",
      title: "Training Labels",
      summary: "Draft labels, completed samples, speakers, and project targets.",
      intro: "Turn uploaded audio into training-ready labels and send completed samples to projects.",
    },
    "/youtube": {
      label: "YouTube Audio Conversion",
      title: "YouTube Audio Conversion",
      summary: "URL queue, conversion controls, run status, and queue issues.",
      intro: "Add video links, convert them to audio, and track which links are ready or already done.",
    },
    "/diarization": {
      label: "Diarization",
      title: "Diarization",
      summary: "Model settings, file selection, queue status, and output links.",
      intro: "Choose a model, select audio, and submit diarization runs with clear status tracking.",
    },
    "/fine-tuning": {
      label: "Fine-Tuning",
      title: "Fine-Tuning",
      summary: "Training samples, project metrics, preparation, and launch controls.",
      intro: "Build diarization training projects from labeled samples, then prepare and launch runs.",
    },
  };

  function pageConfigFor(path) {
    return PAGE_CONFIG[path] || PAGE_CONFIG["/"];
  }

  function navItemsForState(currentState) {
    return (currentState.navItems || []).map((item) => ({
      ...item,
      label: PAGE_CONFIG[item.path]?.label || item.path,
    }));
  }

  function syncDerivedState(nextState) {
    state = nextState || {};
    ctx = state.context || {};
    routes = state.routes || {};
    defaults = state.defaults || {};
    currentPage = pageConfigFor(state.currentPath);
    navItems = navItemsForState(state);
    currentFingerprint = text((ctx.tracking || {}).fingerprint || "");
    document.title = currentPage.title === APP_TITLE ? APP_TITLE : `${currentPage.title} | ${APP_TITLE}`;
  }

  function text(value, fallback = "") {
    if (value === null || value === undefined) {
      return fallback;
    }
    return String(value);
  }

  syncDerivedState(initialState);

  function classNames() {
    return Array.from(arguments).filter(Boolean).join(" ");
  }

  // Tiny helper so rename buttons across the app can fire a single hidden
  // POST without each caller re-implementing the same plumbing. The dashboard
  // is happy with full-page redirects after the post, so we just submit a
  // throwaway form rather than wiring a fetch + reload dance.
  // We need to trigger a POST from JavaScript without building a full SPA form
  // submission flow — a throwaway hidden form is the simplest correct way to do it.
  function submitHiddenForm(action, fields) {
    if (!action) {
      return;
    }
    const form = document.createElement("form");
    form.method = "POST";
    form.action = action;
    Object.entries(fields || {}).forEach(([name, value]) => {
      const input = document.createElement("input");
      input.type = "hidden";
      input.name = name;
      input.value = value == null ? "" : String(value);
      form.appendChild(input);
    });
    document.body.appendChild(form);
    form.submit();
  }

  function promptRenameProject(project) {
    if (!project || !project.slug) return;
    const current = project.displayName || project.slug;
    // window.prompt is exactly the right amount of UI for a single-field rename —
    // no modal lib, no extra state, and it works without any event wiring.
    const next = window.prompt(`Rename project "${current}"`, current);
    if (next === null) return;
    const cleaned = String(next).trim();
    if (!cleaned || cleaned === current) return;
    submitHiddenForm(routes.fineTuneRenameProject, {
      project_slug: project.slug,
      backend: project.backend,
      display_name: cleaned,
    });
  }

  function promptRenameRun(run) {
    if (!run || !run.runDir) return;
    const current = run.displayName || run.versionName || run.runName || "this run";
    const next = window.prompt(`Rename run "${current}"`, current);
    if (next === null) return;
    const cleaned = String(next).trim();
    if (!cleaned || cleaned === current) return;
    submitHiddenForm(routes.fineTuneRenameRun, {
      run_dir: run.runDir,
      display_name: cleaned,
    });
  }

  function confirmDeleteFineTunedModel(run) {
    if (!run || !run.runDir) return;
    const current = run.displayName || run.versionName || run.runName || "this fine-tuned model";
    const confirmed = window.confirm(
      `Delete the fine-tuned model artifacts for "${current}"?\n\nThis removes the checkpoint/model files from the model picker. The run record and Slurm logs are kept.`
    );
    if (!confirmed) return;
    submitHiddenForm(routes.fineTuneDeleteRunModel, {
      run_dir: run.runDir,
    });
  }

  function promptRenameStitch(row) {
    if (!row || !row.runDir) return;
    const current = row.displayName || row.outputName || row.name || "stitched output";
    const next = window.prompt(`Rename stitched output "${current}"`, current);
    if (next === null) return;
    const cleaned = String(next).trim();
    if (!cleaned || cleaned === current) return;
    submitHiddenForm(routes.renameStitching, {
      run_dir: row.runDir,
      display_name: cleaned,
    });
  }

  function confirmDeleteStitch(row) {
    if (!row || !row.runDir || !routes.deleteStitching) return;
    const label = row.displayName || row.outputName || row.name || "this stitched output";
    const trainingNote = (row.trainingUsage || []).length
      ? `\n\nThis will also remove the matching audio + RTTM copies from: ${(row.trainingUsage || [])
          .map((entry) => entry.project_key || entry.project_name || "")
          .filter(Boolean)
          .join(", ")}.`
      : "";
    const confirmed = window.confirm(`Delete the stitched run "${label}"?${trainingNote}\n\nThe run folder under stitched/ will be removed and cannot be recovered.`);
    if (!confirmed) return;
    submitHiddenForm(routes.deleteStitching, { run_dir: row.runDir });
  }

  function StatusPill({ status }) {
    const normalized = text(status || "unknown").toLowerCase().replace(/[_\s]+/g, "-");
    return h("span", { className: `status-pill ${normalized}` }, text(status || "unknown").replace(/_/g, " "));
  }

  function Notice({ message, status }) {
    const [visible, setVisible] = React.useState(true);
    const lines = React.useMemo(
      () => text(message).split(/\r?\n/).map((line) => line.trim()).filter(Boolean),
      [message]
    );
    React.useEffect(() => {
      if (!message) {
        return undefined;
      }
      // Strip ?status=...&message=... from the URL once the banner is on screen.
      // The redirect carried the flash payload here, but leaving it in the
      // address bar makes refreshes re-show stale notifications and bloats
      // shareable URLs. history.replaceState removes it without navigating.
      try {
        const url = new URL(window.location.href);
        if (url.searchParams.has("status") || url.searchParams.has("message")) {
          url.searchParams.delete("status");
          url.searchParams.delete("message");
          const qs = url.searchParams.toString();
          window.history.replaceState({}, "", url.pathname + (qs ? `?${qs}` : "") + url.hash);
        }
      } catch (err) {
        // Older browsers without URL/history.replaceState fall through silently.
      }
      setVisible(true);
      const timer = window.setTimeout(() => setVisible(false), 6500);
      return () => window.clearTimeout(timer);
    }, [message]);
    if (!message || !lines.length || !visible) {
      return null;
    }
    const safeStatus = ["info", "success", "error"].includes(status) ? status : "info";
    const labelMap = {
      success: "Success",
      error: "Action Failed",
      info: "Notice",
    };
    return h(
      "div",
      { className: `notice notice-toast ${safeStatus}`, role: "status", "aria-live": "polite" },
      h("p", { className: "notice-label" }, labelMap[safeStatus]),
      h("p", { className: "notice-summary" }, lines[0]),
      lines.length > 1
        ? h(
            "ul",
            { className: "notice-list" },
            lines.slice(1).map((line, index) => h("li", { key: index }, line))
          )
        : null,
      h(
        "button",
        {
          type: "button",
          className: "notice-dismiss",
          "aria-label": "Dismiss notification",
          onClick: () => setVisible(false),
        },
        "×"
      )
    );
  }

  function QuickJump() {
    const [query, setQuery] = React.useState("");
    const allActions = navItems.map((item) => ({
      href: item.href,
      label: item.label,
      detail: PAGE_CONFIG[item.path]?.summary || "Open page",
    }));
    const normalizedQuery = query.trim().toLowerCase();
    const results = allActions
      .filter((item) => {
        if (!normalizedQuery) {
          return true;
        }
        return `${item.label} ${item.detail}`.toLowerCase().includes(normalizedQuery);
      })
      .slice(0, 4);
    return h(
      "div",
      { className: "quick-jump" },
      h("div", null, h("h2", null, "Find a page")),
      h("input", {
        type: "search",
        value: query,
        placeholder: "Search labels, runs, outputs...",
        onChange: (event) => setQuery(event.target.value),
        "aria-label": "Search workflow pages",
      }),
      h(
        "div",
        { className: "quick-jump-results" },
        results.length
          ? results.map((item) =>
              h("a", { key: item.href, href: item.href }, h("strong", null, item.label), h("small", null, item.detail))
            )
          : h("p", { className: "row-note" }, "No matching page.")
      )
    );
  }

  function WorkspaceStrip() {
    const tracking = ctx.tracking || {};
    const activeCount = Number(tracking.activeRunCount || 0);
    const items = [
      ["Audio Library", `${ctx.audioCount || 0} file(s)`],
      ["YouTube Queue", `${ctx.queueCount || 0} link(s)`],
      ["Fine-Tuning", `${ctx.projectCount || 0} project(s)`],
      ["Live Sync", activeCount ? `${activeCount} active` : "watching"],
    ];
    return h(
      "div",
      { className: "workspace-strip" },
      items.map(([label, value]) => h("article", { key: label }, h("span", null, label), h("strong", null, value)))
    );
  }

  function ProcessSteps({ steps, tone }) {
    return h(
      "div",
      { className: classNames("process-steps", tone && `process-steps-${tone}`) },
      (steps || []).map((step, index) =>
        h("span", { key: step }, h("strong", null, index + 1), step)
      )
    );
  }

  function Field({ id, label, children, note }) {
    return h(
      "div",
      null,
      label ? h("label", { htmlFor: id }, label) : null,
      children,
      note ? h("p", { className: "footer-note" }, note) : null
    );
  }

  function TextInput({ id, name, defaultValue, placeholder, type = "text", extra }) {
    return h("input", {
      id,
      name: name || id,
      type,
      defaultValue: defaultValue === undefined ? "" : defaultValue,
      placeholder,
      ...(extra || {}),
    });
  }

  function SelectInput({ id, name, defaultValue, children, extra }) {
    return h("select", { id, name: name || id, defaultValue, ...(extra || {}) }, children);
  }

  function Option({ value, children, extra }) {
    return h("option", { value, ...(extra || {}) }, children);
  }

  function audioFolderOptions() {
    const rows = ctx.audioFolders || [];
    if (rows.length) {
      return rows;
    }
    return [
      { value: defaults.rootAudioFolderValue || "__root__", name: "Unsorted Root", path: "audio_in", fileCount: 0 },
      { value: defaults.defaultUploadAudioFolder || "file_uploads", name: "File Uploads", path: "audio_in/file_uploads", fileCount: 0 },
      { value: defaults.defaultYoutubeAudioFolder || "youtube_links", name: "Youtube Links", path: "audio_in/youtube_links", fileCount: 0 },
    ];
  }

  function AudioFolderSelect({ id, name, defaultValue }) {
    const rows = audioFolderOptions();
    return h(
      "select",
      { id, name: name || id, defaultValue },
      rows.map((folder) =>
        h(
          Option,
          { key: folder.value, value: folder.value },
          `${folder.name || folder.value} (${folder.fileCount || 0})`
        )
      )
    );
  }

  function fileViewFolderLabel(row) {
    const explicit = text(row?.folder || row?.audioFolder || row?.folderName || "").trim();
    if (explicit) {
      return explicit;
    }
    // Pick the source with the most folder context. Workspace-file rows
    // have ``name`` set to the basename only and ``path`` set to the full
    // workspace-relative path — preferring the one that actually contains a
    // slash keeps the folder filter meaningful for those pickers without
    // breaking the table rows whose ``name`` is already a relative path.
    const candidates = [row?.name, row?.audioFile, row?.path, row?.fileName]
      .map((value) => text(value).replace(/\\/g, "/").trim())
      .filter(Boolean);
    const candidate = candidates.find((value) => value.includes("/")) || candidates[0] || "";
    if (!candidate.includes("/")) {
      return "Unsorted Root";
    }
    const parts = candidate.split("/").filter(Boolean);
    if (!parts.length) {
      return "Unsorted Root";
    }
    if (parts[0] === "audio_in") {
      // For nested sets like ``audio_in/JIBOKids/letters_digits/<file>``
      // group by ``<set>/<sub>`` so the picker can drill into individual
      // tasks. Flat sets (audio_in/<set>/<file>) keep their existing
      // ``<set>`` grouping.
      if (parts.length >= 5) {
        return `${parts[1]}/${parts[2]}`;
      }
      return parts.length > 2 ? parts[1] : "Unsorted Root";
    }
    // Group RTTM/audio/text files inside fine_tuning/projects/<backend>/<slug>/...
    // by the project itself (``<backend>/<slug>``), which is the level the
    // user actually thinks in. The old behavior dumped every project's
    // files into a single ``fine_tuning/projects`` bucket.
    if (parts[0] === "fine_tuning" && parts[1] === "projects" && parts.length > 4) {
      return `${parts[2]}/${parts[3]}`;
    }
    if (parts[0] === "fine_tuning" && parts.length > 1) {
      return parts.slice(0, Math.min(2, parts.length - 1)).join("/") || "fine_tuning";
    }
    if (parts[0] === "outputs" && parts.length > 1) {
      return parts.slice(0, Math.min(4, parts.length - 1)).join("/") || "outputs";
    }
    return parts.length > 1 ? parts[0] : "Unsorted Root";
  }

  function fileViewDisplayName(row) {
    return text(row?.fileName || row?.audioFile || row?.name || row?.path || "file");
  }

  function compareFileViewLabels(left, right) {
    if (left === "Unsorted Root" && right !== "Unsorted Root") return -1;
    if (right === "Unsorted Root" && left !== "Unsorted Root") return 1;
    return text(left).toLowerCase().localeCompare(text(right).toLowerCase());
  }

  function sortFileRowsByFolder(rows, folderGetter = fileViewFolderLabel, nameGetter = fileViewDisplayName) {
    return [...(rows || [])].sort((left, right) => {
      const folderCompare = compareFileViewLabels(folderGetter(left), folderGetter(right));
      if (folderCompare !== 0) {
        return folderCompare;
      }
      return text(nameGetter(left)).toLowerCase().localeCompare(text(nameGetter(right)).toLowerCase());
    });
  }

  function fileViewFolderChoices(rows, folderGetter = fileViewFolderLabel) {
    return Array.from(
      (rows || []).reduce((choices, row) => {
        const key = folderGetter(row);
        const current = choices.get(key) || { key, label: key, count: 0 };
        current.count += 1;
        choices.set(key, current);
        return choices;
      }, new Map()).values()
    ).sort((left, right) => compareFileViewLabels(left.key, right.key));
  }

  function clampFileViewLimit(value, total = Number.MAX_SAFE_INTEGER) {
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) {
      return DEFAULT_FILE_VIEW_LIMIT;
    }
    const max = Math.max(1, Number(total) || 1);
    return Math.min(Math.max(1, Math.floor(parsed)), max);
  }

  function FileViewLimitControl({ id, total, shown, limit, onLimitChange, noun = "file" }) {
    const totalCount = Number(total || 0);
    const shownCount = Number(shown || 0);
    const canShowMore = shownCount < totalCount;
    const displayedLimit = totalCount === 0 ? 0 : Math.min(limit, totalCount);
    const options = FILE_VIEW_LIMIT_OPTIONS.filter((value, index, values) => value <= totalCount && values.indexOf(value) === index);
    if (totalCount === 0) {
      options.push(0);
    } else if (!options.includes(displayedLimit)) {
      options.push(displayedLimit);
      options.sort((left, right) => left - right);
    }
    return h(
      "div",
      { className: "file-view-limit-control" },
      h("span", { className: "file-view-count" }, `${shownCount} shown of ${totalCount} ${noun}${totalCount === 1 ? "" : "s"}`),
      h(
        "label",
        { className: "compact-label", htmlFor: id },
        "Rows",
        h(
          "select",
          {
            id,
            className: "compact-control",
            value: String(displayedLimit),
            disabled: totalCount === 0,
            onChange: (event) => onLimitChange(clampFileViewLimit(event.target.value, totalCount)),
          },
          options.map((value) => h("option", { key: value, value: String(value) }, value)),
          totalCount > Math.max(...options, 0) ? h("option", { value: String(totalCount) }, "All") : null
        )
      ),
      canShowMore
        ? h(
            "button",
            {
              type: "button",
              className: "ghost btn-sm",
              onClick: () => onLimitChange(Math.min(totalCount, limit + DEFAULT_FILE_VIEW_LIMIT)),
            },
            "Show More"
          )
        : null
    );
  }

  function FolderFilterControl({ id, value, onChange, folders }) {
    return h(
      "label",
      { className: "compact-label file-view-folder-filter", htmlFor: id },
      "Folder",
      h(
        "select",
        {
          id,
          className: "compact-control",
          value,
          disabled: !(folders || []).length,
          onChange: (event) => onChange(event.target.value),
        },
        h("option", { value: "all" }, "All folders"),
        (folders || []).map((folder) =>
          h("option", { key: folder.key, value: folder.key }, `${folder.label} (${folder.count})`)
        )
      )
    );
  }

  function diarizationBackendOptions() {
    const configured = defaults.diarizationBackends || [];
    const backends = configured.length ? configured : [{ value: "nemo", label: "NeMo" }, { value: "pyannote", label: "pyannote" }];
    return backends.map((backend) => {
      const value = text(backend.value || backend);
      return h(Option, { key: value, value }, backend.label || value);
    });
  }

  function backendDisplayName(value) {
    const normalized = text(value).toLowerCase();
    if (normalized === "both" || normalized === "nemo+pyannote" || normalized === "pyannote+nemo") {
      return "NeMo + pyannote";
    }
    if (normalized === "nemo") {
      return "NeMo";
    }
    if (normalized === "pyannote") {
      return "pyannote";
    }
    return value || "NeMo + pyannote";
  }

  function diarizationModelOptions() {
    const configured = (ctx.diarization && ctx.diarization.modelOptions) || [];
    if (configured.length) {
      return configured;
    }
    const backends = defaults.diarizationBackends || [];
    if (!backends.length) {
      return [
        { key: "nemo", backend: "nemo", label: "NeMo (default)", shortLabel: "NeMo", kind: "default", description: "" },
        { key: "pyannote", backend: "pyannote", label: "pyannote (default)", shortLabel: "pyannote", kind: "default", description: "" },
      ];
    }
    return backends.map((backend) => {
      const value = text(backend.value || backend);
      return {
        key: value,
        backend: value,
        label: `${backend.label || value} (default)`,
        shortLabel: backend.label || value,
        kind: "default",
        description: "",
      };
    });
  }

  function diarizationModelChoiceOptions() {
    return diarizationModelOptions().map((model) =>
      h(Option, { key: model.key, value: model.key }, model.label || model.key)
    );
  }

  function diarizationHistoryFilterOptions(rows) {
    // Filter dropdown shown above the diarized-files table on /uploads.
    // Entries:
    //   - "All diarized items" (always)
    //   - One per backend that produced output ("nemo runs", "pyannote runs")
    //   - One per fine-tuned model key (only appears once a fine-tuned model
    //     has actually run, by design).
    // We deliberately do NOT add per-batch entries: batch ids are run-grouping
    // metadata, not user-facing model identities, and were surfacing as
    // "Batch 20260503" rows that confused the filter.
    const options = [{ value: "all", label: "All diarized items" }];
    const seenBackends = new Set();
    const seenModels = new Set();
    (rows || []).forEach((row) => {
      const backend = text(row.backend || "").trim();
      if (backend && !seenBackends.has(backend)) {
        seenBackends.add(backend);
        options.push({
          value: `backend:${backend}`,
          label: `${row.backendLabel || backend} runs`,
        });
      }
      const modelKey = text(row.modelKey || "").trim();
      if (
        modelKey &&
        modelKey !== backend &&
        !seenModels.has(modelKey)
      ) {
        seenModels.add(modelKey);
        options.push({
          value: modelKey,
          label: row.modelLabel || modelKey,
        });
      }
    });
    return options;
  }

  function linkWithLabel(links, label) {
    const normalized = text(label).toLowerCase();
    return (links || []).find((link) => text(link.label).toLowerCase() === normalized) || null;
  }

  function modelStatusForComparison(row) {
    if (!row) {
      return "not run";
    }
    return row.status || "unknown";
  }

  function comparisonItemFromModel(option, row) {
    const safeOption = option || {};
    const safeRow = row || {};
    const key = text(safeRow.modelKey || safeOption.key || safeRow.backend || safeOption.backend || "model");
    const backend = text(safeRow.backend || safeOption.backend || "");
    const links = safeRow.links || [];
    const srtLink = linkWithLabel(links, "Diarized Times");
    const reviewLink = linkWithLabel(links, "Review Page");
    const transcriptLink = linkWithLabel(links, "Transcript");
    return {
      key,
      backend,
      label: safeRow.modelLabel || safeOption.label || key,
      shortLabel: safeOption.shortLabel || safeRow.modelLabel || safeOption.label || key,
      kind: safeOption.kind || "historical",
      description: safeOption.description || "",
      status: modelStatusForComparison(row),
      hasRecord: Boolean(row),
      runName: safeRow.runName || "",
      lastRun: safeRow.lastRun || "",
      runtimeSeconds: safeRow.runtimeSeconds || "",
      errorSummary: safeRow.errorSummary || "",
      links,
      srtHref: srtLink?.href || "",
      reviewHref: reviewLink?.href || "",
      transcriptHref: transcriptLink?.href || "",
      labelSegments: safeRow.labelSegments || "",
      transcriptPreview: safeRow.transcriptPreview || "",
      segmentCount: safeRow.segmentCount || "",
      previewSegmentCount: safeRow.previewSegmentCount || "",
      speakerCount: safeRow.speakerCount || "",
      durationSeconds: safeRow.durationSeconds || "",
      previewTruncated: Boolean(safeRow.previewTruncated),
    };
  }

  function diarizationComparisonItems(audioName) {
    const targetName = text(audioName).trim();
    if (!targetName) {
      return [];
    }
    const history = (ctx.diarization && ctx.diarization.history) || [];
    const recordsByModel = new Map();
    history
      .filter((row) => text(row.audioFile).trim() === targetName)
      .forEach((row) => {
        const key = text(row.modelKey || row.backend || "").trim();
        if (key && !recordsByModel.has(key)) {
          recordsByModel.set(key, row);
        }
      });
    const options = diarizationModelOptions();
    const usedKeys = new Set();
    const items = options.map((option) => {
      usedKeys.add(option.key);
      return comparisonItemFromModel(option, recordsByModel.get(option.key));
    });
    recordsByModel.forEach((row, key) => {
      if (!usedKeys.has(key)) {
        items.push(comparisonItemFromModel(null, row));
      }
    });
    return items;
  }

  function defaultComparisonModelKey(items) {
    const withReview = (items || []).find((item) => item.hasRecord && item.reviewHref);
    if (withReview) {
      return withReview.key;
    }
    const withRecord = (items || []).find((item) => item.hasRecord);
    if (withRecord) {
      return withRecord.key;
    }
    return (items && items[0] && items[0].key) || "";
  }

  function DiarizationComparison({ row, selectedModelKey, onSelectedModelKeyChange, onUseSegments, onUseTranscript, comparisonId }) {
    const audioName = row?.name || row?.audioFile || "";
    const items = diarizationComparisonItems(audioName);
    const itemKeys = items.map((item) => item.key).join("|");
    const effectiveKey = items.some((item) => item.key === selectedModelKey)
      ? selectedModelKey
      : defaultComparisonModelKey(items);
    const activeItem = items.find((item) => item.key === effectiveKey) || items[0] || null;
    const safeId = dialogIdFor(comparisonId || "diarization-comparison", audioName || "audio");
    const controlName = `${safeId}-model`;

    React.useEffect(() => {
      if (effectiveKey && effectiveKey !== selectedModelKey && typeof onSelectedModelKeyChange === "function") {
        onSelectedModelKeyChange(effectiveKey);
      }
    }, [audioName, itemKeys, effectiveKey, selectedModelKey]);

    if (!audioName) {
      return h("p", { className: "empty" }, "Select an audio file to compare diarization results.");
    }
    if (!items.length) {
      return h("p", { className: "empty" }, "No diarization model profiles are available yet.");
    }

    return h(
      "section",
      { className: "model-comparison", "aria-labelledby": `${safeId}-heading` },
      h("div", { className: "panel-head" }, h("div", null, h("h3", { id: `${safeId}-heading` }, "Diarization Comparison"), h("p", { className: "row-note" }, audioName))),
      row?.audioHref ? h("div", { className: "audio-review compact-audio" }, h("audio", { controls: true, preload: "none", src: row.audioHref })) : null,
      h(
        "fieldset",
        { className: "model-toggle" },
        h("legend", null, "Diarization model"),
        items.map((item) =>
          h(
            "label",
            { key: item.key, className: classNames("model-choice", item.key === effectiveKey ? "active" : "") },
            h("input", {
              type: "radio",
              name: controlName,
              value: item.key,
              checked: item.key === effectiveKey,
              onChange: () => {
                if (typeof onSelectedModelKeyChange === "function") {
                  onSelectedModelKeyChange(item.key);
                }
              },
            }),
            h("span", null, item.shortLabel || item.label),
            h(StatusPill, { status: item.status })
          )
        )
      ),
      h(DataTable, {
        className: "compact-table model-compare-table",
        headers: ["Model", "Status", "Segments", "Speakers", "Run", "Review"],
        rows: items,
        emptyText: "No model results to compare.",
        renderRow: (item) =>
          h(
            "tr",
            { key: `${safeId}-row-${item.key}` },
            h("td", null, h("strong", null, item.label), item.backend ? h("p", { className: "row-note" }, backendDisplayName(item.backend)) : null),
            h("td", null, h(StatusPill, { status: item.status }), item.errorSummary ? h("p", { className: "row-note" }, item.errorSummary) : null),
            h("td", null, item.segmentCount || "n/a"),
            h("td", null, item.speakerCount || "n/a"),
            h("td", null, item.runName || "No run", item.lastRun ? h("p", { className: "row-note" }, item.lastRun) : null),
            h("td", null, item.reviewHref ? h("a", { href: item.reviewHref }, "Open review page") : h("span", { className: "row-note" }, "Not available"))
          ),
      }),
      activeItem
        ? h(
            "div",
            {
              className: "comparison-detail",
              role: "region",
              "aria-live": "polite",
              "aria-label": `Selected diarization result: ${activeItem.label}`,
            },
            h(
              "div",
              { className: "summary-grid" },
              h(SummaryCard, { title: "Selected Model" }, h("p", null, h("strong", null, activeItem.label)), h("p", { className: "row-note" }, activeItem.runName || "No completed run for this file.")),
              h(SummaryCard, { title: "Timing" }, h("p", null, h("strong", null, activeItem.segmentCount || 0), " segment(s)"), h("p", { className: "row-note" }, `${activeItem.speakerCount || 0} speaker(s), ${activeItem.durationSeconds ? formatDuration(activeItem.durationSeconds) : "duration unavailable"}`))
            ),
            activeItem.previewTruncated
              ? h("p", { className: "row-note" }, `Editable timing preview contains ${activeItem.previewSegmentCount} of ${activeItem.segmentCount} segment(s). Open the review page for the full file.`)
              : null,
            h(
              "div",
              { className: "button-row comparison-actions" },
              activeItem.labelSegments && typeof onUseSegments === "function"
                ? h("button", { className: "secondary", type: "button", onClick: () => onUseSegments(activeItem.labelSegments) }, activeItem.previewTruncated ? "Use Preview Timing" : "Use Model Timing")
                : null,
              activeItem.transcriptPreview && typeof onUseTranscript === "function"
                ? h("button", { className: "secondary", type: "button", onClick: () => onUseTranscript(activeItem.transcriptPreview) }, "Use Transcript Notes")
                : null,
              activeItem.reviewHref ? h("a", { className: "tab-link", href: activeItem.reviewHref }, "Open Review Page") : null,
              activeItem.srtHref ? h("a", { className: "tab-link", href: activeItem.srtHref }, "Open SRT") : null
            ),
            h(
              "details",
              { className: "details-box", open: true },
              h("summary", null, "Selected Model Artifacts"),
              h("div", null, h(ArtifactPreviewBrowser, { links: activeItem.links, preferredLabels: ["Review Page", "Diarized Times", "Transcript", "Review Flags", "Summary"], emptyText: "This model has no linked artifacts for the selected audio." }))
            )
          )
        : null
    );
  }

  function LabelSourcePicker({ row, selectedModelKey, onSelectedModelKeyChange, onUseSegments, onUseTranscript, pickerId }) {
    const audioName = row?.name || row?.audioFile || "";
    const items = diarizationComparisonItems(audioName);
    const itemKeys = items.map((item) => item.key).join("|");
    const effectiveKey = items.some((item) => item.key === selectedModelKey)
      ? selectedModelKey
      : defaultComparisonModelKey(items);
    const activeItem = items.find((item) => item.key === effectiveKey) || items[0] || null;
    const safeId = dialogIdFor(pickerId || "label-source", audioName || "audio");
    const controlName = `${safeId}-model`;

    React.useEffect(() => {
      if (effectiveKey && effectiveKey !== selectedModelKey && typeof onSelectedModelKeyChange === "function") {
        onSelectedModelKeyChange(effectiveKey);
      }
    }, [audioName, itemKeys, effectiveKey, selectedModelKey]);

    // Audio playback is rendered by the parent dialog (TrainingLabelDialog),
    // not here. Previously this picker returned null whenever there were no
    // diarization comparison items, which meant a freshly uploaded WAV that
    // had never been diarized lost its audio player too — exactly the moment
    // a user would need to listen to label it. Skip the model-comparison
    // section here when there is nothing to compare; the audio still plays.
    if (!audioName || !items.length) {
      return null;
    }

    return h(
      "section",
      { className: "label-source-picker", "aria-labelledby": `${safeId}-heading` },
      h("div", { className: "panel-head compact-head" }, h("div", null, h("h3", { id: `${safeId}-heading` }, "Start from an existing diarization"), h("p", { className: "row-note" }, "Pick a model's output as the starting point. Use Timing or Use Transcript to copy it into the fields below. You can still edit anything afterward."))),
      h(
        "fieldset",
        { className: "model-toggle compact-toggle" },
        h("legend", null, "Model"),
        items.map((item) =>
          h(
            "label",
            { key: item.key, className: classNames("model-choice", item.key === effectiveKey ? "active" : "") },
            h("input", {
              type: "radio",
              name: controlName,
              value: item.key,
              checked: item.key === effectiveKey,
              onChange: () => {
                if (typeof onSelectedModelKeyChange === "function") {
                  onSelectedModelKeyChange(item.key);
                }
              },
            }),
            h("span", null, item.shortLabel || item.label),
            h(StatusPill, { status: item.status })
          )
        )
      ),
      activeItem
        ? h(
            "div",
            { className: "source-summary", role: "region", "aria-live": "polite", "aria-label": `Selected review source: ${activeItem.label}` },
            h("p", null, h("strong", null, activeItem.label), activeItem.segmentCount ? ` | ${activeItem.segmentCount} segment(s)` : "", activeItem.speakerCount ? ` | ${activeItem.speakerCount} speaker(s)` : ""),
            activeItem.errorSummary ? h("p", { className: "row-note" }, activeItem.errorSummary) : null,
            h(
              "div",
              { className: "button-row comparison-actions" },
              activeItem.labelSegments && typeof onUseSegments === "function"
                ? h("button", { className: "secondary", type: "button", onClick: () => onUseSegments(activeItem.labelSegments) }, "Use Timing")
                : null,
              activeItem.transcriptPreview && typeof onUseTranscript === "function"
                ? h("button", { className: "secondary", type: "button", onClick: () => onUseTranscript(activeItem.transcriptPreview) }, "Use Transcript")
                : null,
              activeItem.reviewHref ? h("a", { className: "tab-link", href: activeItem.reviewHref }, "Open Review") : null
            )
          )
        : null
    );
  }

  function makeLabelEditorRow(values = {}) {
    return {
      id: values.id || `label-${Date.now()}-${Math.random().toString(36).slice(2)}`,
      start: text(values.start),
      end: text(values.end),
      speaker: text(values.speaker),
      dialogue: text(values.dialogue),
    };
  }

  function secondsForLabelInput(value) {
    const parsed = parseLabelTimestampInput(value);
    return Number.isFinite(parsed) ? parsed.toFixed(3) : "";
  }

  function parseLabelTimestampInput(value) {
    const rawValue = text(value).trim();
    if (!rawValue) {
      return NaN;
    }
    if (rawValue.includes(":")) {
      const parts = rawValue.split(":").map((part) => Number(part.trim()));
      if (parts.length === 2 && parts.every(Number.isFinite)) {
        return parts[0] * 60 + parts[1];
      }
      if (parts.length === 3 && parts.every(Number.isFinite)) {
        return parts[0] * 3600 + parts[1] * 60 + parts[2];
      }
      return NaN;
    }
    const parsed = Number(rawValue);
    return Number.isFinite(parsed) ? parsed : NaN;
  }

  function splitLabelSegmentLine(line) {
    if (line.includes(",")) {
      return line.split(",").map((part) => part.trim());
    }
    if (line.includes("\t")) {
      return line.split("\t").map((part) => part.trim());
    }
    return line.split(/\s+/);
  }

  function parseLabelSegmentRows(rawSegments) {
    return text(rawSegments)
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean)
      .map((line) => {
        const rttmParts = line.split(/\s+/);
        if (rttmParts.length >= 8 && rttmParts[0].toUpperCase() === "SPEAKER") {
          const start = Number(rttmParts[3]);
          const duration = Number(rttmParts[4]);
          return makeLabelEditorRow({
            start: secondsForLabelInput(start),
            end: Number.isFinite(start) && Number.isFinite(duration) ? secondsForLabelInput(start + duration) : "",
            speaker: rttmParts[7] || "",
          });
        }
        const parts = splitLabelSegmentLine(line);
        if (parts.length >= 3) {
          return makeLabelEditorRow({
            start: parts[0],
            end: parts[1],
            speaker: parts.slice(2).join(" "),
          });
        }
        return null;
      })
      .filter(Boolean);
  }

  function labelEditorRowsWithDialogue(rows, rawDialogue) {
    const dialogueLines = text(rawDialogue).split(/\r?\n/);
    return (rows || []).map((row, index) => ({
      ...row,
      dialogue: text(row.dialogue || dialogueLines[index] || ""),
    }));
  }

  function serializeLabelDialogueRows(rows) {
    const lines = (rows || []).map((row) => text(row.dialogue).replace(/\s+$/g, ""));
    while (lines.length && !lines[lines.length - 1].trim()) {
      lines.pop();
    }
    return lines.join("\n");
  }

  function labelEditorRowHasAnyValue(row) {
    return Boolean(text(row.start).trim() || text(row.end).trim() || text(row.speaker).trim());
  }

  function labelEditorRowIsValid(row) {
    const startValue = text(row.start).trim();
    const endValue = text(row.end).trim();
    const speaker = text(row.speaker).trim();
    if (!startValue || !endValue || !speaker) {
      return false;
    }
    const start = parseLabelTimestampInput(startValue);
    const end = parseLabelTimestampInput(endValue);
    return Number.isFinite(start) && Number.isFinite(end) && start >= 0 && end > start;
  }

  function serializeLabelSegmentRows(rows) {
    return (rows || [])
      .filter(labelEditorRowIsValid)
      .map((row) => `${secondsForLabelInput(row.start)} ${secondsForLabelInput(row.end)} ${text(row.speaker).trim()}`)
      .join("\n");
  }

  function labelEditorRowsFromForm(form) {
    if (!form) {
      return [];
    }
    return Array.from(form.querySelectorAll("[data-label-editor-row]")).map((row) => {
      const fieldValue = (fieldName) => {
        const field = row.querySelector(`[data-label-field="${fieldName}"]`);
        return field ? field.value : "";
      };
      return makeLabelEditorRow({
        id: row.getAttribute("data-label-row-id") || "",
        start: fieldValue("start"),
        end: fieldValue("end"),
        speaker: fieldValue("speaker"),
        dialogue: fieldValue("dialogue"),
      });
    });
  }

  function labelEditorStats(rows) {
    const currentRows = rows || [];
    return {
      validCount: currentRows.filter(labelEditorRowIsValid).length,
      incompleteCount: currentRows.filter((row) => labelEditorRowHasAnyValue(row) && !labelEditorRowIsValid(row)).length,
    };
  }

  function syncTrainingLabelEditorForm(form) {
    const rows = labelEditorRowsFromForm(form);
    const serializedSegments = serializeLabelSegmentRows(rows);
    const segmentsInput = form?.querySelector("[data-label-segments-input]");
    if (segmentsInput) {
      segmentsInput.value = serializedSegments;
    }
    const transcriptInput = form?.querySelector("[data-label-transcript-input]");
    if (transcriptInput) {
      const dialogueText = serializeLabelDialogueRows(rows);
      if (dialogueText) {
        transcriptInput.value = dialogueText;
      }
    }
    return { rows, serializedSegments, ...labelEditorStats(rows) };
  }

  function DataTable({ headers, rows, emptyText, renderRow, className }) {
    return h(
      "div",
      { className: classNames("table-scroll", className) },
      h(
        "table",
        null,
        h("thead", null, h("tr", null, headers.map((header) => h("th", { key: header }, header)))),
        h(
          "tbody",
          null,
          rows && rows.length
            ? rows.map(renderRow)
            : h("tr", null, h("td", { colSpan: headers.length }, emptyText || "No rows yet."))
        )
      )
    );
  }

  function SummaryCard({ title, children }) {
    return h("article", { className: "summary-card" }, h("h3", null, title), children);
  }

  function ProgressCard({ title, current, total, detail }) {
    const safeTotal = Math.max(Number(total) || 1, 1);
    const safeCurrent = Math.min(Math.max(Number(current) || 0, 0), safeTotal);
    return h(
      SummaryCard,
      { title },
      h("progress", { value: safeCurrent, max: safeTotal }),
      h("p", { className: "progress-meta" }, h("strong", null, safeCurrent), " of ", h("strong", null, safeTotal)),
      detail ? h("p", { className: "progress-meta" }, detail) : null
    );
  }

  function numericValue(value) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : 0;
  }

  function formatNumber(value, digits = 1) {
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) {
      return "0";
    }
    return parsed.toLocaleString(undefined, {
      maximumFractionDigits: digits,
      minimumFractionDigits: parsed % 1 === 0 ? 0 : Math.min(digits, 2),
    });
  }

  function formatDuration(seconds) {
    const safeSeconds = Math.max(Number(seconds) || 0, 0);
    if (safeSeconds >= 3600) {
      return `${formatNumber(safeSeconds / 3600, 2)} hr`;
    }
    if (safeSeconds >= 60) {
      return `${formatNumber(safeSeconds / 60, 1)} min`;
    }
    return `${formatNumber(safeSeconds, 1)} sec`;
  }

  function formatBytes(bytes) {
    const safeBytes = Math.max(Number(bytes) || 0, 0);
    if (safeBytes >= 1024 * 1024 * 1024) {
      return `${formatNumber(safeBytes / (1024 * 1024 * 1024), 2)} GB`;
    }
    if (safeBytes >= 1024 * 1024) {
      return `${formatNumber(safeBytes / (1024 * 1024), 1)} MB`;
    }
    if (safeBytes >= 1024) {
      return `${formatNumber(safeBytes / 1024, 1)} KB`;
    }
    return `${formatNumber(safeBytes, 0)} bytes`;
  }

  function normalizedAudioCacheUrl(href) {
    const raw = text(href).trim();
    if (!raw) return "";
    try {
      return new URL(raw, window.location.href).href;
    } catch (_error) {
      return "";
    }
  }

  function openAudioCacheDb() {
    return new Promise((resolve) => {
      if (!window.indexedDB) {
        resolve(null);
        return;
      }
      let request;
      try {
        request = window.indexedDB.open(AUDIO_CACHE_DB_NAME, AUDIO_CACHE_DB_VERSION);
      } catch (_error) {
        resolve(null);
        return;
      }
      request.onupgradeneeded = () => {
        const db = request.result;
        if (!db.objectStoreNames.contains(AUDIO_CACHE_STORE)) {
          db.createObjectStore(AUDIO_CACHE_STORE);
        }
      };
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => resolve(null);
      request.onblocked = () => resolve(null);
    });
  }

  // Wait for the txn to commit (oncomplete) before resolving — a parallel
  // write that aborts can still take this read down with it, and we don't
  // want to hand back data that never actually persisted.
  function audioCacheRead(db, key) {
    return new Promise((resolve) => {
      try {
        const tx = db.transaction([AUDIO_CACHE_STORE], "readonly");
        const request = tx.objectStore(AUDIO_CACHE_STORE).get(key);
        let result = null;
        request.onsuccess = () => {
          result = request.result || null;
        };
        request.onerror = () => {
          result = null;
        };
        tx.oncomplete = () => resolve(result);
        tx.onerror = () => resolve(null);
        tx.onabort = () => resolve(null);
      } catch (_error) {
        resolve(null);
      }
    });
  }

  // Same idea on writes — wait for tx.oncomplete. Resolving on request.onsuccess
  // let aborted transactions look successful, so "Cache All Audio" silently
  // dropped writes and reported 0 cached on the next run.
  function audioCacheWrite(db, key, value) {
    return new Promise((resolve) => {
      try {
        const tx = db.transaction([AUDIO_CACHE_STORE], "readwrite");
        const request = tx.objectStore(AUDIO_CACHE_STORE).put(value, key);
        let putOk = false;
        request.onsuccess = () => {
          putOk = true;
        };
        request.onerror = () => {
          putOk = false;
        };
        tx.oncomplete = () => resolve(putOk);
        tx.onerror = () => resolve(false);
        tx.onabort = () => resolve(false);
      } catch (_error) {
        resolve(false);
      }
    });
  }

  async function fetchAudioBlobForCache(url, signal, onProgress) {
    const response = await fetch(url, {
      credentials: "same-origin",
      cache: "force-cache",
      signal,
    });
    if (!response.ok) {
      throw new Error(`Audio fetch failed with status ${response.status}`);
    }
    const total = Number.parseInt(response.headers.get("content-length") || "0", 10) || 0;
    if (!response.body || !response.body.getReader) {
      const blob = await response.blob();
      if (onProgress) onProgress(blob.size, total || blob.size);
      return { blob, size: total || blob.size };
    }
    const reader = response.body.getReader();
    const chunks = [];
    let received = 0;
    let lastProgressAt = 0;
    while (true) {
      const step = await reader.read();
      if (step.done) break;
      chunks.push(step.value);
      received += step.value.byteLength || step.value.length || 0;
      const now = Date.now();
      if (onProgress && now - lastProgressAt > 500) {
        lastProgressAt = now;
        onProgress(received, total);
      }
    }
    const contentType = response.headers.get("content-type") || "audio/wav";
    const blob = new Blob(chunks, { type: contentType });
    if (onProgress) onProgress(blob.size, total || blob.size);
    return { blob, size: total || blob.size };
  }

  async function cacheAudioUrl(db, item, signal, onProgress) {
    const cached = await audioCacheRead(db, item.url);
    if (cached && cached.blob && cached.blob.size > 0) {
      return { status: "cached", bytes: cached.size || cached.blob.size };
    }
    const result = await fetchAudioBlobForCache(item.url, signal, onProgress);
    const stored = await audioCacheWrite(db, item.url, {
      blob: result.blob,
      size: result.size,
      savedAt: Date.now(),
    });
    if (!stored) {
      throw new Error("Could not write audio to local cache");
    }
    return { status: "stored", bytes: result.size };
  }

  function formatPercent(value, digits = 1) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? `${formatNumber(parsed, digits)}%` : "n/a";
  }

  function LinkList({ links, empty = "No artifact links yet." }) {
    return h(
      "div",
      { className: "mini-links" },
      links && links.length
        ? links.map((link, index) => h("a", { key: `${link.href}-${index}`, href: link.href }, link.label || link.path))
        : h("span", null, empty)
    );
  }

  function dialogIdFor() {
    return Array.from(arguments)
      .map((value) => text(value).toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, ""))
      .filter(Boolean)
      .join("-");
  }

  function createdArtifactLabels(links) {
    return (links || [])
      .map((link) => text(link.label || "").trim())
      .filter((label) => label && label !== "Source Audio");
  }

  function preferredArtifactLink(links, preferredLabels) {
    const candidates = links || [];
    const labels = preferredLabels || [];
    for (const label of labels) {
      const match = candidates.find((link) => text(link.label || "") === label);
      if (match) {
        return match;
      }
    }
    return candidates[0] || null;
  }

  function ArtifactPreviewBrowser({ links, preferredLabels, emptyText }) {
    const previewLinks = (links || []).filter((link) => link && link.path && link.previewHref);
    const preferredLink = preferredArtifactLink(previewLinks, preferredLabels);
    const [selectedPath, setSelectedPath] = React.useState(() => (preferredLink ? preferredLink.path : ""));
    const [previewState, setPreviewState] = React.useState({ loading: false, error: "", data: null });
    const previewPaths = previewLinks.map((link) => link.path).join("|");

    React.useEffect(() => {
      if (!previewLinks.length) {
        if (selectedPath) {
          setSelectedPath("");
        }
        return;
      }
      if (!previewLinks.some((link) => link.path === selectedPath)) {
        setSelectedPath((preferredArtifactLink(previewLinks, preferredLabels) || previewLinks[0]).path);
      }
    }, [previewPaths, selectedPath]);

    const activeLink = previewLinks.find((link) => link.path === selectedPath) || preferredLink;
    const activePreviewHref = activeLink ? activeLink.previewHref : "";

    React.useEffect(() => {
      if (!activeLink) {
        setPreviewState({ loading: false, error: "", data: null });
        return undefined;
      }
      if (typeof window.fetch !== "function") {
        setPreviewState({
          loading: false,
          error: "Browser previews are not available in this environment.",
          data: null,
        });
        return undefined;
      }
      let cancelled = false;
      setPreviewState({ loading: true, error: "", data: null });
      window
        .fetch(activeLink.previewHref, {
          cache: "no-store",
          headers: { Accept: "application/json" },
        })
        .then((response) => {
          if (!response.ok) {
            throw new Error(`Preview request failed with ${response.status}`);
          }
          return response.json();
        })
        .then((payload) => {
          if (!cancelled) {
            setPreviewState({
              loading: false,
              error: "",
              data: {
                ...payload,
                href: activeLink.href,
                label: activeLink.label,
              },
            });
          }
        })
        .catch(() => {
          if (!cancelled) {
            setPreviewState({
              loading: false,
              error: "Preview could not be loaded, but the site link still works.",
              data: {
                href: activeLink.href,
                label: activeLink.label,
                kind: activeLink.kind || "binary",
                path: activeLink.path,
              },
            });
          }
        });
      return () => {
        cancelled = true;
      };
    }, [activePreviewHref]);

    if (!previewLinks.length) {
      return h("p", { className: "empty" }, emptyText || "No previewable artifacts are attached to this item yet.");
    }

    const data = previewState.data || {};
    const activeKind = data.kind || activeLink?.kind || "";
    const canOpenSelected = Boolean(activeLink && activeLink.href && ["html", "text", "audio"].includes(activeKind));
    const content = (() => {
      if (previewState.loading) {
        return h("p", { className: "row-note" }, "Loading artifact preview...");
      }
      if (previewState.error && !data.kind) {
        return h("p", { className: "row-note" }, previewState.error);
      }
      if (data.kind === "text") {
        return h(
          React.Fragment,
          null,
          h("p", { className: "row-note" }, data.path || activeLink?.path || ""),
          h("div", { className: "mono-box" }, data.previewText || "This text artifact is empty."),
          data.truncated ? h("p", { className: "row-note" }, "Preview is truncated so large logs do not overload the page.") : null
        );
      }
      if (data.kind === "audio") {
        return h(
          React.Fragment,
          null,
          h("p", { className: "row-note" }, data.path || activeLink?.path || ""),
          data.href ? h("div", { className: "audio-review" }, h("audio", { controls: true, preload: "none", src: data.href })) : null
        );
      }
      return h(
        React.Fragment,
        null,
        h("p", { className: "row-note" }, data.path || activeLink?.path || ""),
        h("p", { className: "row-note" }, data.message || previewState.error || "Preview is not available for this artifact type."),
        canOpenSelected ? h("p", { className: "row-note" }, "Use Open Selected File to inspect this artifact through the site.") : null
      );
    })();

    return h(
      "div",
      { className: "artifact-browser" },
      h(
        "div",
        { className: "artifact-picker" },
        h("label", null, "Preview file"),
        h(
          "select",
          {
            "aria-label": "Preview file",
            value: selectedPath,
            onChange: (event) => setSelectedPath(event.target.value),
          },
          previewLinks.map((link) => h("option", { key: link.path, value: link.path }, link.label || link.path))
        ),
        canOpenSelected
          ? h("a", { className: "tab-link artifact-open-link", href: activeLink.href }, "Open Selected File")
          : activeLink
            ? h("span", { className: "artifact-open-note" }, "Download-only file")
            : null
      ),
      content
    );
  }

  function ArtifactInspectionDialog({ id, title, detail, status, noteLines, audioHref, links, preferredLabels }) {
    const created = createdArtifactLabels(links);
    return h(
      Dialog,
      { id, title, detail },
      h(
        "div",
        { className: "summary-grid" },
        h(
          SummaryCard,
          { title: "Status" },
          h("p", null, h(StatusPill, { status: status || "unknown" })),
          (noteLines || []).map((line, index) => h("p", { key: `${id}-note-${index}`, className: "row-note" }, line))
        ),
        h(
          SummaryCard,
          { title: "Artifacts" },
          created.length
            ? h(React.Fragment, null, h("p", null, h("strong", null, created.length), " linked file(s)"), h("p", { className: "row-note" }, "Use the preview selector below to move between logs, transcripts, flags, and review pages."))
            : h("p", null, "No generated artifacts are linked yet.")
        )
      ),
      audioHref ? h("div", { className: "audio-review" }, h("audio", { controls: true, preload: "none", src: audioHref })) : null,
      h(
        "details",
        { className: "details-box", open: true },
        h("summary", null, "Artifact Preview"),
        h("div", null, h(ArtifactPreviewBrowser, { links, preferredLabels, emptyText: "No text, log, or review artifact is available yet." }))
      )
    );
  }

  function Dialog({ id, title, detail, children }) {
    // <dialog> elements need aria-modal + aria-labelledby for screen readers
    // to announce them correctly. The h2 inside the head supplies the label;
    // we reference it by a stable id derived from the dialog id.
    const titleId = `${id}-title`;
    // Lazy-render the body. Diarization tables and training-label tables put a
    // dialog under every row; without this the page eagerly mounts hundreds of
    // ArtifactPreviewBrowsers / TrainingLabelDialog forms on first load — each
    // one firing its own /api/artifact-preview fetch the second it mounts.
    // We only build the body once the dialog has actually been shown; after
    // that we keep it mounted so re-opens are instant.
    const dialogRef = React.useRef(null);
    const [hasOpened, setHasOpened] = React.useState(false);
    React.useEffect(() => {
      const node = dialogRef.current;
      if (!node) {
        return undefined;
      }
      if (node.open) {
        setHasOpened(true);
        return undefined;
      }
      if (typeof MutationObserver !== "function") {
        // Old browsers we don't really expect — render eagerly so nothing
        // breaks.
        setHasOpened(true);
        return undefined;
      }
      const observer = new MutationObserver(() => {
        if (node.open) {
          setHasOpened(true);
          observer.disconnect();
        }
      });
      observer.observe(node, { attributes: true, attributeFilter: ["open"] });
      return () => observer.disconnect();
    }, []);
    return h(
      "dialog",
      {
        id,
        className: "dashboard-dialog",
        "aria-modal": "true",
        "aria-labelledby": titleId,
        ref: dialogRef,
      },
      h(
        "div",
        { className: "dialog-body" },
        h(
          "div",
          { className: "dialog-head" },
          h("div", null, h("h2", { id: titleId }, title), detail ? h("p", null, detail) : null),
          h(
            "button",
            {
              className: "ghost dialog-close",
              type: "button",
              "data-close-dialog": "yes",
              "aria-label": `Close ${title || "dialog"}`,
            },
            "Close"
          )
        ),
        hasOpened ? children : null
      )
    );
  }

  function LoadingVisual() {
    return h(
      "div",
      { id: "loading-visual", className: "loading-visual", hidden: true },
      h("span", { className: "loader-dot", "aria-hidden": "true" }),
      h("span", { "data-loading-text": "yes" }, "Submitting request...")
    );
  }

  function ThemeToggle() {
    const [theme, setTheme] = React.useState(() => document.documentElement.getAttribute("data-theme") || preferredTheme());
    const isDark = theme === "dark";
    return h(
      "button",
      {
        type: "button",
        className: "theme-toggle",
        "aria-pressed": isDark ? "true" : "false",
        onClick: () => {
          const nextTheme = isDark ? "light" : "dark";
          setTheme(applyTheme(nextTheme));
        },
      },
      isDark ? "Switch to light" : "Switch to dark"
    );
  }

  function AppLayout({ children }) {
    return h(
      "main",
      { className: "app-shell" },
      h(
        "section",
        { className: "masthead" },
        h(
          "article",
          { className: "hero-card" },
          h(
            "div",
            { className: "hero-copy" },
            h("p", { className: "eyebrow" }, APP_TITLE),
            h("h1", null, currentPage.title),
            h("p", { className: "page-intro" }, currentPage.intro || currentPage.summary)
          )
        ),
        h(
          "article",
          { className: "command-card" },
          h("div", { className: "command-tools" }, h(ThemeToggle)),
          h(QuickJump)
        )
      ),
      h(
        "nav",
        { className: "tab-nav" },
        navItems.map((item) =>
          h(
            "a",
            { key: item.path, href: item.href, className: classNames("tab-link", state.currentPath === item.path && "active") },
            item.label
          )
        )
      ),
      h(Notice, { message: state.message, status: state.messageStatus }),
      h(WorkspaceStrip),
      h(LoadingVisual),
      h("section", { className: "workspace" }, h("section", { className: "page-stack" }, children))
    );
  }

  function overviewPrimaryAction() {
    if (Number(ctx.queueCount || 0) > 0) {
      return {
        title: "Process the queued YouTube links",
        detail: `${ctx.queueCount} queued link(s) are waiting. Convert those first so they become numbered audio files.`,
        href: navItems.find((item) => item.path === "/youtube")?.href || "/youtube",
        label: "Open YouTube Audio Conversion",
      };
    }
    if (Number(ctx.audioCount || 0) === 0) {
      return {
        title: "Load source media into the workspace",
        detail: "Start by uploading local media or by adding YouTube links to the queue.",
        href: navItems.find((item) => item.path === "/uploads")?.href || "/uploads",
        label: "Open Media Library",
      };
    }
    if (!ctx.latest?.latestSiteDiarization) {
      return {
        title: "Run diarization on the current library",
        detail: `The workspace already has ${ctx.audioCount} audio file(s). The next practical step is a diarization run.`,
        href: navItems.find((item) => item.path === "/diarization")?.href || "/diarization",
        label: "Open Diarization",
      };
    }
    return {
      title: "Review diarized files",
      detail: "A diarization run exists. Review transcripts, timing files, flags, and logs in the Media Library.",
      href: navItems.find((item) => item.path === "/uploads")?.href || "/uploads",
      label: "Open Media Library",
    };
  }

  function RecentOutputsTable({ limit = 25 }) {
    return h(DataTable, {
      headers: ["Path", "Type", "Link"],
      rows: (ctx.recentOutputs || []).slice(0, limit),
      emptyText: "No output files yet.",
      renderRow: (row) =>
        h(
          "tr",
          { key: row.path },
          h("td", null, row.path),
          h("td", null, row.type),
          h("td", null, row.href ? h("a", { href: row.href }, "Open") : "")
        ),
    });
  }

  function AudioInventoryTable() {
    const rows = ctx.audioFiles || [];
    const folders = audioFolderOptions();
    const moveTargets = folders.filter((folder) => folder.value && folder.value !== (defaults.rootAudioFolderValue || "__root__"));
    const [selected, setSelected] = React.useState(() => new Set());
    const [selectedFolderKeys, setSelectedFolderKeys] = React.useState(null);
    const [rowLimit, setRowLimit] = React.useState(DEFAULT_FILE_VIEW_LIMIT);
    const folderLabel = fileViewFolderLabel;
    const sortedRows = React.useMemo(
      () => sortFileRowsByFolder(rows, folderLabel, (row) => row.fileName || row.name),
      [rows]
    );
    const folderChoices = React.useMemo(() => fileViewFolderChoices(sortedRows, folderLabel), [sortedRows]);
    const folderKeys = folderChoices.map((folder) => folder.key);
    const visibleRows = selectedFolderKeys === null
      ? sortedRows
      : sortedRows.filter((row) => selectedFolderKeys.has(folderLabel(row)));
    const renderedRows = visibleRows.slice(0, Math.min(rowLimit, visibleRows.length));
    const selectedVisibleCount = renderedRows.filter((row) => selected.has(row.name)).length;
    const allFoldersVisible = selectedFolderKeys === null || selectedFolderKeys.size === folderKeys.length;

    // Drop selections that no longer match a row (e.g., the inventory refreshed
    // after the user came back from a delete). Keeps the bulk-delete count
    // honest and avoids posting paths that disappeared.
    React.useEffect(() => {
      const valid = new Set(rows.map((row) => row.name));
      let changed = false;
      const next = new Set();
      selected.forEach((name) => {
        if (valid.has(name)) {
          next.add(name);
        } else {
          changed = true;
        }
      });
      if (changed) {
        setSelected(next);
      }
    }, [rows]);

    React.useEffect(() => {
      if (selectedFolderKeys === null) {
        return;
      }
      const validFolders = new Set(folderKeys);
      const next = new Set(Array.from(selectedFolderKeys).filter((key) => validFolders.has(key)));
      if (next.size === folderKeys.length) {
        setSelectedFolderKeys(null);
        return;
      }
      if (next.size !== selectedFolderKeys.size) {
        setSelectedFolderKeys(next);
      }
    }, [folderKeys.join("\u0000"), selectedFolderKeys]);

    const toggleRow = (name, checked) => {
      setSelected((prev) => {
        const next = new Set(prev);
        if (checked) {
          next.add(name);
        } else {
          next.delete(name);
        }
        return next;
      });
    };

    const allChecked = renderedRows.length > 0 && renderedRows.every((row) => selected.has(row.name));
    const someChecked = !allChecked && renderedRows.some((row) => selected.has(row.name));
    const headerRef = React.useRef(null);
    React.useEffect(() => {
      if (headerRef.current) {
        headerRef.current.indeterminate = someChecked;
      }
    }, [someChecked]);

    const toggleShownRows = (checked) => {
      setSelected((prev) => {
        const next = new Set(prev);
        renderedRows.forEach((row) => {
          if (checked) {
            next.add(row.name);
          } else {
            next.delete(row.name);
          }
        });
        return next;
      });
    };

    const toggleFolder = (folderKey, checked) => {
      const base = selectedFolderKeys === null ? new Set(folderKeys) : new Set(selectedFolderKeys);
      if (checked) {
        base.add(folderKey);
      } else {
        base.delete(folderKey);
      }
      setSelectedFolderKeys(base.size === folderKeys.length ? null : base);
    };

    const renderMoveControl = (row) => {
      const targets = moveTargets.filter((folder) => folder.value !== (row.folder || ""));
      if (!targets.length) {
        return h("p", { className: "row-note" }, "No other folders to move into.");
      }
      const selectId = `move_target_${row.name.replace(/[^A-Za-z0-9_-]+/g, "_")}`;
      return h(
        "form",
        {
          method: "post",
          action: routes.moveAudioFile,
          className: "inline-form",
          onSubmit: (event) => {
            const dest = event.currentTarget.elements.namedItem("target_folder");
            const destValue = dest && "value" in dest ? dest.value : "";
            if (!destValue) {
              event.preventDefault();
              window.alert("Pick a destination folder first.");
            }
          },
        },
        h("input", { type: "hidden", name: "audio_path", value: row.name }),
        h(
          "select",
          { id: selectId, name: "target_folder", className: "compact-control", defaultValue: targets[0]?.value || "" },
          targets.map((folder) =>
            h("option", { key: folder.value, value: folder.value }, folder.name || folder.value)
          )
        ),
        h("button", { className: "secondary", type: "submit" }, "Move")
      );
    };
    const renderDeleteControl = (row) =>
      h(
        "form",
        {
          method: "post",
          action: routes.deleteAudioFile,
          className: "inline-form",
          onSubmit: (event) => {
            if (!window.confirm(`Delete ${row.name}? This removes the WAV, its diarization transcripts/SRT/review files, runtime summary rows, and selection-list entries. The matching YouTube URL stays in youtube_links.txt so it can be redownloaded.`)) {
              event.preventDefault();
            }
          },
        },
        h("input", { type: "hidden", name: "audio_path", value: row.name }),
        h("button", { className: "secondary danger", type: "submit" }, "Delete")
      );

    const selectedCount = selected.size;
    // The bulk-delete form is rendered as an outer wrapper around the table
    // so its hidden inputs (one per checked row) post together. Toolbar
    // controls live in the form so the submit button + counts stay in sync.
    return h(
      "form",
      {
        method: "post",
        action: routes.bulkDeleteAudioFiles,
        className: "audio-inventory-form",
        onSubmit: (event) => {
          if (!selectedCount) {
            event.preventDefault();
            window.alert("Pick at least one file to delete.");
            return;
          }
          if (
            !window.confirm(
              `Delete ${selectedCount} audio file(s)? This removes each WAV, its diarization transcripts/SRT/review files, runtime summary rows, and selection-list entries. Matching YouTube URLs stay in youtube_links.txt so they can be redownloaded.`
            )
          ) {
            event.preventDefault();
          }
        },
      },
      Array.from(selected).map((name) => h("input", { key: name, type: "hidden", name: "audio_paths", value: name })),
      h(
        "div",
        { className: "audio-inventory-toolbar" },
        h(
          "div",
          { className: "audio-inventory-toolbar-left" },
	          h(
	            "span",
	            { className: "audio-inventory-summary" },
		            rows.length
		              ? `${renderedRows.length} visible of ${visibleRows.length} matching; ${selectedVisibleCount} visible selected${selectedCount > selectedVisibleCount ? ` (${selectedCount} total)` : ""}`
		              : "No audio files yet"
		          ),
		          h(FileViewLimitControl, {
		            id: "audio_inventory_row_limit",
		            total: visibleRows.length,
		            shown: renderedRows.length,
		            limit: rowLimit,
		            onLimitChange: setRowLimit,
		          })
		      ),
	        h(
	          "div",
	          { className: "audio-inventory-bulk-actions" },
	          h(
	            "button",
	            { type: "button", className: "secondary btn-sm", disabled: !renderedRows.length, onClick: () => toggleShownRows(true) },
	            "Select shown"
	          ),
          h(
            "button",
            { type: "button", className: "secondary btn-sm", disabled: !selectedCount, onClick: () => setSelected(new Set()) },
            "Clear selected"
          ),
          h(
            "button",
            { type: "submit", className: "secondary danger btn-sm", disabled: !selectedCount },
            selectedCount ? `Delete selected (${selectedCount})` : "Delete selected"
          )
        ),
        rows.length
          ? h(
              "fieldset",
              { className: "audio-folder-filter" },
              h("legend", null, "View folders"),
              h(
                "div",
                { className: "audio-folder-filter-actions" },
                h(
                  "button",
                  { type: "button", className: "ghost btn-sm", disabled: allFoldersVisible, onClick: () => setSelectedFolderKeys(null) },
                  "View all"
                ),
                h(
                  "button",
                  { type: "button", className: "ghost btn-sm", disabled: selectedFolderKeys !== null && selectedFolderKeys.size === 0, onClick: () => setSelectedFolderKeys(new Set()) },
                  "Clear folders"
                )
              ),
              h(
                "div",
                { className: "audio-folder-options" },
                folderChoices.map((folder) =>
                  h(
                    "label",
                    { key: folder.key, className: "audio-folder-option" },
                    h("input", {
                      type: "checkbox",
                      checked: selectedFolderKeys === null || selectedFolderKeys.has(folder.key),
                      onChange: (event) => toggleFolder(folder.key, event.target.checked),
                    }),
                    h("span", null, folder.label),
                    h("em", null, folder.count)
                  )
                )
              )
            )
          : null
      ),
      h(DataTable, {
        headers: [
          h("input", {
            ref: headerRef,
            type: "checkbox",
	            "aria-label": "Select all audio files",
	            checked: allChecked,
	            disabled: !renderedRows.length,
	            onChange: (event) => toggleShownRows(event.target.checked),
	          }),
          "#",
          "Folder",
          "Filename",
          "Type",
          "Actions",
        ],
	        className: "audio-inventory-table dense-file-table file-view-table",
	        rows: renderedRows,
        emptyText: rows.length
          ? h("div", { className: "empty-state" }, h("strong", null, "No files match the selected folder view."), "Choose another folder or use View all.")
          : h(
              "div",
              { className: "empty-state" },
              h("strong", null, "No audio files yet."),
              "Upload one with the form above, or queue a YouTube URL on the YouTube Audio Conversion tab to download one straight into ",
              h("code", null, "audio_in/youtube_links/"),
              "."
            ),
        renderRow: (row, index) =>
          h(
            "tr",
            { key: row.name, className: selected.has(row.name) ? "is-selected" : "" },
            h(
              "td",
              { className: "select-cell" },
              h("input", {
                type: "checkbox",
                "aria-label": `Select ${row.name}`,
                checked: selected.has(row.name),
                onChange: (event) => toggleRow(row.name, event.target.checked),
              })
            ),
            h("td", null, index + 1),
            h("td", null, folderLabel(row)),
	            h("td", { title: row.path || row.name }, h("strong", { className: "file-name" }, row.fileName || row.name)),
            h("td", null, row.type),
            h("td", null, h("div", { className: "row-actions" }, renderMoveControl(row), renderDeleteControl(row)))
          ),
      })
    );
  }

  function AudioFolderManager() {
    const folders = audioFolderOptions();
    const renamable = folders.filter((folder) => folder.renamable !== false);
    const deletable = folders.filter((folder) => folder.value);
    return h(
      "div",
      { className: "folder-manager" },
      h(
        "form",
        { method: "post", action: routes.createAudioFolder },
        h(Field, { id: "folder_name", label: "New folder" }, h(TextInput, { id: "folder_name", placeholder: "example_set" })),
        h("p", null, h("button", { className: "secondary", type: "submit" }, "Create"))
      ),
      h(
        "form",
        { method: "post", action: routes.renameAudioFolder },
        h(Field, { id: "current_folder", label: "Rename folder" }, h(AudioFolderSelect, { id: "current_folder", name: "current_folder", defaultValue: renamable[0]?.value || defaults.defaultUploadAudioFolder || "file_uploads" })),
        h(Field, { id: "new_folder_name", label: "New name" }, h(TextInput, { id: "new_folder_name", placeholder: "renamed_set" })),
        h("p", null, h("button", { className: "secondary", type: "submit" }, "Rename"))
      ),
      deletable.length
        ? h(
            "form",
            {
              method: "post",
              action: routes.deleteAudioFolder,
              onSubmit: (event) => {
                const select = event.currentTarget.elements.namedItem("folder_name");
                const folderValue = select && "value" in select ? select.value : "";
                const folderRow = folders.find((folder) => folder.value === folderValue);
                const folderLabel = folderRow?.name || folderValue;
                const fileCount = folderRow ? folderRow.fileCount || 0 : 0;
                const fileSummary = fileCount ? ` ALL ${fileCount} file(s) inside, plus their diarization transcripts/SRT/review files and YouTube URL associations,` : "";
                if (!window.confirm(`Delete folder "${folderLabel}"?${fileSummary} will be permanently removed. This cannot be undone.`)) {
                  event.preventDefault();
                }
              },
            },
            h(Field, { id: "delete_folder_name", label: "Delete folder", note: "Cascades through every WAV inside and removes its derived diarization artifacts. Unsorted Root is cleared but the media library stays available." }, h(AudioFolderSelect, { id: "delete_folder_name", name: "folder_name", defaultValue: deletable[0]?.value || "" })),
            h("p", null, h("button", { className: "secondary danger", type: "submit" }, "Delete"))
          )
        : null
    );
  }

  function DiarizedFilesTable({ rows }) {
    const allRows = rows || [];
    const [folderFilter, setFolderFilter] = React.useState("all");
    const [rowLimit, setRowLimit] = React.useState(DEFAULT_FILE_VIEW_LIMIT);
    const sortedRows = React.useMemo(
      () => sortFileRowsByFolder(allRows, fileViewFolderLabel, (row) => row.fileName || row.audioFile),
      [allRows]
    );
    const folderChoices = React.useMemo(() => fileViewFolderChoices(sortedRows, fileViewFolderLabel), [sortedRows]);
    React.useEffect(() => {
      if (folderFilter !== "all" && !folderChoices.some((folder) => folder.key === folderFilter)) {
        setFolderFilter("all");
      }
    }, [folderFilter, folderChoices]);
    const filteredRows = folderFilter === "all"
      ? sortedRows
      : sortedRows.filter((row) => fileViewFolderLabel(row) === folderFilter);
    const renderedRows = filteredRows.slice(0, Math.min(rowLimit, filteredRows.length));
    return h(
      React.Fragment,
      null,
      allRows.length
        ? h(
            "div",
            { className: "selection-toolbar file-view-toolbar" },
            h(FolderFilterControl, { id: "diarized_folder_filter", value: folderFilter, onChange: setFolderFilter, folders: folderChoices }),
            h(FileViewLimitControl, {
              id: "diarized_row_limit",
              total: filteredRows.length,
              shown: renderedRows.length,
              limit: rowLimit,
              onLimitChange: setRowLimit,
            })
          )
        : null,
      h(DataTable, {
        className: "diarized-files-table dense-file-table file-view-table",
        headers: ["Audio", "Status", "Model / Run", "Files", "Inspect"],
        rows: renderedRows,
        emptyText: h(
          "div",
          { className: "empty-state" },
          h("strong", null, allRows.length ? "No diarized files match this folder." : "No diarized files yet."),
          allRows.length
            ? "Choose another folder or switch back to All folders."
            : "Pick one or more audio files on the Diarization tab and submit a run; transcripts, SRT timing, and review pages will appear here when each file finishes."
        ),
        renderRow: (row, index) => {
          const dialogId = dialogIdFor("media-artifact", row.audioFile, row.runName, index);
          const created = createdArtifactLabels(row.links);
          return h(
            "tr",
            { key: `${row.audioFile}-${row.runName}-${index}` },
            h(
              "td",
              { title: row.audioFile || "" },
              h("strong", { className: "file-name" }, row.fileName || row.audioFile),
              h("p", { className: "row-note" }, fileViewFolderLabel(row)),
              row.audioHref
                ? h("div", { className: "audio-review compact-audio" }, h("audio", { controls: true, preload: "none", src: row.audioHref }))
                : h("p", { className: "row-note" }, "Source audio not found")
            ),
            h("td", null, h(StatusPill, { status: row.status || "unknown" }), row.errorSummary ? h("p", { className: "row-note" }, row.errorSummary) : null),
            h("td", null, h("span", { className: "run-name" }, row.runName || "unknown run"), h("p", { className: "row-note" }, row.lastRun || ""), row.modelLabel ? h("p", { className: "row-note" }, `Model: ${row.modelLabel}`) : row.backend ? h("p", { className: "row-note" }, `Model: ${row.backendLabel || row.backend}`) : null, row.batchId ? h("p", { className: "row-note" }, `Batch: ${row.batchId}`) : null),
            h("td", null, h(LinkList, { links: row.links, empty: "No output files linked yet" }), created.length ? h("p", { className: "row-note" }, `Created: ${created.join(", ")}`) : null),
            h("td", null, h("button", { className: "secondary", type: "button", "data-open-dialog": dialogId }, "Inspect"), h(ArtifactInspectionDialog, { id: dialogId, title: row.audioFile || "Diarization artifact", detail: row.runName || "Generated diarization outputs", status: row.status || "unknown", noteLines: [row.lastRun ? `Last run: ${row.lastRun}` : "", row.modelLabel ? `Model: ${row.modelLabel}` : "", row.errorSummary || ""].filter(Boolean), audioHref: row.audioHref, links: row.links || [], preferredLabels: ["Transcript", "Diarized Times", "Summary", "Review Flags", "Review Page"] }))
          );
        },
      })
    );
  }

  function ModelCoverage({ statuses }) {
    const rows = statuses || [];
    if (!rows.length) {
      return h("p", { className: "row-note" }, "No model tracking data yet.");
    }
    return h(
      "div",
      { className: "model-status-list" },
      rows.map((item) =>
        h(
          "div",
          { className: "model-status-item", key: item.key || item.backend || item.label },
          h("strong", null, item.label || item.backend),
          h(StatusPill, { status: item.status || "not run" }),
          item.runName ? h("span", { className: "row-note" }, item.runName) : null
        )
      )
    );
  }

  function ReviewBundleForm({ historyRows }) {
    const srtOptions = (historyRows || ctx.diarization?.history || [])
      .flatMap((row) => row.links || [])
      .filter((link) => link.label === "Diarized Times" && link.path);
    return h(
      "details",
      { className: "details-box" },
      h("summary", null, "Generate A Review Bundle From An Existing SRT"),
      h(
        "div",
        null,
        h(
          "form",
          { method: "post", action: routes.review },
          h(Field, { id: "srt_path", label: "Existing SRT path" }, h(SelectInput, { id: "srt_path" }, srtOptions.length ? srtOptions.map((link) => h(Option, { key: link.path, value: link.path }, link.path)) : h(Option, { value: "" }, "No SRT files found"))),
          h(Field, { id: "media_path", label: "Optional media path" }, h(TextInput, { id: "media_path", placeholder: "audio_in/001_example.wav" })),
          h("p", null, h("button", { className: "secondary", type: "submit" }, "Generate Review Bundle"))
        )
      )
    );
  }

  function DiarizationRunItemsTable({ items, runName }) {
    return h(DataTable, {
      headers: ["Audio File", "Status", "Runtime (s)", "Created Files", "Inspect"],
      rows: items || [],
      emptyText: "This run has not written any per-file outputs yet.",
      renderRow: (item, index) => {
        const dialogId = dialogIdFor("run-item", runName, item.audioFile, index);
        const created = createdArtifactLabels(item.links);
        const rowClassName = item.status === "failed"
          ? "diarization-item-row diarization-item-row--failed"
          : "diarization-item-row";
        return h(
          "tr",
          { key: `${item.audioFile}-${index}`, className: rowClassName },
          h("td", null, item.audioFile || ""),
          h("td", null, h(StatusPill, { status: item.status || "unknown" }), item.errorSummary ? h("p", { className: "row-note" }, item.errorSummary) : null),
          h("td", null, item.runtimeSeconds || ""),
          h("td", null, created.length ? created.join(", ") : "No linked outputs yet."),
          h("td", null, h("button", { className: "secondary", type: "button", "data-open-dialog": dialogId }, "Inspect"), h(ArtifactInspectionDialog, { id: dialogId, title: item.audioFile || "Run item", detail: runName || "Latest diarization run output", status: item.status || "unknown", noteLines: [item.runtimeSeconds ? `Runtime: ${item.runtimeSeconds} seconds` : "", item.errorSummary || ""].filter(Boolean), audioHref: item.audioHref, links: item.links || [], preferredLabels: ["stdout", "stderr", "Transcript", "Diarized Times", "Review Flags", "Review Page"] }))
        );
      },
    });
  }

  function OverviewPage() {
    const action = overviewPrimaryAction();
    const sections = [
      ["Bring Data In", "Upload audio or convert queued links into the shared library.", [["/uploads", "Media Library"], ["/youtube", "YouTube Audio Conversion"]]],
      ["Run Core Processing", "Select audio, stitch training clips, run diarization, then review generated files.", [["/stitching", "Audio Stitching"], ["/diarization", "Diarization"], ["/uploads", "Diarized Files"]]],
      ["Train And Inspect", "Create labels and prepare training artifacts.", [["/training-labels", "Training Labels"], ["/fine-tuning", "Fine-Tuning"]]],
    ];
    return h(
      React.Fragment,
      null,
      h(
        "article",
        { className: "panel" },
        h(
          "div",
          { className: "summary-grid" },
          h(SummaryCard, { title: "Recommended Next Step" }, h("p", null, h("strong", null, action.title)), h("p", null, action.detail), h("div", { className: "mini-links" }, h("a", { href: action.href }, action.label))),
          h(SummaryCard, { title: "Current Workspace" }, h("ul", null, h("li", null, `${ctx.audioCount || 0} audio input file(s)`), h("li", null, `${ctx.queueCount || 0} queued YouTube link(s)`), h("li", null, `${ctx.projectCount || 0} fine-tuning project(s)`), h("li", null, `Failed-link reports: ${ctx.failedLinksPresent ? "present" : "none"}`))),
          h(SummaryCard, { title: "Latest Site Runs" }, h("ul", null, h("li", null, `Diarization: ${ctx.latest?.latestSiteDiarization?.name || "none"}`), h("li", null, `YouTube: ${ctx.latest?.latestSiteYoutube?.name || "none"}`)), h("p", { className: "progress-meta" }, "The homepage stays short on purpose. Use the page links below when you are ready to work."))
        )
      ),
      h(
        "article",
        { className: "panel" },
        h("div", { className: "panel-head" }, h("div", null, h("h2", null, "Workflow Sections"), h("p", null, "Grouped entry points for the main tasks."))),
        h(
          "div",
          { className: "card-grid" },
          sections.map(([title, description, links]) =>
            h(
              "article",
              { className: "feature-card", key: title },
              h("h2", null, title),
              h("p", null, description),
              h(
                "div",
                { className: "mini-links" },
                links.map(([path, label]) =>
                  h("a", { key: path, href: navItems.find((item) => item.path === path)?.href || path }, label)
                )
              )
            )
          )
        )
      ),
      h("article", { className: "panel" }, h("div", { className: "panel-head" }, h("div", null, h("h2", null, "Recent Activity"), h("p", null, "Newest artifacts from the workspace."))), h(RecentOutputsTable, { limit: 6 })),
      h(FileSearchPanel)
    );
  }

  function UploadsPage() {
    const diarization = ctx.diarization || {};
    const history = diarization.history || [];
    const [modelFilter, setModelFilter] = React.useState(() => uiState.uploadsModelFilter || "all");
    const filterOptions = diarizationHistoryFilterOptions(history);
    React.useEffect(() => {
      if (!filterOptions.some((option) => option.value === modelFilter)) {
        setModelFilter("all");
      }
    }, [filterOptions, modelFilter]);
    React.useEffect(() => {
      uiState.uploadsModelFilter = modelFilter;
    }, [modelFilter]);
    const filteredHistory = history.filter((row) => {
      if (modelFilter === "all") {
        return true;
      }
      if (modelFilter.startsWith("backend:")) {
        return row.backend === modelFilter.slice("backend:".length);
      }
      return row.modelKey === modelFilter;
    });
    return h(
      React.Fragment,
      null,
      h(
        "article",
        { className: "panel" },
        h(ProcessSteps, { steps: ["Choose a folder", "Upload source files", "Review generated outputs"] }),
        h("div", { className: "panel-head" }, h("div", null, h("h2", null, "Upload Source Files"), h("p", null, "Add one or more supported audio files."))),
        h(
          "form",
          { method: "post", action: routes.uploadAudio, encType: "multipart/form-data", "data-upload-progress": "media_upload_progress", "data-loading-message": "Uploading media files..." },
          h(Field, { id: "audio_folder", label: "Target folder" }, h(AudioFolderSelect, { id: "audio_folder", name: "audio_folder", defaultValue: defaults.defaultUploadAudioFolder || "file_uploads" })),
          h(Field, { id: "new_audio_folder", label: "Optional new folder" }, h(TextInput, { id: "new_audio_folder", name: "new_audio_folder", placeholder: "new_set_name" })),
          h(Field, { id: "audio_files", label: "Supported audio files" }, h("input", { id: "audio_files", type: "file", name: "audio_files", multiple: true, required: true, accept: defaults.uploadAudioAccept, "data-file-summary": "audio_upload_summary" })),
          h("p", { id: "audio_upload_summary", className: "field-status" }, "No file selected yet."),
          h(
            "div",
            { id: "media_upload_progress", className: "upload-progress-tracker", hidden: true, "aria-live": "polite" },
            h("div", { className: "upload-progress-head" }, h("strong", { "data-upload-progress-status": true }, "Ready to upload"), h("span", { "data-upload-progress-percent": true }, "0%")),
            h("progress", { value: 0, max: 100, "data-upload-progress-bar": true }),
            h("p", { className: "progress-meta", "data-upload-progress-meta": true }, "Waiting for upload to start.")
          ),
          h("p", { className: "footer-note" }, "Accepted uploads are converted to ", h("code", null, ".wav"), ". Source formats: ", h("code", null, (defaults.uploadAudioSuffixes || []).join(", ")), "."),
          h("p", null, h("button", { type: "submit" }, "Upload files"))
        )
      ),
      h("article", { className: "panel compact-panel" }, h("div", { className: "panel-head" }, h("div", null, h("h2", null, "Audio Folders"), h("p", null, "Create or rename set folders used by uploads, YouTube conversion, and diarization."))), h(AudioFolderManager)),
      h("article", { className: "panel" }, h("div", { className: "panel-head" }, h("div", null, h("h2", null, "Source Audio File Tracker"), h("p", null, "This table shows the uploaded audio files that the rest of the pipeline can already see."))), h(AudioInventoryTable)),
      h(
        "article",
        { className: "panel" },
        h(
          "div",
          { className: "panel-head" },
          h("div", null, h("h2", null, "Diarized File Tracker"), h("p", null, "Review diarized audio, transcripts, speaker-time files, review pages, flags, and logs in one place."))
        ),
        h("div", { className: "summary-grid" }, h(SummaryCard, { title: "Tracked Files" }, h("p", null, h("strong", null, filteredHistory.length), " diarized item(s)")), h(SummaryCard, { title: "Latest Run" }, h("p", null, h(StatusPill, { status: diarization.latestRun?.status || "not started" })), h("p", { className: "run-name" }, diarization.latestRun?.name || "No site run yet."))),
        h(LatestRunQueueTracker, { latestRun: diarization.latestRun, activeRuns: diarization.activeRuns, emptyText: "No diarization Slurm job has been submitted yet.", storageKey: "diarization-active-tab" }),
        h(
          "div",
          { className: "selection-toolbar" },
          h("label", { className: "compact-label", htmlFor: "uploads_model_filter" }, "Show diarized files from"),
          h(
            "select",
            {
              id: "uploads_model_filter",
              className: "compact-control",
              value: modelFilter,
              onChange: (event) => setModelFilter(event.target.value),
            },
            filterOptions.map((option) => h("option", { key: option.value, value: option.value }, option.label))
          )
        ),
        h(DiarizedFilesTable, { rows: filteredHistory }),
        h(ReviewBundleForm, { historyRows: filteredHistory })
      )
    );
  }

  function StitchedInspectionDialog({ row, id }) {
    const segments = row?.segments || [];
    const preferredLabels = ["Timing Review", "Segment Manifest", "RTTM", "Segment SRT", "Transcript Notes", "metadata.json", "stdout.log", "stderr.log"];
    const trainingText = row.trainingUsage?.length
      ? row.trainingUsage.map((item) => item.project_key || `${item.backend}/${item.project_name}`).filter(Boolean).join(", ")
      : "Not added to training";
    return h(
      Dialog,
      { id, title: row.displayName || row.outputName || row.name || "Stitched audio", detail: row.path || "stitched/" },
      h(
        "div",
        { className: "summary-grid" },
        h(SummaryCard, { title: "Status" }, h("p", null, h(StatusPill, { status: row.status || "unknown" })), row.error ? h("p", { className: "row-note" }, row.error) : null),
        h(SummaryCard, { title: "Timing" }, h("p", null, h("strong", null, row.segmentCount || segments.length || 0), " segment(s)"), h("p", { className: "row-note" }, `${formatDuration(row.durationSeconds || 0)} total, seed ${row.seed || "auto"}`)),
        h(SummaryCard, { title: "Training" }, h("p", null, trainingText), row.trainingTargets?.length ? h("p", { className: "row-note" }, `Requested: ${row.trainingTargets.join(", ")}`) : null)
      ),
      row.audioHref ? h("div", { className: "audio-review" }, h("audio", { controls: true, preload: "none", src: row.audioHref })) : null,
      h("div", { className: "button-row" }, h("button", { className: "secondary", type: "button", onClick: () => promptRenameStitch(row) }, "Rename")),
      h(
        "details",
        { className: "details-box", open: true },
        h("summary", null, "Segment Timing"),
        h(
          "div",
          null,
          h(DataTable, {
            className: "compact-table",
            headers: ["#", "Start", "End", "Speaker", "Source"],
            rows: segments,
            emptyText: "No segment metadata is available yet.",
            renderRow: (segment) =>
              h(
                "tr",
                { key: `${segment.index}-${segment.audio_file}` },
                h("td", null, segment.index),
                h("td", null, formatNumber(segment.start || 0, 3)),
                h("td", null, formatNumber(segment.end || 0, 3)),
                h("td", null, segment.speaker || ""),
                h("td", null, segment.audio_file || "")
              ),
          })
        )
      ),
      h(
        "details",
        { className: "details-box", open: true },
        h("summary", null, "Artifacts"),
        h("div", null, h(ArtifactPreviewBrowser, { links: row.links || [], preferredLabels, emptyText: "No stitched artifacts are available yet." }))
      )
    );
  }

  function StitchedRunsTable({ rows }) {
    return h(DataTable, {
      className: "stitched-runs-table",
      headers: ["Output", "Status", "Timing", "Training", "Files", "Inspect"],
      rows: rows || [],
      emptyText: h("div", { className: "empty-state" }, h("strong", null, "No stitched samples yet."), "Select audio files above to create the first stitched WAV and RTTM pair."),
      renderRow: (row, index) => {
        const dialogId = dialogIdFor("stitched-run", row.name, index);
        const displayName = row.displayName || row.outputName || row.name;
        const trainingTargets = row.trainingUsage?.length
          ? row.trainingUsage.map((item) => item.project_key || `${item.backend}/${item.project_name}`).filter(Boolean)
          : row.trainingTargets || [];
        return h(
          "tr",
          { key: `${row.name}-${index}` },
          h("td", null, h("strong", { className: "file-name" }, displayName), h("p", { className: "row-note" }, row.outputName && row.outputName !== displayName ? `Artifact name: ${row.outputName}` : row.path || ""), row.outputName && row.outputName !== displayName ? h("p", { className: "row-note" }, row.path || "") : null),
          h("td", null, h(StatusPill, { status: row.status || "unknown" }), row.error ? h("p", { className: "row-note" }, row.error) : null),
          h("td", null, `${row.segmentCount || 0} segment(s)`, h("p", { className: "row-note" }, `${formatDuration(row.durationSeconds || 0)} · seed ${row.seed || "auto"}`)),
          h("td", null, trainingTargets.length ? trainingTargets.join(", ") : h("span", { className: "row-note" }, "Not added")),
          h("td", null, h(LinkList, { links: row.links || [], empty: "No linked files yet" })),
          h("td", null, h("div", { className: "row-actions" }, h("button", { className: "secondary", type: "button", "data-open-dialog": dialogId }, "Inspect"), h("button", { className: "ghost", type: "button", onClick: () => promptRenameStitch(row) }, "Rename"), h("button", { className: "ghost danger", type: "button", onClick: () => confirmDeleteStitch(row) }, "Delete")), h(StitchedInspectionDialog, { row, id: dialogId }))
        );
      },
    });
  }

  function StitchingPage() {
    const rows = ctx.audioFiles || [];
    const stitching = ctx.stitching || {};
    const stitchedRows = stitching.rows || [];
    const summary = stitching.summary || {};
    const [selected, setSelected] = React.useState(() => new Set());
    const projectOptions = React.useMemo(
      () =>
        (ctx.projects || [])
          .filter((project) => project && project.slug && project.backend)
          .map((project) => ({
            key: `${project.backend}/${project.slug}`,
            backend: project.backend,
            slug: project.slug,
            label: `${backendDisplayName(project.backend)} / ${project.displayName || project.slug}`,
            sampleCount: project.sampleCount || 0,
          }))
          .sort((left, right) => left.label.toLowerCase().localeCompare(right.label.toLowerCase())),
      [JSON.stringify((ctx.projects || []).map((project) => [project.backend, project.slug, project.displayName, project.sampleCount]))]
    );
    const [addToTraining, setAddToTraining] = React.useState(true);
    const [includeManualTarget, setIncludeManualTarget] = React.useState(() => projectOptions.length === 0);
    const [trainingTargets, setTrainingTargets] = React.useState(() => new Set());
    const [randomPickCount, setRandomPickCount] = React.useState("");
    const [folderRandomCounts, setFolderRandomCounts] = React.useState({});

    const folderChoices = Array.from(
      rows.reduce((choices, row) => {
        const key = text(row.folder || "Unsorted Root", "Unsorted Root");
        const current = choices.get(key) || { key, label: key, count: 0 };
        current.count += 1;
        choices.set(key, current);
        return choices;
      }, new Map()).values()
    ).sort((left, right) => left.label.toLowerCase().localeCompare(right.label.toLowerCase()));

    const initialFolderSpeakers = React.useMemo(() => {
      const mapping = {};
      folderChoices.forEach((folder, index) => {
        mapping[folder.key] = `Speaker_${index}`;
      });
      return mapping;
    }, [folderChoices.map((folder) => folder.key).join("|")]);

    const [folderSpeakers, setFolderSpeakers] = React.useState(initialFolderSpeakers);
    const [speakerByFile, setSpeakerByFile] = React.useState({});

    React.useEffect(() => {
      setFolderSpeakers((current) => ({ ...initialFolderSpeakers, ...current }));
    }, [initialFolderSpeakers]);

    React.useEffect(() => {
      const valid = new Set(rows.map((row) => row.name));
      setSelected((current) => new Set(Array.from(current).filter((name) => valid.has(name))));
      setSpeakerByFile((current) => {
        const next = {};
        Object.entries(current).forEach(([name, speaker]) => {
          if (valid.has(name)) {
            next[name] = speaker;
          }
        });
        return next;
      });
    }, [rows]);

    React.useEffect(() => {
      const validTargets = new Set(projectOptions.map((project) => project.key));
      setTrainingTargets((current) => new Set(Array.from(current).filter((target) => validTargets.has(target))));
      if (projectOptions.length === 0) {
        setIncludeManualTarget(true);
      }
    }, [projectOptions.map((project) => project.key).join("|")]);

    const folderLabel = (row) => text(row.folder || "Unsorted Root", "Unsorted Root");
    const speakerForRow = (row) => speakerByFile[row.name] || folderSpeakers[folderLabel(row)] || "Speaker_0";
    const updateSpeakerForFile = (name, speaker) => {
      setSpeakerByFile((current) => ({ ...current, [name]: speaker }));
    };
    const updateSpeakerForFolder = (folderKey, speaker) => {
      setFolderSpeakers((current) => ({ ...current, [folderKey]: speaker }));
      setSpeakerByFile((current) => {
        const next = { ...current };
        rows.filter((row) => folderLabel(row) === folderKey).forEach((row) => {
          next[row.name] = speaker;
        });
        return next;
      });
    };
    const toggleRow = (name, checked) => {
      setSelected((current) => {
        const next = new Set(current);
        if (checked) {
          next.add(name);
        } else {
          next.delete(name);
        }
        return next;
      });
    };
    const selectAll = () => setSelected(new Set(rows.map((row) => row.name)));
    const clearSelected = () => setSelected(new Set());
    const selectedRows = rows.filter((row) => selected.has(row.name));
    const projectNames = Array.from(new Set((ctx.projects || []).map((project) => project.slug).filter(Boolean))).sort();
    const defaultProject = projectNames[0] || "stitched-site-training";
    const halfCount = (total) => Math.ceil(Math.max(Number(total) || 0, 0) / 2);
    const boundedRandomCount = (rawValue, total, fallback) => {
      const safeTotal = Math.max(Number(total) || 0, 0);
      const parsed = Number.parseInt(rawValue, 10);
      const requested = Number.isFinite(parsed) ? parsed : fallback;
      return Math.min(Math.max(requested, 0), safeTotal);
    };
    const randomNamesFromRows = (candidateRows, count) => {
      const names = candidateRows.map((row) => row.name);
      for (let index = names.length - 1; index > 0; index -= 1) {
        const swapIndex = Math.floor(Math.random() * (index + 1));
        [names[index], names[swapIndex]] = [names[swapIndex], names[index]];
      }
      return names.slice(0, count);
    };
    const chooseRandomRows = (candidateRows, rawCount, { replaceAll = false, replaceCandidates = false } = {}) => {
      const count = boundedRandomCount(rawCount, candidateRows.length, halfCount(candidateRows.length));
      const picks = new Set(randomNamesFromRows(candidateRows, count));
      setSelected((current) => {
        const next = replaceAll ? new Set() : new Set(current);
        if (replaceCandidates) {
          candidateRows.forEach((row) => next.delete(row.name));
        }
        picks.forEach((name) => next.add(name));
        return next;
      });
    };
    const randomSelectAll = () => chooseRandomRows(rows, randomPickCount, { replaceAll: true });
    const randomHalfAll = () => chooseRandomRows(rows, halfCount(rows.length), { replaceAll: true });
    const folderRowsFor = (folderKey) => rows.filter((row) => folderLabel(row) === folderKey);
    const folderRandomValue = (folderKey, total) => {
      const value = folderRandomCounts[folderKey];
      return value === undefined ? String(halfCount(total)) : value;
    };
    const updateFolderRandomCount = (folderKey, value) => {
      setFolderRandomCounts((current) => ({ ...current, [folderKey]: value }));
    };
    const randomSelectFolder = (folderKey, rawCount) => {
      const folderRows = folderRowsFor(folderKey);
      chooseRandomRows(folderRows, rawCount, { replaceCandidates: true });
    };
    const folderStats = (folderKey) => {
      const folderRows = folderRowsFor(folderKey);
      const selectedCount = folderRows.filter((row) => selected.has(row.name)).length;
      return { total: folderRows.length, selected: selectedCount };
    };
    const setFolderSelected = (folderKey, checked) => {
      setSelected((current) => {
        const next = new Set(current);
        rows.filter((row) => folderLabel(row) === folderKey).forEach((row) => {
          if (checked) {
            next.add(row.name);
          } else {
            next.delete(row.name);
          }
        });
        return next;
      });
    };
    const toggleTrainingTarget = (target, checked) => {
      setTrainingTargets((current) => {
        const next = new Set(current);
        if (checked) {
          next.add(target);
        } else {
          next.delete(target);
        }
        return next;
      });
    };
    const selectedTrainingTargetCount = addToTraining ? trainingTargets.size + (includeManualTarget ? 1 : 0) : 0;

    return h(
      React.Fragment,
      null,
      h(
        "article",
        { className: "panel stitching-builder-panel" },
        h(ProcessSteps, { tone: "warm", steps: ["Assign speaker labels", "Randomize selected clips", "Inspect RTTM timing"] }),
        projectNames.length ? h("datalist", { id: "stitching_project_names" }, projectNames.map((name) => h("option", { key: name, value: name }))) : null,
        h(
          "div",
          { className: "summary-grid" },
          h(SummaryCard, { title: "Audio Available" }, h("p", null, h("strong", null, rows.length), " file(s)")),
          h(SummaryCard, { title: "Selected" }, h("p", null, h("strong", null, selected.size), " clip(s)"), h("p", { className: "row-note" }, "Each selected clip becomes one RTTM segment.")),
          h(SummaryCard, { title: "Stitched Outputs" }, h("p", null, h("strong", null, summary.total || 0), " run(s)"), h("p", { className: "row-note" }, `${summary.training_added || 0} added to training`)),
          h(SummaryCard, { title: "Latest Status" }, stitchedRows[0] ? h(React.Fragment, null, h("p", null, h(StatusPill, { status: stitchedRows[0].status || "unknown" })), h("p", { className: "row-note" }, stitchedRows[0].name)) : h("p", null, "No run yet."))
        ),
        h(
          "form",
          {
            method: "post",
            action: routes.stitchAudio,
            "data-loading-message": "Creating stitched WAV and RTTM...",
            onSubmit: (event) => {
              if (selected.size < 2) {
                event.preventDefault();
                window.alert("Select at least two audio files to stitch.");
                return;
              }
              const manualProjectField = event.currentTarget.elements.namedItem("project_name");
              const manualProjectName = manualProjectField && "value" in manualProjectField ? text(manualProjectField.value).trim() : "";
              if (addToTraining && trainingTargets.size === 0 && (!includeManualTarget || !manualProjectName)) {
                event.preventDefault();
                window.alert("Choose at least one existing fine-tuning project or enter an additional project name.");
              }
            },
          },
          selectedRows.flatMap((row) => [
            h("input", { key: `${row.name}-audio`, type: "hidden", name: "selected_audio", value: row.name }),
            h("input", { key: `${row.name}-speaker`, type: "hidden", name: "speaker_labels", value: speakerForRow(row) }),
          ]),
          addToTraining ? Array.from(trainingTargets).map((target) => h("input", { key: `training-${target}`, type: "hidden", name: "training_targets", value: target, readOnly: true })) : null,
          h(
            "div",
            { className: "inline-3" },
            h(Field, { id: "stitch_name", label: "Output name", note: "Files are written under stitched/<run>/." }, h(TextInput, { id: "stitch_name", name: "stitch_name", placeholder: "child-vs-adult-randomized" })),
            h(Field, { id: "stitch_seed", label: "Random seed", note: "Leave blank for a new random order." }, h(TextInput, { id: "stitch_seed", name: "stitch_seed", placeholder: "optional" })),
            h(
              "div",
              { className: "stitching-toggles" },
              h("label", { className: "checkbox-row" }, h("input", { type: "checkbox", name: "add_to_training", checked: addToTraining, onChange: (event) => setAddToTraining(event.target.checked) }), " Add stitched pair to training"),
              h("label", { className: "checkbox-row" }, h("input", { type: "checkbox", name: "stitch_use_slurm" }), " Use Slurm for this stitch")
            )
          ),
          h(
            "details",
            { className: "details-box compact-details", open: true },
            h("summary", null, "Fine-Tuning Targets"),
            h(
              "div",
              null,
              projectOptions.length
                ? h(
                    React.Fragment,
                    null,
                    h("div", { className: "selection-toolbar compact-toolbar" }, h("button", { className: "secondary", type: "button", disabled: !addToTraining, onClick: () => setTrainingTargets(new Set(projectOptions.map((project) => project.key))) }, "Select all existing"), h("button", { className: "ghost", type: "button", disabled: !addToTraining || trainingTargets.size === 0, onClick: () => setTrainingTargets(new Set()) }, "Clear existing"), h("span", { className: "field-status" }, `${selectedTrainingTargetCount} target(s) selected`)),
                    h(
                      "div",
                      { className: "training-target-checklist" },
                      projectOptions.map((project) =>
                        h(
                          "label",
                          { key: project.key, className: "checkbox-row file-choice" },
                          h("input", {
                            type: "checkbox",
                            checked: trainingTargets.has(project.key),
                            disabled: !addToTraining,
                            onChange: (event) => toggleTrainingTarget(project.key, event.target.checked),
                          }),
                          h("span", null, h("strong", null, project.label), h("small", null, `${project.sampleCount} sample(s)`))
                        )
                      )
                    )
                  )
                : h("p", { className: "field-status" }, "No existing fine-tuning projects yet. Use the additional target below to create one when the stitched sample is saved."),
              h(
                "label",
                { className: "checkbox-row" },
                h("input", { type: "checkbox", checked: includeManualTarget, disabled: !addToTraining, onChange: (event) => setIncludeManualTarget(event.target.checked) }),
                projectOptions.length ? " Also send to another or new project" : " Send to new project"
              ),
              includeManualTarget
                ? h(
                    "div",
                    { className: "inline" },
                    h(Field, { id: "stitch_backend", label: "Additional backend" }, h(SelectInput, { id: "stitch_backend", name: "fine_tuning_backend", defaultValue: "pyannote", extra: { disabled: !addToTraining } }, h(Option, { value: "pyannote" }, "pyannote"), h(Option, { value: "nemo" }, "NeMo"), h(Option, { value: "both" }, "NeMo + pyannote"))),
                    h(Field, { id: "stitch_project_name", label: "Additional project name", note: "Can be an existing slug or a new project." }, h("input", { id: "stitch_project_name", name: "project_name", list: "stitching_project_names", defaultValue: projectOptions.length ? "" : defaultProject, placeholder: "stitched-site-training", disabled: !addToTraining }))
                  )
                : null
            )
          ),
          folderChoices.length
            ? h(
                "details",
                { className: "details-box compact-details", open: true },
                h("summary", null, "Speaker Labels By Folder"),
                h(
                  "p",
                  { className: "row-note" },
                  "Set a speaker name per folder. Use ",
                  h("strong", null, "Select All"),
                  " to grab every clip, or ",
                  h("strong", null, "Random Pick"),
                  " to keep only N. Order is randomized at submit."
                ),
                h(
                  "div",
                  { className: "speaker-assignment-grid" },
                  folderChoices.map((folder) => {
                    const stats = folderStats(folder.key);
                    return h(
                      "div",
                      { key: folder.key, className: "folder-stitch-control" },
                      h(
                        Field,
                        { id: `folder_speaker_${dialogIdFor(folder.key)}`, label: `${folder.label} (${folder.count})` },
                        h("input", {
                          id: `folder_speaker_${dialogIdFor(folder.key)}`,
                          type: "text",
                          value: folderSpeakers[folder.key] || "Speaker_0",
                          onChange: (event) => updateSpeakerForFolder(folder.key, event.target.value),
                        })
                      ),
                      h("p", { className: "field-status" }, `${stats.selected} of ${stats.total} file(s) selected`),
                      h(
                        "div",
                        { className: "button-row compact-toolbar" },
                        h("button", { className: "secondary", type: "button", onClick: () => setFolderSelected(folder.key, true), disabled: stats.total > 0 && stats.selected === stats.total }, "Select all"),
                        h("button", { className: "ghost", type: "button", onClick: () => setFolderSelected(folder.key, false), disabled: stats.selected === 0 }, "Clear")
                      ),
                      h(
                        "div",
                        { className: "random-pick-control" },
                        h(
                          "label",
                          { className: "compact-number-field" },
                          h("span", null, "Random count"),
                          h("input", {
                            type: "number",
                            min: "0",
                            max: stats.total,
                            value: folderRandomValue(folder.key, stats.total),
                            onChange: (event) => updateFolderRandomCount(folder.key, event.target.value),
                            "aria-label": `Random file count for ${folder.label}`,
                          })
                        ),
                        h("button", { className: "secondary", type: "button", onClick: () => randomSelectFolder(folder.key, folderRandomValue(folder.key, stats.total)), disabled: stats.total === 0 }, "Random Pick"),
                        h("button", { className: "ghost", type: "button", onClick: () => randomSelectFolder(folder.key, halfCount(stats.total)), disabled: stats.total === 0 }, "Random Half")
                      )
                    );
                  })
                )
              )
            : null,
          h(
            "div",
            { className: "selection-toolbar stitching-selection-toolbar" },
            h("button", { className: "secondary", type: "button", onClick: selectAll }, "Select all"),
            h("button", { className: "ghost", type: "button", onClick: clearSelected }, "Clear"),
            h(
              "label",
              { className: "compact-number-field" },
              h("span", null, "Random count"),
              h("input", {
                type: "number",
                min: "0",
                max: rows.length,
                value: randomPickCount,
                placeholder: String(halfCount(rows.length)),
                onChange: (event) => setRandomPickCount(event.target.value),
                "aria-label": "Random file count for all audio",
              })
            ),
            h("button", { className: "secondary", type: "button", onClick: randomSelectAll, disabled: rows.length === 0 }, "Random Select"),
            h("button", { className: "ghost", type: "button", onClick: randomHalfAll, disabled: rows.length === 0 }, "Random Half"),
            h("span", { className: "field-status" }, `${selected.size} file(s) selected`)
          ),
          h(DataTable, {
            className: "compact-table stitching-audio-table",
            headers: ["Select", "#", "Folder", "Audio File", "Speaker In RTTM"],
            rows,
            emptyText: h("div", { className: "empty-state" }, h("strong", null, "No source audio files yet."), "Upload files in Media Library or convert YouTube links first."),
            renderRow: (row, index) =>
              h(
                "tr",
                { key: row.name, className: selected.has(row.name) ? "is-selected" : "" },
                h("td", null, h("input", { type: "checkbox", checked: selected.has(row.name), "data-check-group": "stitch-audio", onChange: (event) => toggleRow(row.name, event.target.checked) })),
                h("td", null, index + 1),
                h("td", null, folderLabel(row)),
                h("td", { title: row.name }, h("strong", { className: "file-name" }, row.fileName || row.name)),
                h("td", null, h("input", { type: "text", value: speakerForRow(row), onChange: (event) => updateSpeakerForFile(row.name, event.target.value), "aria-label": `Speaker label for ${row.name}` }))
              ),
          }),
          h("p", { className: "field-status" }, selected.size, " file(s) selected for stitching."),
          h("div", { className: "button-row" }, h("button", { type: "submit", disabled: selected.size < 2 }, "Randomize And Stitch Selected"))
        )
      ),
      h(
        "article",
        { className: "panel" },
        h("div", { className: "panel-head" }, h("div", null, h("h2", null, "Stitched Outputs"), h("p", null, "Open Inspect to verify segment start/end times against the stitched WAV and review page."))),
        h(StitchedRunsTable, { rows: stitchedRows })
      )
    );
  }

  function SlurmQueueTracker({ queue = {}, metadata = {} }) {
    const jobId = queue.job_id || metadata.slurm_job_id || "";
    const localPid = metadata.pid ? String(metadata.pid) : "";
    // When state_source is "sacct" the job has already left the live queue —
    // show timing/exit fields instead of position counts, which are null for finished jobs.
    const isTerminal = queue.state_source === "sacct";
    const samePartitionPosition = queue.queue_position_same_partition;
    const resourceSummary = [
      queue.partition ? `partition ${queue.partition}` : "",
      queue.nodes ? `${queue.nodes} node(s)` : "",
      queue.node_count && !queue.nodes ? `${queue.node_count} node(s)` : "",
      queue.cpus ? `${queue.cpus} CPU(s)` : "",
      queue.gres ? `GRES ${queue.gres}` : "",
    ].filter(Boolean).join(" / ");
    const fetchedAt = queue.fetched_at_utc ? text(queue.fetched_at_utc).replace("T", " ").replace(/\..*$/, "") : "";
    const fmtUtc = (v) => (v && v !== "Unknown" ? text(v).replace("T", " ").replace(/\..*$/, "") : null);
    return h(
      "article",
      { className: "queue-tracker" },
      h(
        "div",
        null,
        h("h3", null, "Slurm Queue Tracker"),
        isTerminal
          ? h("p", null, "This job has left the live queue — details below are from ", h("code", null, "sacct"), ".")
          : h("p", null, "Updates live while this run is submitted or running.")
      ),
      jobId
        ? h(
            "div",
            { className: "queue-tracker-grid" },
            h("p", null, h("strong", null, "Job ID: "), jobId),
            h("p", null, h("strong", null, "State: "), h(StatusPill, { status: queue.state || "unknown" })),
            queue.exit_code
              ? h("p", null, h("strong", null, "Exit code: "), h("code", null, queue.exit_code))
              : null,
            isTerminal
              ? null
              : h("p", null, h("strong", null, "Queue position: "), queue.queue_position === 0 ? "running now" : queue.queue_position ?? "not in queue"),
            isTerminal || samePartitionPosition === undefined || samePartitionPosition === null
              ? null
              : h("p", null, h("strong", null, "Partition position: "), samePartitionPosition === 0 ? "running now" : samePartitionPosition),
            isTerminal
              ? null
              : h("p", null, h("strong", null, "Jobs ahead: "), queue.jobs_ahead ?? "unknown"),
            isTerminal || queue.jobs_ahead_same_partition === undefined || queue.jobs_ahead_same_partition === null
              ? null
              : h("p", null, h("strong", null, "Ahead in partition: "), queue.jobs_ahead_same_partition),
            fmtUtc(queue.started_at)
              ? h("p", null, h("strong", null, "Started: "), fmtUtc(queue.started_at), " UTC")
              : null,
            fmtUtc(queue.ended_at)
              ? h("p", null, h("strong", null, "Ended: "), fmtUtc(queue.ended_at), " UTC")
              : null,
            resourceSummary ? h("p", null, h("strong", null, "Resources: "), resourceSummary) : null,
            queue.time_used || queue.time_left || queue.time_limit
              ? h(
                  "p",
                  null,
                  h("strong", null, "Runtime: "),
                  queue.time_used || "0:00",
                  queue.time_left
                    ? ` elapsed / ${queue.time_left} left`
                    : queue.time_limit
                      ? ` elapsed / ${queue.time_limit} limit`
                      : " elapsed"
                )
              : null,
            queue.name || queue.user ? h("p", null, h("strong", null, "Job: "), [queue.name, queue.user].filter(Boolean).join(" / ")) : null,
            !isTerminal
              ? h("p", null, h("strong", null, "Reason: "), queue.reason || queue.message || "none reported")
              : null,
            !isTerminal
              ? h("p", null, h("strong", null, "Estimated start: "), queue.estimated_start || "not reported")
              : null,
            isTerminal && queue.message
              ? h("p", null, h("strong", null, "Note: "), queue.message)
              : null,
            fetchedAt ? h("p", null, h("strong", null, "Checked: "), fetchedAt, " UTC", queue.state_source ? ` via ${queue.state_source}` : "") : null
          )
        : localPid
          ? h("p", { className: "empty" }, "This run was launched as a local background process, so it doesn't have a Slurm queue position. Local PID: ", localPid)
          : h("p", { className: "empty" }, "No Slurm job ID attached to this run yet. Submit the run and refresh to see queue details.")
    );
  }

  function FileSearchPanel({ title = "Find a file across runs", placeholder = "e.g. 017_call" }) {
    // Cross-run file index search. Hits /api/file-search?q=... and groups
    // matches by stem so the user can see every artifact (audio, srt, log,
    // review html) tied to a given file name.
    const url = (state.routes && state.routes.fileSearch) || "/api/file-search";
    const [query, setQuery] = React.useState("");
    const [results, setResults] = React.useState([]);
    const [loading, setLoading] = React.useState(false);

    React.useEffect(() => {
      if (!query) {
        setResults([]);
        return;
      }
      let cancelled = false;
      const timer = window.setTimeout(() => {
        if (typeof window.fetch !== "function") return;
        setLoading(true);
        window
          .fetch(`${url}?q=${encodeURIComponent(query)}`, { cache: "no-store", headers: { Accept: "application/json" } })
          .then((response) => response.json())
          .then((payload) => {
            if (cancelled) return;
            setResults(payload.results || []);
            setLoading(false);
          })
          .catch(() => {
            if (!cancelled) setLoading(false);
          });
      }, 250);
      return () => {
        cancelled = true;
        window.clearTimeout(timer);
      };
    }, [url, query]);

    return h(
      "article",
      { className: "panel file-search-panel" },
      h(
        "div",
        { className: "panel-head" },
        h("div", null, h("h2", null, title), h("p", null, "Type any part of an audio file name to see every run, output, and source file that mentions it."))
      ),
      h("input", {
        type: "search",
        className: "file-search-input",
        value: query,
        placeholder,
        onChange: (e) => setQuery(e.target.value),
        "aria-label": "File search",
      }),
      query && loading ? h("p", { className: "footer-note" }, "Searching...") : null,
      query && !loading && results.length === 0
        ? h("p", { className: "empty" }, `No matches for "${query}".`)
        : null,
      results.length > 0
        ? h(
            "ul",
            { className: "file-search-results" },
            results.map((group) =>
              h(
                "li",
                { key: group.stem, className: "file-search-group" },
                h("span", { className: "file-search-stem" }, group.stem),
                h("span", { className: "file-search-count" }, `${group.match_count} match(es)`),
                h(
                  "ul",
                  { className: "file-search-matches" },
                  (group.matches || []).slice(0, 12).map((m, i) =>
                    h(
                      "li",
                      { key: `${m.rel_path}-${i}`, className: `file-search-match file-search-match--${m.kind}` },
                      m.href ? h("a", { href: m.href }, m.rel_path) : m.rel_path,
                      h("span", { className: "file-search-kind" }, ` (${m.kind}${m.run_name ? ` · ${m.run_name}` : ""})`)
                    )
                  )
                )
              )
            )
          )
        : null
    );
  }

  function RuntimeEstimateBadge({ group, backend }) {
    // Listens to ``change`` events on checkboxes in ``group`` and (debounced)
    // fetches /api/runtime-estimate so the user can see "~12 min" before
    // submitting. When the estimate engine has too little history, renders
    // a quiet "ETA unknown" tag instead of bluffing a number.
    const url = (state.routes && state.routes.runtimeEstimate) || "/api/runtime-estimate";
    const [estimate, setEstimate] = React.useState(null);
    const [loading, setLoading] = React.useState(false);

    React.useEffect(() => {
      let cancelled = false;
      let timer = null;
      const update = () => {
        if (cancelled) return;
        const boxes = document.querySelectorAll(
          `input[type="checkbox"][data-check-group="${group}"]:checked`
        );
        const files = Array.from(boxes)
          .map((b) => (b instanceof HTMLInputElement ? b.value : ""))
          .filter(Boolean);
        if (files.length === 0) {
          setEstimate(null);
          setLoading(false);
          return;
        }
        if (typeof window.fetch !== "function") return;
        const params = new URLSearchParams();
        params.set("backend", backend || "nemo");
        files.forEach((f) => params.append("audio_files", f));
        setLoading(true);
        window
          .fetch(`${url}?${params.toString()}`, { cache: "no-store", headers: { Accept: "application/json" } })
          .then((response) => response.json())
          .then((payload) => {
            if (cancelled) return;
            setEstimate(payload);
            setLoading(false);
          })
          .catch(() => {
            if (!cancelled) setLoading(false);
          });
      };

      const onChange = (event) => {
        const t = event.target;
        if (!(t instanceof HTMLInputElement)) return;
        if (t.getAttribute && t.getAttribute("data-check-group") !== group) return;
        if (timer) window.clearTimeout(timer);
        timer = window.setTimeout(update, 350);
      };

      // Run once on mount in case checkboxes were already checked.
      update();
      document.addEventListener("change", onChange);
      return () => {
        cancelled = true;
        if (timer) window.clearTimeout(timer);
        document.removeEventListener("change", onChange);
      };
    }, [url, group, backend]);

    if (!estimate) return null;
    if (!estimate.available) {
      return h(
        "span",
        { className: "runtime-estimate-badge runtime-estimate-badge--unknown", title: estimate.message || "" },
        loading ? "Estimating..." : "ETA unknown"
      );
    }
    const detail = estimate.based_on_runs && estimate.based_on_files
      ? ` · based on ${estimate.based_on_runs} runs / ${estimate.based_on_files} files`
      : "";
    return h(
      "span",
      { className: "runtime-estimate-badge", title: `Ratio ${estimate.ratio || ""}, ${estimate.total_audio_seconds || 0} s of audio` },
      `ETA ${estimate.estimate_label || ""}`,
      h("span", { className: "runtime-estimate-badge-detail" }, detail)
    );
  }

  function LatestRunQueueTracker({ latestRun, activeRuns, emptyText, storageKey }) {
    // The "unified" run-status panel: collapses to the same single-run tracker
    // every page used to render. When `activeRuns` carries 2+ runs, we render
    // a tab strip above so the user can flip between them. Tab choice is
    // persisted in sessionStorage so navigating away and back keeps it.
    const runs = Array.isArray(activeRuns) && activeRuns.length > 0
      ? activeRuns
      : (latestRun ? [latestRun] : []);

    const showTabs = runs.length >= 2;
    const effectiveStorageKey = storageKey || "run-status-tab";

    const [selectedIndex, setSelectedIndex] = React.useState(() => {
      if (!showTabs || typeof window === "undefined") return 0;
      try {
        const stored = window.sessionStorage.getItem(effectiveStorageKey);
        const parsed = stored == null ? 0 : parseInt(stored, 10);
        return Number.isFinite(parsed) && parsed >= 0 && parsed < runs.length ? parsed : 0;
      } catch (_err) {
        return 0;
      }
    });

    React.useEffect(() => {
      if (!showTabs || typeof window === "undefined") return;
      try {
        window.sessionStorage.setItem(effectiveStorageKey, String(selectedIndex));
      } catch (_err) {
        /* ignore storage failures; tab still works in-memory */
      }
    }, [selectedIndex, showTabs, effectiveStorageKey]);

    if (runs.length === 0) {
      return h(
        "article",
        { className: "queue-tracker" },
        h("div", null, h("h3", null, "Slurm Queue Tracker"), h("p", null, "Updates live while this run is submitted or running.")),
        h("p", { className: "empty" }, emptyText || "No site run has been submitted yet.")
      );
    }

    const safeIndex = Math.min(selectedIndex, runs.length - 1);
    const selected = runs[safeIndex];

    return h(
      React.Fragment,
      null,
      showTabs && h(
        "div",
        { className: "run-tab-strip", role: "tablist", "aria-label": "Active runs" },
        runs.map((run, index) => h(
          "button",
          {
            key: (run && run.name) || index,
            type: "button",
            role: "tab",
            "aria-selected": index === safeIndex,
            className: "run-tab" + (index === safeIndex ? " run-tab--active" : ""),
            onClick: () => setSelectedIndex(index),
          },
          h("span", { className: "run-tab-name" }, (run && run.name) || `Run ${index + 1}`),
          run && run.status ? h(StatusPill, { status: run.status }) : null
        ))
      ),
      h(SlurmQueueTracker, { queue: (selected && selected.slurmQueue) || {}, metadata: (selected && selected.metadata) || {} })
    );
  }

  function ClusterQueuePanel({ title = "Cluster Queue", refreshIntervalMs = 5000 }) {
    const url = (state.routes && state.routes.clusterQueue) || "/api/cluster-queue";
    const [snapshot, setSnapshot] = React.useState({ available: true, jobs: [], message: null });
    const [loading, setLoading] = React.useState(true);
    const [error, setError] = React.useState(null);
    const [stateFilter, setStateFilter] = React.useState("PD,R,CG");
    const [partitionFilter, setPartitionFilter] = React.useState("");
    const [userFilter, setUserFilter] = React.useState("");

    React.useEffect(() => {
      let cancelled = false;
      let timer = null;
      const run = () => {
        if (typeof window.fetch !== "function") {
          return;
        }
        const params = new URLSearchParams();
        if (stateFilter) params.set("state", stateFilter);
        if (partitionFilter) params.set("partition", partitionFilter);
        if (userFilter) params.set("user", userFilter);
        const target = params.toString() ? `${url}?${params.toString()}` : url;
        window
          .fetch(target, { cache: "no-store", headers: { Accept: "application/json" } })
          .then((response) => {
            if (!response.ok) {
              throw new Error(`Cluster queue request failed with ${response.status}`);
            }
            return response.json();
          })
          .then((payload) => {
            if (cancelled) return;
            setSnapshot(payload || { available: false, jobs: [] });
            setError(null);
            setLoading(false);
          })
          .catch((err) => {
            if (cancelled) return;
            setError(err.message || "Cluster queue fetch failed.");
            setLoading(false);
          });
      };
      run();
      timer = window.setInterval(run, refreshIntervalMs);
      return () => {
        cancelled = true;
        if (timer) window.clearInterval(timer);
      };
    }, [url, stateFilter, partitionFilter, userFilter, refreshIntervalMs]);

    const jobs = snapshot.jobs || [];
    const me = snapshot.current_user || "";
    const fetchedAt = snapshot.fetched_at_utc ? snapshot.fetched_at_utc.replace("T", " ").replace(/\..*$/, "") : "";

    const renderRow = (job) => h(
      "tr",
      { key: job.job_id, className: job.is_self ? "cluster-queue-row cluster-queue-row--self" : "cluster-queue-row" },
      h("td", null, job.job_id),
      h("td", null, job.is_self ? h("strong", null, job.user, " (you)") : job.user),
      h("td", null, h(StatusPill, { status: job.state })),
      h("td", null, job.partition),
      h("td", null, job.time_used),
      h("td", null, job.time_left),
      h("td", null, job.nodes),
      h("td", null, job.cpus),
      h("td", { title: job.name }, job.name),
      h("td", { title: job.reason }, job.reason)
    );

    return h(
      "article",
      { className: "panel cluster-queue-panel" },
      h(
        "div",
        { className: "panel-head" },
        h("div", null, h("h2", null, title), h("p", null, "Live view of every job currently known to ", h("code", null, "squeue"), ". Your jobs are highlighted."))
      ),
      h(
        "div",
        { className: "cluster-queue-controls" },
        h(
          "label",
          { className: "field" },
          h("span", { className: "label" }, "States"),
          h(
            "select",
            { value: stateFilter, onChange: (e) => setStateFilter(e.target.value) },
            h("option", { value: "PD,R,CG" }, "Pending + Running + Completing"),
            h("option", { value: "PD" }, "Pending only"),
            h("option", { value: "R" }, "Running only"),
            h("option", { value: "PD,R" }, "Pending + Running")
          )
        ),
        h(
          "label",
          { className: "field" },
          h("span", { className: "label" }, "Partition contains"),
          h("input", { type: "text", value: partitionFilter, onChange: (e) => setPartitionFilter(e.target.value), placeholder: "e.g. gpu" })
        ),
        h(
          "label",
          { className: "field" },
          h("span", { className: "label" }, "User contains"),
          h("input", { type: "text", value: userFilter, onChange: (e) => setUserFilter(e.target.value), placeholder: me ? `e.g. ${me}` : "username" })
        )
      ),
      error
        ? h("p", { className: "empty" }, error)
        : !snapshot.available
          ? h("p", { className: "empty" }, snapshot.message || "squeue is not available on this machine.")
          : loading && jobs.length === 0
            ? h("p", { className: "empty" }, "Loading cluster queue...")
            : jobs.length === 0
              ? h("p", { className: "empty" }, snapshot.message || "Cluster queue is empty for the selected filters.")
              : h(DataTable, {
                  headers: ["Job ID", "User", "State", "Partition", "Elapsed", "Time left", "Nodes", "CPUs", "Name", "Reason"],
                  rows: jobs,
                  renderRow,
                }),
      h(
        "p",
        { className: "footer-note" },
        fetchedAt
          ? `Last refreshed ${fetchedAt} UTC. Refreshes every ${Math.round(refreshIntervalMs / 1000)} s.`
          : `Refreshes every ${Math.round(refreshIntervalMs / 1000)} s.`
      )
    );
  }


  function YoutubeRunPanel({ latestRun }) {
    if (!latestRun) {
      return h("p", { className: "empty" }, "No site-launched YouTube conversion run exists yet.");
    }
    const summary = latestRun.summary || {};
    const metadata = latestRun.metadata || {};
    return h(
      React.Fragment,
      null,
      h(
        "div",
        { className: "summary-grid" },
        h(SummaryCard, { title: "Run Status" }, h("p", null, h(StatusPill, { status: latestRun.status })), h("p", null, h("strong", null, "Run folder: "), latestRun.name), h("p", null, h("strong", null, "Launch mode: "), metadata.mode || "selected"), h("p", null, h("strong", null, "Selected links: "), metadata.selected_url_count || "0")),
        h(SummaryCard, { title: "Result Counts" }, h("ul", null, h("li", null, `Downloaded: ${summary.downloaded || 0}`), h("li", null, `Skipped: ${summary.skipped || 0}`), h("li", null, `Retry needed: ${summary.retry || 0}`), h("li", null, `No data: ${summary.no_data || 0}`), h("li", null, `Removed from queue: ${latestRun.removedFromQueue || 0}`))),
        h(SummaryCard, { title: "Storage" }, h("ul", null, h("li", null, "Downloaded links created or refreshed WAV files under ", h("code", null, metadata.audio_output_dir || "audio_in/youtube_links"), "."), h("li", null, "Retry-needed links stayed in the queue."), h("li", null, "No-data links were removed because public audio was not reachable.")))
      ),
      h(LatestRunQueueTracker, { latestRun }),
      h(LinkList, { links: latestRun.artifactLinks, empty: "No artifact files are available yet." }),
      h("details", { className: "details-box", open: true }, h("summary", null, "Conversion Report"), h("div", null, h(DataTable, { headers: ["URL", "Status", "Queue Action", "Summary"], rows: latestRun.resolutionPreview || [], emptyText: "This run has not written a conversion report yet.", renderRow: (row, index) => h("tr", { key: `${row.url}-${index}` }, h("td", null, row.url || ""), h("td", null, row.status || row.result || ""), h("td", null, text(row.queue_action || "kept_in_queue").replace(/_/g, " ")), h("td", null, row.summary || row.note || "")) }))),
      h("div", { className: "log-grid" }, h("details", { className: "details-box" }, h("summary", null, "Activity Log"), h("div", null, h("div", { className: "mono-box" }, latestRun.stdoutTail || "No activity log entries yet."))), h("details", { className: "details-box" }, h("summary", null, "Error Log"), h("div", null, h("div", { className: "mono-box" }, latestRun.stderrTail || "No wrapper-level errors were written."))))
    );
  }

  function YoutubeQueueTable({ rows }) {
    return h(DataTable, {
      className: "youtube-queue-table",
      headers: ["Select", "#", "Queue Status", "Queued URL", "Notes", "Remove"],
      rows: rows || [],
      emptyText: h("div", { className: "empty-state" }, h("strong", null, "No queued links yet."), "Paste one or more YouTube URLs in the box above to add them to the conversion queue."),
      renderRow: (row) =>
        h(
          "tr",
          { key: row.url },
          h("td", null, h("input", { type: "checkbox", name: "selected_urls", value: row.url, "data-check-group": "youtube-queue", "data-ready": row.selection_ready || "yes" })),
          h("td", null, row.index),
          h("td", null, h("span", { className: `queue-state ${row.state_class || "ready"}` }, row.queue_state || "ready to convert")),
          h("td", null, h("span", { className: "url-text" }, row.url)),
          h("td", null, row.detail),
          h(
            "td",
            null,
            h(
              "button",
              {
                className: "secondary danger btn-sm",
                type: "submit",
                formAction: routes.deleteYoutubeLink,
                formMethod: "post",
                name: "youtube_url",
                value: row.url,
                onClick: (event) => {
                  const extra = row.queue_state === "already converted" ? " The converted WAV and generated artifacts will also be deleted, then remaining files will be renumbered." : "";
                  if (!window.confirm(`Remove this YouTube link from the queue?${extra}`)) {
                    event.preventDefault();
                  }
                },
              },
              "Remove"
            )
          )
        ),
    });
  }

  function YoutubePage() {
    const youtube = ctx.youtube || {};
    const summary = youtube.queueSummary || {};
    const latestRun = youtube.latestRun;
    return h(
      React.Fragment,
      null,
      h(
        "article",
        { className: "panel" },
        h(ProcessSteps, { steps: ["Paste links", "Select ready or retry links", "Convert to audio"] }),
        h(
          "div",
          { className: "split-grid" },
          h(
            "section",
            { className: "subpanel" },
            h("div", { className: "panel-head" }, h("div", null, h("h2", null, "Add Links"), h("p", null, "Paste one YouTube URL per line. Duplicate links are ignored automatically."))),
            h("form", { method: "post", action: routes.youtubeLinks }, h(Field, { id: "youtube_urls", label: "YouTube URLs" }, h("textarea", { id: "youtube_urls", name: "youtube_urls", placeholder: "https://youtu.be/example" })), h("p", { className: "footer-note" }, "New links are appended to ", h("code", null, "youtube_links.txt"), "."), h("p", null, h("button", { type: "submit" }, "Add links")))
          ),
          h(
            "section",
            { className: "subpanel" },
            h("div", { className: "panel-head" }, h("div", null, h("h2", null, "Queue Status"), h("p", null, "Use these counts to choose a single run, a bulk run, or a clean reset."))),
            h("div", { className: "summary-grid" }, h(SummaryCard, { title: "Total queued" }, h("p", null, h("strong", null, summary.total || 0), " link(s)")), h(SummaryCard, { title: "Ready right now" }, h("p", null, h("strong", null, summary.ready || 0), " new link(s)"), h("p", null, `${summary.retry || 0} link(s) are marked for retry.`)), h(SummaryCard, { title: "Already converted" }, h("p", null, h("strong", null, summary.converted || 0), " link(s)"), h("p", null, `${summary.no_data || 0} link(s) are marked as no public data.`)), h(SummaryCard, { title: "Latest run" }, h("p", null, h(StatusPill, { status: latestRun?.status || "stopped" })), h("p", null, latestRun?.name || "No site run yet."))),
            h(LatestRunQueueTracker, { latestRun, activeRuns: youtube.activeRuns, emptyText: "No YouTube audio conversion Slurm job has been submitted yet.", storageKey: "youtube-active-tab" }),
            h("form", { method: "post", action: routes.resetYoutube, onSubmit: (event) => { if (!window.confirm("This clears the YouTube queue, removes generated audio files from the youtube_links folder, and deletes old YouTube run logs. Continue?")) { event.preventDefault(); } } }, h("p", { className: "footer-note" }, "Use reset before a brand-new bulk batch."), h("div", { className: "button-row" }, h("button", { className: "danger", type: "submit" }, "Reset queue")))
          )
        )
      ),
      h(
        "article",
        { className: "panel" },
        h("div", { className: "panel-head" }, h("div", null, h("h2", null, "Conversion Queue"), h("p", null, "Single conversion means one checked row. Bulk conversion means the ready-selection shortcut or the full-queue button."))),
        h(
          "form",
          { method: "post", action: routes.convertYoutube },
          h(Field, { id: "youtube_audio_folder", label: "Target folder" }, h(AudioFolderSelect, { id: "youtube_audio_folder", name: "audio_folder", defaultValue: defaults.defaultYoutubeAudioFolder || "youtube_links" })),
          h(Field, { id: "youtube_new_audio_folder", label: "Optional new folder" }, h(TextInput, { id: "youtube_new_audio_folder", name: "new_audio_folder", placeholder: "youtube_batch_set" })),
          h("div", { className: "selection-toolbar" }, h("button", { className: "secondary", type: "button", "data-select-group": "youtube-queue", "data-select-mode": "ready" }, "Select ready + retry"), h("button", { className: "secondary", type: "button", "data-select-group": "youtube-queue", "data-select-mode": "all" }, "Select all"), h("button", { className: "ghost", type: "button", "data-select-group": "youtube-queue", "data-select-mode": "none" }, "Clear"), h("label", { className: "checkbox-row" }, h("input", { type: "checkbox", name: "youtube_force_redownload" }), " Force re-download")),
          h(YoutubeQueueTable, { rows: youtube.queueRows || [] }),
          h("p", { className: "field-status" }, h("span", { "data-selection-count": "youtube-queue" }, "0"), " link(s) selected for the next audio conversion run."),
          h("p", { className: "table-note" }, "Use ", h("strong", null, "Select ready + retry"), " for the normal case."),
          h("div", { className: "button-row" }, h("button", { type: "submit", name: "youtube_mode", value: "selected", "data-selection-submit": "youtube-queue" }, "Convert selected"), h("button", { className: "secondary", type: "submit", name: "youtube_mode", value: "all" }, "Convert all"))
        )
      ),
      h("article", { className: "panel" }, h("div", { className: "panel-head" }, h("div", null, h("h2", null, "More Details"), h("p", null, "Open logs, history, and queue exceptions only when needed."))), h("div", { className: "button-row" }, h("button", { className: "secondary", type: "button", "data-open-dialog": "youtube-run-dialog" }, "Latest run"), h("button", { className: "secondary", type: "button", "data-open-dialog": "youtube-history-dialog" }, "History"), h("button", { className: "secondary", type: "button", "data-open-dialog": "youtube-issues-dialog" }, "Queue issues")), h(Dialog, { id: "youtube-run-dialog", title: "Latest Audio Conversion Run", detail: "Newest result counts, artifact files, and log previews." }, h(YoutubeRunPanel, { latestRun })), h(Dialog, { id: "youtube-history-dialog", title: "Recent Conversion History", detail: "Confirm whether a link already produced audio or needs another attempt." }, h(DataTable, { headers: ["Status", "Title / Video", "Audio File", "Note", "Last Attempt"], rows: youtube.history || [], emptyText: "No conversion history yet.", renderRow: (row, index) => h("tr", { key: `${row.title}-${index}` }, h("td", null, row.status), h("td", null, row.title), h("td", null, row.audioHref ? h("a", { href: row.audioHref }, row.audioFile || "Open audio") : row.audioFile), h("td", null, row.note), h("td", null, row.lastAttempt)) })), h(Dialog, { id: "youtube-issues-dialog", title: "Queue Issues", detail: "Retryable downloader errors stay available. No-data links are separated." }, h(DataTable, { headers: ["URL", "Summary", "Queue Result", "Last Attempt"], rows: youtube.retryRows || [], emptyText: "No retry-needed URLs are recorded right now.", renderRow: (row, index) => h("tr", { key: `${row.url}-${index}` }, h("td", null, row.url), h("td", null, row.summary), h("td", null, row.queue_result || "kept"), h("td", null, row.when)) }), h("h3", null, "URLs With No Public Audio Data"), h(DataTable, { headers: ["URL", "Summary", "Queue Result", "Last Attempt"], rows: youtube.noDataRows || [], emptyText: "No no-data URLs are recorded right now.", renderRow: (row, index) => h("tr", { key: `${row.url}-${index}` }, h("td", null, row.url), h("td", null, row.summary), h("td", null, row.queue_result || "removed"), h("td", null, row.when)) })))
    );
  }

  function DiarizationRunPanel() {
    const diarization = ctx.diarization || {};
    const run = diarization.latestRun;
    const activeRuns = diarization.activeRuns || [];
    if (!run && !activeRuns.length) {
      return h(LatestRunQueueTracker, { latestRun: null, activeRuns, emptyText: "No site-launched diarization run exists yet.", storageKey: "diarization-run-panel-active-tab" });
    }
    if (!run) {
      return h(LatestRunQueueTracker, { latestRun: null, activeRuns, emptyText: "No site-launched diarization run exists yet.", storageKey: "diarization-run-panel-active-tab" });
    }
    const total = Math.max(Number(run.selectedCount || 0), 1);
    const metadata = run.metadata || {};
    const live = run.liveProgress || {};
    return h(
      React.Fragment,
      null,
      live.current_file
        ? h(
            "div",
            { className: "live-progress-banner" },
            h("span", { className: "live-progress-dot", "aria-hidden": true }),
            h(
              "span",
              { className: "live-progress-text" },
              `Currently processing ${live.current_index} of ${live.current_total}: `,
              h("code", null, live.current_file)
            )
          )
        : null,
      h(
        "div",
        { className: "summary-grid" },
        h(
          SummaryCard,
          { title: "Run Status" },
          h("p", null, h(StatusPill, { status: run.status })),
          h("p", null, h("strong", null, "Run folder: "), run.name),
          h("p", null, h("strong", null, "Backend: "), metadata.backend || "unknown"),
          metadata.model_label ? h("p", null, h("strong", null, "Model profile: "), metadata.model_label) : null,
          metadata.slurm_job_id ? h("p", null, h("strong", null, "Slurm job: "), metadata.slurm_job_id) : null,
          run.batchId ? h("p", null, h("strong", null, "Batch: "), run.batchId, metadata.batch_size_total ? ` (${(metadata.batch_index ?? 0) + 1} of ${metadata.batch_size_total})` : "") : null
        ),
        h(ProgressCard, { title: "Files Processed", current: run.completedCount, total, detail: "Updates after each file finishes." }),
        h(ProgressCard, { title: "Successful Files", current: run.succeededCount, total }),
        h(ProgressCard, { title: "Needs Review", current: Number(run.failedCount || 0) + Number(run.noSpeechCount || 0), total, detail: `No speech: ${run.noSpeechCount || 0}. Failed: ${run.failedCount || 0}.` })
      ),
      h(LatestRunQueueTracker, { latestRun: run, activeRuns, emptyText: "No diarization Slurm job has been submitted yet.", storageKey: "diarization-run-panel-active-tab" }),
      h("div", { className: "artifact-strip" }, h("strong", null, "Run files: "), h(LinkList, { links: run.artifactLinks })),
      h("p", { className: "table-note" }, h("a", { href: navItems.find((item) => item.path === "/uploads")?.href || "/uploads" }, "Open Media Library"), " to review diarized audio, transcripts, time files, review pages, flags, and logs."),
      h(
        "details",
        { className: "details-box" },
        h("summary", null, "Runtime Summary"),
        h("div", null, h(DataTable, { headers: ["Audio File", "Status", "Runtime (s)"], rows: run.summaryRows || [], emptyText: "The runtime summary file has not been written yet.", renderRow: (row, index) => h("tr", { key: `${row.audio_file}-${index}` }, h("td", null, row.audio_file), h("td", null, row.status), h("td", null, row.runtime_seconds)) }))
      ),
      h(
        "details",
        { className: "details-box" },
        h("summary", null, "Created Files And Item Logs"),
        h("div", null, h("p", { className: "row-note" }, "Inspect each audio item here to see created outputs, transcript previews, speaker-time previews, and per-file logs without downloading them."), h(DiarizationRunItemsTable, { items: run.items || [], runName: run.name || "" }))
      ),
      h(
        "div",
        { className: "log-grid" },
        h("details", { className: "details-box" }, h("summary", null, "Processing Log"), h("div", null, h("div", { className: "mono-box" }, run.stdoutTail || "No processing log entries yet."))),
        h("details", { className: "details-box" }, h("summary", null, "Error Log"), h("div", null, h("div", { className: "mono-box" }, run.stderrTail || "No errors have been written.")))
      )
    );
  }

  function DiarizationModelSettings({ preferences }) {
    const modelOptions = diarizationModelChoiceOptions();
    const defaultModelKey = preferences.default_diarization_model_key || preferences.default_backend || "nemo";
    return h(
      "article",
      { className: "panel" },
      h("div", { className: "panel-head" }, h("div", null, h("h2", null, "Model Settings"), h("p", null, "This replaces the old separate Model Selection tab. Save defaults here, then launch runs from the same page."))),
      h(
        "form",
        { method: "post", action: routes.saveModels, "data-loading-message": "Saving model settings..." },
        h(
          "div",
          { className: "split-grid" },
          h("section", { className: "subpanel" }, h(Field, { id: "default_diarization_model_key", label: "Default diarization model", note: "Built-in backends and reusable fine-tuned checkpoints appear here once training artifacts exist." }, h(SelectInput, { id: "default_diarization_model_key", defaultValue: defaultModelKey }, modelOptions)), h(Field, { id: "whisper_model", label: "Default speech-to-text model", note: "The transcript links come from this Whisper model." }, h(TextInput, { id: "whisper_model", defaultValue: preferences.whisper_model || "" })), h(Field, { id: "whisper_batch_size", label: "Default batch size" }, h(TextInput, { id: "whisper_batch_size", defaultValue: preferences.whisper_batch_size || "" }))),
          h("section", { className: "subpanel" }, h(Field, { id: "pyannote_pipeline_model", label: "pyannote pipeline model" }, h(TextInput, { id: "pyannote_pipeline_model", defaultValue: preferences.pyannote_pipeline_model || "" })), h(Field, { id: "pyannote_segmentation_model", label: "pyannote segmentation model", note: "Default is the CallHome ZHO fine-tuned segmentation model. Leave blank for the pyannote pipeline default." }, h(TextInput, { id: "pyannote_segmentation_model", defaultValue: preferences.pyannote_segmentation_model || "" })), h("p", { className: "footer-note" }, "Fine-tuning-specific settings still live on the Fine-Tuning page."))
        ),
        h("p", null, h("button", { className: "secondary", type: "submit" }, "Save Model Settings"))
      )
    );
  }

  function DiarizationPage() {
    const preferences = ctx.preferences || {};
    const diarization = ctx.diarization || {};
    const rawFiles = diarization.audioRows || ctx.audioFiles || [];
    // Pre-stringify the per-model selection map once per row instead of inside
    // the renderRow callback. JSON.stringify on the same dict is wasteful when
    // a long file list re-renders for unrelated state changes; React diffing
    // doesn't catch identical-by-value strings, so memoizing the source array
    // keeps the cost paid only when the data actually changes.
    const files = React.useMemo(
      () =>
        rawFiles.map((row) => ({
          ...row,
          selectionByModelJson: JSON.stringify(row.selectionByModel || {}),
        })),
      [rawFiles]
    );
    const summary = diarization.librarySummary || {};
    const latestRun = diarization.latestRun;
    const modelCounts = summary.modelCounts || [];
    const selectedModelKey = diarization.selectedModelKey || preferences.default_diarization_model_key || preferences.default_backend || "nemo";
    const [folderFilter, setFolderFilter] = React.useState("all");
    const [rowLimit, setRowLimit] = React.useState(DEFAULT_FILE_VIEW_LIMIT);
    const sortedFiles = React.useMemo(
      () => sortFileRowsByFolder(files, fileViewFolderLabel, (row) => row.fileName || row.name),
      [files]
    );
    const folderChoices = React.useMemo(() => fileViewFolderChoices(sortedFiles, fileViewFolderLabel), [sortedFiles]);
    React.useEffect(() => {
      if (folderFilter !== "all" && !folderChoices.some((folder) => folder.key === folderFilter)) {
        setFolderFilter("all");
      }
    }, [folderFilter, folderChoices]);
    const folderFilteredFiles = folderFilter === "all"
      ? sortedFiles
      : sortedFiles.filter((row) => fileViewFolderLabel(row) === folderFilter);
    const renderedFiles = folderFilteredFiles.slice(0, Math.min(rowLimit, folderFilteredFiles.length));
    return h(
      React.Fragment,
      null,
      h(
        "article",
        { className: "panel diarization-panel" },
        h(ProcessSteps, { tone: "warm", steps: ["Choose model profiles", "Select ready or retry files", "Submit a Slurm run"] }),
        h(
          "div",
          { className: "summary-grid" },
          h(SummaryCard, { title: "Ready" }, h("p", null, h("strong", null, summary.ready || 0), " new file(s)")),
          h(SummaryCard, { title: "Completed" }, h("p", null, h("strong", null, summary.diarized || 0), " tracked file(s)")),
          h(SummaryCard, { title: "Retry" }, h("p", null, h("strong", null, summary.retry || 0), " failed file(s)")),
          h(SummaryCard, { title: "By Model" }, modelCounts.length ? h("ul", { className: "compact-list" }, modelCounts.map((item) => h("li", { key: item.key || item.backend || item.label }, `${item.label || item.backend}: ${item.diarized || 0} done, ${item.ready || 0} not run`))) : h("p", null, "No model tracking yet.")),
          h(SummaryCard, { title: "Latest" }, h("p", null, h(StatusPill, { status: latestRun?.status || "not started" })), h("p", { className: "run-name" }, latestRun?.name || "No site run yet."))
        ),
        h(
          "form",
          { method: "post", action: routes.runDiarization, "data-loading-message": "Submitting diarization job to Slurm..." },
          h(
            "div",
            { className: "inline-3" },
            h(
              Field,
              {
                id: "diarization_model_keys",
                label: "Diarization models",
                note: "Hold Ctrl or Shift to multi-select. Each selected model launches its own run; sibling runs share a batch id.",
              },
              h(
                SelectInput,
                {
                  id: "diarization_model_keys",
                  name: "diarization_model_keys",
                  defaultValue: [selectedModelKey],
                  extra: { multiple: true, size: Math.min(8, Math.max(3, diarizationModelOptions().length)) },
                },
                diarizationModelChoiceOptions()
              ),
              h(
                "div",
                { className: "selection-toolbar compact-toolbar" },
                h("button", { className: "secondary", type: "button", "data-multiselect-target": "diarization_model_keys", "data-multiselect-mode": "all" }, "All models"),
                h("button", { className: "ghost", type: "button", "data-multiselect-target": "diarization_model_keys", "data-multiselect-mode": "none" }, "Clear")
              )
            ),
            h(Field, { id: "diarization_whisper_model", label: "Speech-to-text model" }, h(TextInput, { id: "diarization_whisper_model", defaultValue: preferences.whisper_model || "" })),
            h(Field, { id: "diarization_language", label: "Language hint", note: "Optional. Leave blank when you are unsure." }, h(TextInput, { id: "diarization_language", placeholder: "example: en" }))
          ),
          h("details", { className: "details-box compact-details" }, h("summary", null, "Advanced Runtime Setting"), h("div", null, h(Field, { id: "diarization_batch_size", label: "Batch size" }, h(TextInput, { id: "diarization_batch_size", defaultValue: preferences.whisper_batch_size || "" })), h(Field, { id: "pyannote_hf_token", label: "Hugging Face token for pyannote", note: "Only needed when running pyannote. The token is passed to Slurm for this run and is not saved in site preferences." }, h(TextInput, { id: "pyannote_hf_token", type: "password", defaultValue: "", placeholder: "hf_..." })), h("p", { className: "footer-note" }, "The site submits GPU Slurm jobs, so the device is handled automatically."))),
          h("div", { className: "tracker-heading" }, h("h3", null, "Audio File Tracker"), h("p", null, "Select from files that are new or need another attempt. Completed files stay locked unless re-running is allowed.")),
          h(
            "div",
            { className: "selection-toolbar file-view-toolbar" },
            h("button", { className: "secondary", type: "button", "data-select-group": "diarization-audio", "data-select-mode": "ready", "data-ready-select": "diarization_model_keys" }, "Select shown ready"),
            h("button", { className: "secondary", type: "button", "data-select-group": "diarization-audio", "data-select-mode": "all" }, "Select shown"),
            h("button", { className: "ghost", type: "button", "data-select-group": "diarization-audio", "data-select-mode": "none" }, "Clear"),
            h("label", { className: "checkbox-row" }, h("input", { type: "checkbox", name: "diarization_include_completed" }), " Allow re-running already diarized files"),
            h(FolderFilterControl, { id: "diarization_audio_folder_filter", value: folderFilter, onChange: setFolderFilter, folders: folderChoices }),
            h(FileViewLimitControl, {
              id: "diarization_audio_row_limit",
              total: folderFilteredFiles.length,
              shown: renderedFiles.length,
              limit: rowLimit,
              onLimitChange: setRowLimit,
            })
          ),
          h(DataTable, { className: "compact-table diarization-audio-table dense-file-table file-view-table", headers: ["Select", "#", "Selected Model Status", "Audio File", "Folder", "Model Coverage", "Last Result"], rows: renderedFiles, emptyText: h("div", { className: "empty-state" }, h("strong", null, files.length ? "No audio files match this folder." : "No audio files to diarize yet."), files.length ? "Choose another folder or switch back to All folders." : "Upload audio in Media Library or convert a YouTube URL first; this tracker fills in once files land in audio_in/."), renderRow: (row, index) => h("tr", { key: row.name }, h("td", null, h("input", { type: "checkbox", name: "selected_audio", value: row.name, "data-check-group": "diarization-audio", "data-ready": row.selection_ready || "yes", "data-ready-by-model": row.selectionByModelJson || JSON.stringify(row.selectionByModel || {}) })), h("td", null, row.index || index + 1), h("td", null, h("span", { className: `queue-state ${row.state_class || "ready"}` }, row.queue_state || "ready"), h("p", { className: "row-note" }, row.targetModelLabel || row.targetBackendLabel || row.targetBackend || "selected model")), h("td", { title: row.name }, h("strong", { className: "file-name" }, row.fileName || row.name), row.name !== (row.fileName || row.name) ? h("p", { className: "row-note" }, row.name) : null), h("td", null, fileViewFolderLabel(row)), h("td", null, h(ModelCoverage, { statuses: row.modelStatuses || [] })), h("td", null, h("span", { className: "detail-text" }, row.detail || "No previous run."), row.lastRun ? h("p", { className: "row-note" }, row.lastRun) : null)) }),
          h("p", { className: "field-status" }, h("span", { "data-selection-count": "diarization-audio" }, "0"), " file(s) selected for the next diarization run. ", h(RuntimeEstimateBadge, { group: "diarization-audio", backend: (selectedModelKey || "nemo").split("/")[0] })),
          h("div", { className: "button-row" }, h("button", { type: "submit", name: "diarization_mode", value: "selected", "data-loading-message": "Submitting selected files to Slurm...", "data-selection-submit": "diarization-audio" }, "Run selected"), h("button", { className: "secondary", type: "submit", name: "diarization_mode", value: "all", "data-loading-message": "Submitting the full library to Slurm..." }, "Run all"))
        )
      ),
      h("article", { className: "panel" }, h("div", { className: "panel-head" }, h("div", null, h("h2", null, "Latest Diarization Run"), h("p", null, "Progress and queue status stay here. Output files are in Media Library."))), h(DiarizationRunPanel)),
      h("article", { className: "panel compact-panel" }, h("div", { className: "panel-head" }, h("div", null, h("h2", null, "Details"), h("p", null, "Saved defaults are available when needed. Diarized outputs now live in Media Library."))), h("div", { className: "button-row" }, h("a", { className: "tab-link", href: navItems.find((item) => item.path === "/uploads")?.href || "/uploads" }, "Open Media Library"), h("button", { className: "secondary", type: "button", "data-open-dialog": "diarization-model-dialog" }, "Open Model Defaults")), h(Dialog, { id: "diarization-model-dialog", title: "Saved Model Defaults", detail: "These defaults fill future runs." }, h(DiarizationModelSettings, { preferences })))
    );
  }

  function TrainingLabelDialog({ row, preferences, projectNames }) {
    const suffix = `${row.index}_${text(row.name).replace(/[^A-Za-z0-9_-]+/g, "_")}`;
    const backendId = `label_backend_${suffix}`;
    const projectId = `label_project_${suffix}`;
    const segmentsId = `label_segments_${suffix}`;
    const transcriptId = `label_transcript_${suffix}`;
    const questionsId = `label_questions_${suffix}`;
    const formKey = `training-label:${row.name}`;
    const [segmentRows, setSegmentRows] = React.useState(() => {
      const parsedRows = parseLabelSegmentRows(row.labelSegments || "");
      return labelEditorRowsWithDialogue(parsedRows.length ? parsedRows : [makeLabelEditorRow()], row.transcriptText || "");
    });
    const [transcriptText, setTranscriptText] = React.useState(row.transcriptText || "");
    // Default off unless the saved label explicitly kept dialogue.
    const [includeTranscript, setIncludeTranscript] = React.useState(() => {
      return row.includeTranscript === true;
    });
    const [issueQuestions, setIssueQuestions] = React.useState(row.issueQuestions || "");
    const [autoSaveStatus, setAutoSaveStatus] = React.useState({ state: "idle", at: "" });
    const targetOptions = React.useMemo(() => fineTunedTrainingTargetOptions(ctx.projects || []), []);
    const initialTarget = React.useMemo(() => {
      const savedTargets = row.targetProjects || [];
      return targetOptions.find((option) => savedTargets.includes(option.targetKey))?.optionKey || (targetOptions[0]?.optionKey || "__new__");
    }, [row.targetProjects, targetOptions]);
    const [targetSelection, setTargetSelection] = React.useState(initialTarget);
    const [newTargetBackend, setNewTargetBackend] = React.useState(() => {
      const normalized = text(row.backend).toLowerCase();
      if (normalized === "nemo" || normalized === "pyannote") return normalized;
      const preferred = text(preferences?.fine_tuning_backend || preferences?.default_backend).toLowerCase();
      return preferred === "nemo" ? "nemo" : "pyannote";
    });
    const [newTargetName, setNewTargetName] = React.useState(row.projectName || ctx.trainingLabels?.defaultProjectName || "uploaded-site-training");
    const [queueTraining, setQueueTraining] = React.useState(true);
    const [trainingName, setTrainingName] = React.useState("");
    const [audioStatus, setAudioStatus] = React.useState({ currentTime: 0, duration: 0, paused: true });
    const trainingNameId = `label_training_name_${suffix}`;
    const audioRef = React.useRef(null);
    const stopHandlerRef = React.useRef(null);
    const dragRowIdRef = React.useRef("");
    const formRef = React.useRef(null);
    const autoSaveAbortRef = React.useRef(null);
    const autoSaveTimerRef = React.useRef(null);
    const initialAutoSaveSkipRef = React.useRef(true);
    const firstDraftSaveRef = React.useRef(false);
    const draftDirtyRef = React.useRef(false);
    const autoSaveGenerationRef = React.useRef(0);
    const activeTimeFieldRef = React.useRef(null);
    const serializedSegments = serializeLabelSegmentRows(segmentRows);
    const rowDialogueText = serializeLabelDialogueRows(segmentRows);
    const submittedTranscriptText = includeTranscript ? rowDialogueText || transcriptText : "";
    const selectedTargetOption = targetOptions.find((option) => option.optionKey === targetSelection) || null;
    const targetBackend = selectedTargetOption?.backend || newTargetBackend;
    const targetProjectName = selectedTargetOption?.projectName || slugFineTuningTargetName(newTargetName || ctx.trainingLabels?.defaultProjectName || "uploaded-site-training");
    const targetKey = `${targetBackend}/${targetProjectName}`;
    const targetDisplayLabel = selectedTargetOption?.label || `${backendDisplayName(targetBackend)} / ${targetProjectName}`;
    const validSegmentCount = segmentRows.filter(labelEditorRowIsValid).length;
    const incompleteSegmentCount = segmentRows.filter((segment) => labelEditorRowHasAnyValue(segment) && !labelEditorRowIsValid(segment)).length;
    const completeDisabled = validSegmentCount === 0 || incompleteSegmentCount > 0;

    React.useEffect(() => {
      return () => {
        if (audioRef.current && stopHandlerRef.current) {
          audioRef.current.removeEventListener("timeupdate", stopHandlerRef.current);
        }
        flushAutoSaveDraft();
        if (autoSaveAbortRef.current) {
          autoSaveAbortRef.current.abort();
          autoSaveAbortRef.current = null;
        }
      };
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    React.useEffect(() => {
      const audio = audioRef.current;
      if (!audio) return undefined;
      const syncAudioStatus = () => {
        setAudioStatus({
          currentTime: Number.isFinite(audio.currentTime) ? audio.currentTime : 0,
          duration: Number.isFinite(audio.duration) ? audio.duration : 0,
          paused: audio.paused || audio.ended,
        });
      };
      syncAudioStatus();
      audio.addEventListener("loadedmetadata", syncAudioStatus);
      audio.addEventListener("timeupdate", syncAudioStatus);
      audio.addEventListener("seeked", syncAudioStatus);
      audio.addEventListener("play", syncAudioStatus);
      audio.addEventListener("pause", syncAudioStatus);
      audio.addEventListener("ended", syncAudioStatus);
      return () => {
        audio.removeEventListener("loadedmetadata", syncAudioStatus);
        audio.removeEventListener("timeupdate", syncAudioStatus);
        audio.removeEventListener("seeked", syncAudioStatus);
        audio.removeEventListener("play", syncAudioStatus);
        audio.removeEventListener("pause", syncAudioStatus);
        audio.removeEventListener("ended", syncAudioStatus);
      };
    }, [row.audioHref]);

    React.useEffect(() => {
      const handlePageHide = () => flushAutoSaveDraft();
      const handleVisibilityChange = () => {
        if (document.visibilityState === "hidden") flushAutoSaveDraft();
      };
      window.addEventListener("pagehide", handlePageHide);
      document.addEventListener("visibilitychange", handleVisibilityChange);
      return () => {
        window.removeEventListener("pagehide", handlePageHide);
        document.removeEventListener("visibilitychange", handleVisibilityChange);
      };
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    // Auto-save every manual-label change as a draft. The first edit saves
    // immediately so the label leaves "not started" even if the user navigates
    // away right after editing; later edits are debounced.
    React.useEffect(() => {
      if (initialAutoSaveSkipRef.current) {
        initialAutoSaveSkipRef.current = false;
        return undefined;
      }
      draftDirtyRef.current = true;
      autoSaveGenerationRef.current += 1;
      const saveGeneration = autoSaveGenerationRef.current;
      if (!firstDraftSaveRef.current) {
        firstDraftSaveRef.current = true;
        runAutoSaveDraft(saveGeneration);
        return undefined;
      }
      if (autoSaveTimerRef.current) window.clearTimeout(autoSaveTimerRef.current);
      autoSaveTimerRef.current = window.setTimeout(() => {
        autoSaveTimerRef.current = null;
        runAutoSaveDraft(saveGeneration);
      }, 1500);
      return () => {
        if (autoSaveTimerRef.current) {
          window.clearTimeout(autoSaveTimerRef.current);
          autoSaveTimerRef.current = null;
        }
      };
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [serializedSegments, rowDialogueText, transcriptText, issueQuestions, includeTranscript]);

    function slugFineTuningTargetName(value) {
      let cleaned = text(value).trim().toLowerCase().replace(/[^a-z0-9]+/g, "-");
      while (cleaned.includes("--")) cleaned = cleaned.replace(/--/g, "-");
      return cleaned.replace(/^-+|-+$/g, "") || "uploaded-site-training";
    }

    function fineTunedTrainingTargetOptions(projects) {
      const options = [];
      (projects || []).forEach((project) => {
        const backend = text(project.backend).toLowerCase();
        if (backend !== "nemo" && backend !== "pyannote") return;
        const projectName = slugFineTuningTargetName(project.slug || project.displayName);
        const targetKey = `${backend}/${projectName}`;
        const projectLabel = project.displayName || project.slug || projectName;
        const backendLabel = backendDisplayName(backend);
        const runs = Array.isArray(project.recentRuns) ? project.recentRuns : [];
        if (runs.length) {
          runs.forEach((run, index) => {
            const runLabel = run.displayName || run.versionName || run.runName || `trained version ${index + 1}`;
            options.push({
              optionKey: `${targetKey}::${run.versionName || run.runName || index}`,
              targetKey,
              backend,
              projectName,
              label: `${backendLabel} / ${runLabel}`,
              detail: projectLabel !== projectName ? projectLabel : "",
            });
          });
          return;
        }
        if (Number(project.sampleCount || 0) > 0 || project.prepared) {
          options.push({
            optionKey: `${targetKey}::project`,
            targetKey,
            backend,
            projectName,
            label: `${backendLabel} / ${projectLabel}`,
            detail: project.prepared ? "prepared project" : "label project",
          });
        }
      });
      return options.sort((left, right) => left.label.localeCompare(right.label));
    }

    function autoSaveFormData() {
      if (!formRef.current) return null;
      const data = new FormData(formRef.current);
      data.set("label_action", "draft");
      data.set("label_auto_save", "1");
      return data;
    }

    function runAutoSaveDraft(saveGeneration = autoSaveGenerationRef.current) {
      if (!routes.saveTrainingLabel || !formRef.current) return;
      if (autoSaveAbortRef.current) {
        autoSaveAbortRef.current.abort();
      }
      const controller = new AbortController();
      autoSaveAbortRef.current = controller;
      const data = autoSaveFormData();
      if (!data) return;
      setAutoSaveStatus({ state: "saving", at: "" });
      fetch(routes.saveTrainingLabel, {
        method: "POST",
        body: data,
        credentials: "same-origin",
        signal: controller.signal,
        redirect: "manual",
      })
        .then((response) => {
          if (response.type === "opaqueredirect" || (response.status >= 200 && response.status < 400) || response.status === 0) {
            autoSaveAbortRef.current = null;
            if (saveGeneration === autoSaveGenerationRef.current) {
              draftDirtyRef.current = false;
            }
            const stamp = new Date().toLocaleTimeString();
            setAutoSaveStatus({ state: "saved", at: stamp });
          } else {
            setAutoSaveStatus({ state: "error", at: "" });
          }
        })
        .catch((error) => {
          if (error?.name === "AbortError") return;
          autoSaveAbortRef.current = null;
          setAutoSaveStatus({ state: "error", at: "" });
        });
    }

    function flushAutoSaveDraft() {
      if (!routes.saveTrainingLabel || !formRef.current || !draftDirtyRef.current) return;
      if (autoSaveTimerRef.current) {
        window.clearTimeout(autoSaveTimerRef.current);
        autoSaveTimerRef.current = null;
      }
      if (autoSaveAbortRef.current) {
        autoSaveAbortRef.current.abort();
        autoSaveAbortRef.current = null;
      }
      const data = autoSaveFormData();
      if (!data) return;
      let delivered = false;
      if (navigator.sendBeacon) {
        try {
          delivered = navigator.sendBeacon(routes.saveTrainingLabel, data);
        } catch (_error) {
          delivered = false;
        }
      }
      if (!delivered) {
        try {
          fetch(routes.saveTrainingLabel, {
            method: "POST",
            body: data,
            credentials: "same-origin",
            keepalive: true,
            redirect: "manual",
          });
        } catch (_error) {}
      }
    }

    function clearSegmentStopHandler() {
      const audio = audioRef.current;
      if (audio && stopHandlerRef.current) {
        audio.removeEventListener("timeupdate", stopHandlerRef.current);
      }
      stopHandlerRef.current = null;
    }

    function syncAudioStatusNow() {
      const audio = audioRef.current;
      if (!audio) return;
      setAudioStatus({
        currentTime: Number.isFinite(audio.currentTime) ? audio.currentTime : 0,
        duration: Number.isFinite(audio.duration) ? audio.duration : 0,
        paused: audio.paused || audio.ended,
      });
    }

    function toggleAudioPlayback() {
      const audio = audioRef.current;
      if (!audio) return;
      clearSegmentStopHandler();
      if (audio.paused || audio.ended) {
        const playPromise = audio.play();
        if (playPromise && typeof playPromise.catch === "function") {
          playPromise.catch(() => {});
        }
      } else {
        audio.pause();
      }
      syncAudioStatusNow();
    }

    function rewindAudio(seconds = 2) {
      const audio = audioRef.current;
      if (!audio || !Number.isFinite(audio.currentTime)) return;
      try {
        audio.currentTime = Math.max(0, audio.currentTime - seconds);
      } catch (_error) {
        return;
      }
      syncAudioStatusNow();
    }

    function updateSegmentRow(rowId, updates) {
      setSegmentRows((currentRows) => currentRows.map((segment) => (segment.id === rowId ? { ...segment, ...updates } : segment)));
    }

    function addSegmentRow() {
      setSegmentRows((currentRows) => [...currentRows, makeLabelEditorRow()]);
    }

    function deleteSegmentRow(rowId) {
      setSegmentRows((currentRows) => {
        const nextRows = currentRows.filter((segment) => segment.id !== rowId);
        return nextRows.length ? nextRows : [makeLabelEditorRow()];
      });
    }

    function moveSegmentRow(rowId, direction) {
      setSegmentRows((currentRows) => {
        const index = currentRows.findIndex((segment) => segment.id === rowId);
        const nextIndex = index + direction;
        if (index < 0 || nextIndex < 0 || nextIndex >= currentRows.length) {
          return currentRows;
        }
        const nextRows = currentRows.slice();
        const [moving] = nextRows.splice(index, 1);
        nextRows.splice(nextIndex, 0, moving);
        return nextRows;
      });
    }

    function dropSegmentRow(targetRowId) {
      const draggedRowId = dragRowIdRef.current;
      if (!draggedRowId || draggedRowId === targetRowId) {
        return;
      }
      setSegmentRows((currentRows) => {
        const fromIndex = currentRows.findIndex((segment) => segment.id === draggedRowId);
        const toIndex = currentRows.findIndex((segment) => segment.id === targetRowId);
        if (fromIndex < 0 || toIndex < 0 || fromIndex === toIndex) {
          return currentRows;
        }
        const nextRows = currentRows.slice();
        const [moving] = nextRows.splice(fromIndex, 1);
        nextRows.splice(toIndex, 0, moving);
        return nextRows;
      });
      dragRowIdRef.current = "";
    }

    function playSegmentRow(segment) {
      const audio = audioRef.current;
      if (!audio || !labelEditorRowIsValid(segment)) {
        return;
      }
      clearSegmentStopHandler();
      const start = parseLabelTimestampInput(segment.start);
      const end = parseLabelTimestampInput(segment.end);
      try {
        audio.pause();
        audio.currentTime = Math.max(start, 0);
      } catch (_error) {
        return;
      }
      const stopAtEnd = () => {
        if (audio.currentTime >= end - 0.025) {
          audio.pause();
          audio.removeEventListener("timeupdate", stopAtEnd);
          stopHandlerRef.current = null;
        }
      };
      stopHandlerRef.current = stopAtEnd;
      audio.addEventListener("timeupdate", stopAtEnd);
      const playPromise = audio.play();
      if (playPromise && typeof playPromise.catch === "function") {
        playPromise.catch(() => {});
      }
    }

    function rememberActiveTimeField(rowId, field) {
      activeTimeFieldRef.current = { rowId, field };
    }

    function fillActiveTimeField() {
      const audio = audioRef.current;
      const target = activeTimeFieldRef.current;
      if (!audio || !Number.isFinite(audio.currentTime)) {
        window.alert("The audio player does not have a current time yet.");
        return;
      }
      if (!target?.rowId || !target?.field) {
        window.alert("Click a Start or End time box first.");
        return;
      }
      const timestamp = secondsForLabelInput(audio.currentTime);
      setSegmentRows((currentRows) =>
        currentRows.map((segment) => (segment.id === target.rowId ? { ...segment, [target.field]: timestamp } : segment))
      );
      window.setTimeout(() => {
        const input = formRef.current?.querySelector(`[data-label-row-id="${target.rowId}"] [data-label-field="${target.field}"]`);
        if (input) {
          input.focus();
          input.select();
        }
      }, 0);
    }

    function handleSubmit(event) {
      const submitter = event.nativeEvent?.submitter;
      const stats = syncTrainingLabelEditorForm(event.currentTarget);
      if (submitter?.value === "complete" && (stats.validCount === 0 || stats.incompleteCount > 0)) {
        event.preventDefault();
        window.alert("Complete labels need at least one valid row, and every filled row needs start, end, and speaker.");
      }
    }

    return h(
      "form",
      {
        method: "post",
        action: routes.saveTrainingLabel,
        onSubmit: handleSubmit,
        ref: formRef,
        "data-form-key": formKey,
        "data-pause-refresh": "true",
        "data-replace-submit": "true",
        "data-training-label-editor": "true",
        "data-loading-message": "Saving training label...",
      },
      h("input", { type: "hidden", name: "audio_file", value: row.name }),
      h("input", { type: "hidden", name: "label_backend", value: targetBackend, readOnly: true }),
      h("input", { type: "hidden", name: "label_project_name", value: targetProjectName, readOnly: true }),
      h("input", { type: "hidden", name: "label_training_targets", value: targetKey, readOnly: true }),
      h("input", { type: "hidden", name: "label_training_target_label", value: targetDisplayLabel, readOnly: true }),
      queueTraining ? h("input", { type: "hidden", name: "label_auto_train_targets", value: targetKey, readOnly: true }) : h("input", { type: "hidden", name: "label_auto_train_skip", value: "1", readOnly: true }),
      h("input", { type: "hidden", id: segmentsId, name: "label_segments", value: serializedSegments, readOnly: true, "data-label-segments-input": "true" }),
      h("input", { type: "hidden", name: "label_transcript_text", value: submittedTranscriptText, readOnly: true, "data-label-transcript-input": "true" }),
      row.systemQuestions?.length
        ? h("div", { className: "callout warm compact-callout" }, h("h2", null, "Questions To Resolve"), h("ul", null, row.systemQuestions.map((question, index) => h("li", { key: index }, question))))
        : null,
      // Audio player lives at the dialog level so it shows for every uploaded
      // file, even one that has not been diarized yet. The bug it replaces:
      // LabelSourcePicker used to host the player and returned null entirely
      // when there were no diarization comparison rows, leaving the user no
      // way to listen to the WAV they were trying to label.
      row.audioHref
        ? h(
            "div",
            { className: "label-audio" },
            h("p", { className: "label-audio-caption" }, "Audio sample"),
            h("h3", { className: "label-audio-title" }, row.fileName || row.name),
            h("audio", { ref: audioRef, controls: true, preload: "metadata", src: row.audioHref, "aria-label": `Audio preview for ${row.fileName || row.name}` }),
            h(
              "div",
              { className: "label-audio-toolbar", role: "toolbar", "aria-label": "Manual label playback controls" },
              h("button", { className: "secondary", type: "button", onClick: toggleAudioPlayback }, audioStatus.paused ? "Play" : "Pause"),
              h("button", { className: "secondary", type: "button", onClick: () => rewindAudio(2) }, "Back 2 sec"),
              h("button", { className: "secondary", type: "button", onClick: fillActiveTimeField }, "Use Current Seconds"),
              h("span", { className: "label-current-time", role: "status" }, `Current: ${secondsForLabelInput(audioStatus.currentTime) || "0.000"}s`)
            )
          )
        : h(
            "div",
            { className: "label-audio empty-state" },
            h("p", { className: "label-audio-caption" }, "Audio sample"),
            h("h3", { className: "label-audio-title" }, "Audio file is not on disk."),
            h("p", { className: "row-note" }, "The WAV may have been moved or deleted. Re-upload it from the Media Library to label.")
          ),
      // Diarization-model presets used to live here (LabelSourcePicker). Pulled
      // out so every new file starts with empty Start/End fields and the user
      // commits to typing each timestamp, instead of accepting the model's word.
      h(
        "section",
        { className: "label-training-target" },
        h(
          "div",
          { className: "label-training-target-head" },
          h("div", null, h("h3", null, "Fine-tune target"), h("p", null, targetDisplayLabel)),
          h("span", { className: "target-backend-pill" }, backendDisplayName(targetBackend))
        ),
        h(
          "div",
          { className: "inline" },
          h(
            Field,
            { id: `label_target_${suffix}`, label: "Fine-tuned target" },
            h(
              "select",
              {
                id: `label_target_${suffix}`,
                value: targetSelection,
                onChange: (event) => setTargetSelection(event.target.value),
              },
              targetOptions.map((option) =>
                h("option", { key: option.optionKey, value: option.optionKey }, option.detail ? `${option.label} (${option.detail})` : option.label)
              ),
              h("option", { value: "__new__" }, "Create new fine-tuned model")
            )
          ),
          h(
            Field,
            { id: trainingNameId, label: "New trained version name" },
            h("input", {
              id: trainingNameId,
              name: "label_new_training_name",
              type: "text",
              value: trainingName,
              onChange: (event) => setTrainingName(event.target.value),
              placeholder: "cleaned-stage-2",
              maxLength: 80,
            })
          )
        ),
        targetSelection === "__new__"
          ? h(
              "div",
              { className: "inline" },
              h(
                Field,
                { id: backendId, label: "Backend" },
                h("select", { id: backendId, value: newTargetBackend, onChange: (event) => setNewTargetBackend(event.target.value) }, h("option", { value: "pyannote" }, "pyannote"), h("option", { value: "nemo" }, "NeMo"))
              ),
              h(
                Field,
                { id: projectId, label: "Fine-tuned model name" },
                h("input", {
                  id: projectId,
                  list: "training_label_project_names",
                  value: newTargetName,
                  onChange: (event) => setNewTargetName(event.target.value),
                  placeholder: "uploaded-site-training",
                  required: true,
                })
              )
            )
          : null,
        h(
          "label",
          { className: "checkbox-row label-queue-toggle" },
          h("input", { type: "checkbox", checked: queueTraining, onChange: (event) => setQueueTraining(event.target.checked) }),
          " Queue sbatch training when complete"
        )
      ),
      h(
        "section",
        { className: "label-editor", "aria-labelledby": `${segmentsId}_heading` },
        h(
          "div",
          { className: "label-editor-head" },
          h("div", null, h("h3", { id: `${segmentsId}_heading` }, "Speaker-Time Labels"), h("p", { className: "row-note" }, `${validSegmentCount} valid row(s), ${incompleteSegmentCount} row(s) need fixes.`)),
          h(
            "div",
            { className: "label-editor-actions" },
            h("button", { className: "secondary", type: "button", onClick: addSegmentRow }, "Add Label"),
            h(
              "label",
              { className: "checkbox-row label-dialogue-toggle" },
              h("input", { type: "checkbox", checked: includeTranscript, onChange: (event) => setIncludeTranscript(event.target.checked) }),
              " Show Dialogue"
            )
          )
        ),
        h(
          "div",
          { className: "table-scroll label-editor-table-wrap" },
          h(
            "table",
            { className: classNames("label-editor-table", !includeTranscript && "hide-dialogue") },
            h("thead", null, h("tr", null, h("th", null, "#"), h("th", null, "Start"), h("th", null, "End"), h("th", null, "Speaker"), h("th", { className: "label-dialogue-col" }, "Dialogue"), h("th", null, "Actions"))),
            h(
              "tbody",
              null,
              segmentRows.map((segment, index) => {
                return h(
                  "tr",
                  {
                    key: segment.id,
                    "data-label-editor-row": "true",
                    "data-label-row-id": segment.id,
                    className: classNames(labelEditorRowHasAnyValue(segment) && !labelEditorRowIsValid(segment) ? "needs-work" : ""),
                    onDragOver: (event) => event.preventDefault(),
                    onDrop: (event) => {
                      event.preventDefault();
                      dropSegmentRow(segment.id);
                    },
                  },
                  h(
                    "td",
                    { className: "label-editor-index" },
                    h(
                      "button",
                      {
                        className: "ghost label-row-handle",
                        type: "button",
                        draggable: true,
                        title: "Drag to reorder",
                        "aria-label": `Move label row ${index + 1}`,
                        onDragStart: () => {
                          dragRowIdRef.current = segment.id;
                        },
                        onDragEnd: () => {
                          dragRowIdRef.current = "";
                        },
                      },
                      String(index + 1)
                    )
                  ),
                  h(
                    "td",
                    null,
                    h("input", { "aria-label": `Start time for label ${index + 1}`, type: "text", inputMode: "decimal", value: segment.start, "data-label-field": "start", onFocus: () => rememberActiveTimeField(segment.id, "start"), onChange: (event) => updateSegmentRow(segment.id, { start: event.target.value }) })
                  ),
                  h(
                    "td",
                    null,
                    h("input", { "aria-label": `End time for label ${index + 1}`, type: "text", inputMode: "decimal", value: segment.end, "data-label-field": "end", onFocus: () => rememberActiveTimeField(segment.id, "end"), onChange: (event) => updateSegmentRow(segment.id, { end: event.target.value }) })
                  ),
                  h("td", null, h("input", { "aria-label": `Speaker for label ${index + 1}`, type: "text", list: "training_label_speaker_names", value: segment.speaker, "data-label-field": "speaker", onChange: (event) => updateSegmentRow(segment.id, { speaker: event.target.value }), placeholder: `SPEAKER_${String(index).padStart(2, "0")}` })),
                  h("td", { className: "label-dialogue-cell" }, h("input", { "aria-label": `Dialogue for label ${index + 1}`, type: "text", value: segment.dialogue || "", "data-label-field": "dialogue", onChange: (event) => updateSegmentRow(segment.id, { dialogue: event.target.value }), placeholder: "Optional dialogue" })),
                  h(
                    "td",
                    null,
                    h(
                      "div",
                      { className: "label-editor-actions" },
                      h("button", { className: "secondary", type: "button", onClick: () => playSegmentRow(segment), disabled: !row.audioHref || !labelEditorRowIsValid(segment) }, "Listen"),
                      h("button", { className: "ghost", type: "button", onClick: () => moveSegmentRow(segment.id, -1), disabled: index === 0 }, "Up"),
                      h("button", { className: "ghost", type: "button", onClick: () => moveSegmentRow(segment.id, 1), disabled: index === segmentRows.length - 1 }, "Down"),
                      h("button", { className: "ghost danger", type: "button", onClick: () => deleteSegmentRow(segment.id) }, "Delete")
                    )
                  )
                );
              })
            )
          )
        )
      ),
      h(
        "details",
        { className: "details-box label-optional-details", open: Boolean(transcriptText || issueQuestions) },
        h("summary", null, "Optional transcript and questions"),
        // Hidden input mirrors the toggle so the form ALWAYS submits a "1" or
        // "0", letting the server distinguish "explicitly off" from "legacy
        // form that doesn't have this toggle at all" (treated as on).
        h("input", {
          type: "hidden",
          name: "label_include_transcript",
          value: includeTranscript ? "1" : "0",
        }),
        h(
          "label",
          { className: "checkbox-row", title: "Diarization fine-tuning ignores transcripts. The toggle just controls whether the dialogue is saved alongside the sample for your records." },
          h("input", {
            type: "checkbox",
            checked: includeTranscript,
            onChange: (event) => setIncludeTranscript(event.target.checked),
          }),
          " Include dialogue with this sample (does not affect diarization training)"
        ),
        includeTranscript
          ? h(Field, { id: transcriptId, label: "Transcript export", note: "Saved in the project's text/ folder. Marked as not aligned with speech-time labels." }, h("textarea", { id: transcriptId, value: transcriptText, onChange: (event) => setTranscriptText(event.target.value), placeholder: "Transcript text or annotation notes" }))
          : h("p", { className: "row-note" }, "Dialogue is excluded from this sample. Toggle on to save it for your records (training is unaffected either way)."),
        h(Field, { id: questionsId, label: "Questions or issues", note: "Anything here keeps the item out of completed training until it is answered." }, h("textarea", { id: questionsId, name: "label_issue_questions", value: issueQuestions, onChange: (event) => setIssueQuestions(event.target.value), placeholder: "What needs to be clarified before this can be used for training?" }))
      ),
      h(
        "div",
        { className: "button-row" },
        h("button", { className: "secondary", type: "submit", name: "label_action", value: "draft" }, "Save For Later"),
        h("button", { type: "submit", name: "label_action", value: "complete", disabled: completeDisabled }, "Complete For Training"),
        row.reviewHref ? h("a", { className: "tab-link", href: row.reviewHref }, "Open Review Page") : null,
        h(
          "span",
          { className: `auto-save-status ${autoSaveStatus.state}`, role: "status" },
          autoSaveStatus.state === "saving"
            ? "Auto-saving draft…"
            : autoSaveStatus.state === "saved"
              ? `Auto-saved at ${autoSaveStatus.at}`
              : autoSaveStatus.state === "error"
                ? "Auto-save failed (changes still in form)"
                : "Auto-save ready"
        )
      )
    );
  }

  function TrainingAudioCacheAll({ rows }) {
    const cacheItems = React.useMemo(() => {
      const seen = new Set();
      const items = [];
      (rows || []).forEach((row) => {
        const url = normalizedAudioCacheUrl(row.audioHref);
        if (!url || seen.has(url)) return;
        seen.add(url);
        items.push({
          url,
          name: row.fileName || row.name || url,
        });
      });
      return items;
    }, [rows]);
    const [cacheState, setCacheState] = React.useState({
      status: "idle",
      done: 0,
      total: 0,
      message: "",
    });
    const cacheAbortRef = React.useRef(null);
    React.useEffect(() => {
      return () => {
        if (cacheAbortRef.current) {
          cacheAbortRef.current.abort();
          cacheAbortRef.current = null;
        }
      };
    }, []);
    const isRunning = cacheState.status === "running";
    const browserCacheAvailable = Boolean(window.indexedDB && window.fetch);
    const canCache = cacheItems.length > 0 && browserCacheAvailable;

    async function startCachingAll() {
      if (isRunning || !canCache) return;
      const db = await openAudioCacheDb();
      if (!db) {
        setCacheState({
          status: "error",
          done: 0,
          total: cacheItems.length,
          message: "Audio cache is not available in this browser.",
        });
        return;
      }
      const controller = new AbortController();
      cacheAbortRef.current = controller;
      let nextIndex = 0;
      let done = 0;
      let stored = 0;
      let alreadyCached = 0;
      let failed = 0;
      let cachedBytes = 0;
      const total = cacheItems.length;
      setCacheState({ status: "running", done: 0, total, message: `Caching 0 of ${total} audio file(s)...` });

      async function worker() {
        while (!controller.signal.aborted) {
          const item = cacheItems[nextIndex];
          nextIndex += 1;
          if (!item) return;
          setCacheState((previous) => ({
            ...previous,
            message: `Caching ${done + 1} of ${total}: ${item.name}`,
          }));
          try {
            const result = await cacheAudioUrl(db, item, controller.signal, (received, expected) => {
              setCacheState((previous) => ({
                ...previous,
                message: expected
                  ? `Caching ${item.name}: ${formatBytes(received)} of ${formatBytes(expected)}`
                  : `Caching ${item.name}: ${formatBytes(received)}`,
              }));
            });
            if (result.status === "cached") {
              alreadyCached += 1;
            } else {
              stored += 1;
            }
            cachedBytes += Number(result.bytes || 0);
          } catch (error) {
            if (error && error.name === "AbortError") return;
            failed += 1;
          } finally {
            done += 1;
            setCacheState({
              status: "running",
              done,
              total,
              message: `Cached ${done} of ${total}.`,
            });
          }
        }
      }

      try {
        const workerCount = Math.min(AUDIO_CACHE_ALL_CONCURRENCY, total);
        await Promise.all(Array.from({ length: workerCount }, () => worker()));
        const wasCancelled = controller.signal.aborted;
        setCacheState({
          status: wasCancelled ? "idle" : failed ? "error" : "done",
          done,
          total,
          message: wasCancelled
            ? `Stopped after ${done} of ${total} audio file(s).`
            : `Cache all finished: ${stored} added, ${alreadyCached} already cached${failed ? `, ${failed} failed` : ""}. ${formatBytes(cachedBytes)} available locally.`,
        });
      } finally {
        try {
          db.close();
        } catch (_error) {}
        if (cacheAbortRef.current === controller) {
          cacheAbortRef.current = null;
        }
      }
    }

    function stopCachingAll() {
      if (cacheAbortRef.current) {
        cacheAbortRef.current.abort();
      }
    }

    return h(
      "div",
      { className: "training-cache-controls" },
      h(
        "button",
        {
          className: "secondary",
          type: "button",
          onClick: startCachingAll,
          disabled: isRunning || !canCache,
        },
        isRunning ? "Caching Audio..." : "Cache All Audio"
      ),
      isRunning ? h("button", { className: "ghost", type: "button", onClick: stopCachingAll }, "Stop") : null,
      h(
        "p",
        { className: `field-status cache-status-${cacheState.status}` },
        cacheState.message ||
          (browserCacheAvailable
            ? cacheItems.length
              ? `${cacheItems.length} audio file(s) available to cache.`
              : "No audio files to cache."
            : "Audio cache is not available in this browser.")
      )
    );
  }

  function uncompleteTrainingLabel(row) {
    if (!row || !row.name || !routes.uncompleteTrainingLabel) return;
    // Confirm before deleting sample files. The audio + RTTM in the project's
    // training set get removed, so we don't want this triggered by a
    // mis-click. Status rolls back to draft; the segments stay intact for
    // editing.
    const ok = window.confirm(
      `Remove "${row.fileName || row.name}" from completed training? ` +
      `This deletes the sample files from ${row.trainingProjects?.length ? row.trainingProjects.join(", ") : "the fine-tuning project"} and rolls the label back to a draft.`
    );
    if (!ok) return;
    submitHiddenForm(routes.uncompleteTrainingLabel, { audio_file: row.name });
  }

  function TrainingLabelsTable({ rows }) {
    return h(DataTable, {
      className: "training-label-table dense-file-table file-view-table",
      headers: ["#", "Audio", "Status", "Training Target", "Notes", "Label"],
      rows,
      emptyText: "No uploaded audio matches this view.",
      renderRow: (row) => {
        const trainingProjects = row.status === "completed" && row.trainingProjects?.length ? row.trainingProjects : row.targetProjects || [];
        const projectName = row.projectName || ctx.trainingLabels?.defaultProjectName || "uploaded-site-training";
        const backendLabel = row.backendLabel || backendDisplayName(row.backend || "both");
        return h(
          "tr",
          { key: row.name },
          h("td", null, row.index),
	          h(
	            "td",
	            { title: row.name || "" },
	            h("strong", { className: "file-name" }, row.fileName || row.name),
	            h("p", { className: "row-note" }, fileViewFolderLabel(row))
	          ),
          h("td", null, h(StatusPill, { status: row.status || "not_started" }), row.updatedAt ? h("p", { className: "row-note" }, `Updated: ${row.updatedAt}`) : null),
          h(
            "td",
            null,
            h("strong", null, backendLabel),
            h("p", { className: "row-note" }, trainingProjects.length ? trainingProjects.join(", ") : projectName)
          ),
          h(
            "td",
            null,
            h("span", { className: "detail-text" }, row.detail || ""),
            row.source ? h("p", { className: "row-note" }, `Source: ${text(row.source).replace(/_/g, " ")}`) : null,
            row.completedAt ? h("p", { className: "row-note" }, `Completed: ${row.completedAt}`) : null,
            row.segmentCount ? h("p", { className: "row-note" }, `${row.segmentCount} segment(s), ${row.speakerCount || 0} speaker(s)`) : null,
            row.trainingRttmPath ? h("p", { className: "row-note" }, row.trainingRttmPath) : null
          ),
          h(
            "td",
            null,
            row.reviewHref
              ? h("a", { className: "tab-link", href: row.reviewHref }, row.status === "completed" ? "Review Label" : "Inspect")
              : h("span", { className: "row-note" }, "Run diarization to create a review page"),
            row.status === "completed"
              ? h(
                  "button",
                  {
                    type: "button",
                    className: "ghost danger training-label-uncomplete",
                    onClick: () => uncompleteTrainingLabel(row),
                    title: "Remove this sample from the fine-tuning project's training set and roll the label back to a draft.",
                  },
                  "Remove From Training"
                )
              : null
          )
        );
      },
    });
  }

  function TrainingLabelsPage() {
    const labels = ctx.trainingLabels || {};
    const rows = labels.rows || [];
    const summary = labels.summary || {};
    const [showCompleted, setShowCompleted] = React.useState(() => Boolean(uiState.showCompletedTrainingLabels));
    const [folderFilter, setFolderFilter] = React.useState("all");
    const [rowLimit, setRowLimit] = React.useState(DEFAULT_FILE_VIEW_LIMIT);
    React.useEffect(() => {
      uiState.showCompletedTrainingLabels = showCompleted;
    }, [showCompleted]);
    const statusFilteredRows = React.useMemo(
      () => showCompleted ? rows : rows.filter((row) => row.status !== "completed"),
      [rows, showCompleted]
    );
    const sortedRows = React.useMemo(
      () => sortFileRowsByFolder(statusFilteredRows, fileViewFolderLabel, (row) => row.fileName || row.name),
      [statusFilteredRows]
    );
    const folderChoices = React.useMemo(() => fileViewFolderChoices(sortedRows, fileViewFolderLabel), [sortedRows]);
    React.useEffect(() => {
      if (folderFilter !== "all" && !folderChoices.some((folder) => folder.key === folderFilter)) {
        setFolderFilter("all");
      }
    }, [folderFilter, folderChoices]);
    const folderFilteredRows = folderFilter === "all"
      ? sortedRows
      : sortedRows.filter((row) => fileViewFolderLabel(row) === folderFilter);
    const visibleRows = folderFilteredRows.slice(0, Math.min(rowLimit, folderFilteredRows.length));
    const projectNames = Array.from(new Set((ctx.projects || []).map((project) => project.slug).filter(Boolean))).sort();
    const speakerNames = Array.from(
      new Set(
        rows
          .flatMap((row) => parseLabelSegmentRows(row.labelSegments || "").map((segment) => text(segment.speaker).trim()))
          .filter(Boolean)
      )
    ).sort();
    const defaultProjectName = labels.defaultProjectName || "uploaded-site-training";
    const completedCount = Number(summary.completed || 0);
    return h(
      React.Fragment,
      null,
      h(
        "article",
        { className: "panel" },
        projectNames.length ? h("datalist", { id: "training_label_project_names" }, projectNames.map((name) => h("option", { key: name, value: name }))) : null,
        speakerNames.length ? h("datalist", { id: "training_label_speaker_names" }, speakerNames.map((name) => h("option", { key: name, value: name }))) : null,
        h(
          "div",
          { className: "panel-head" },
          h(
            "div",
            null,
            h("h2", null, "Training Labels"),
            h("p", null, "Review each uploaded recording, refine the speaker timings, and save the finished label as a fine-tuning sample.")
          )
        ),
        h(ProcessSteps, { tone: "warm", steps: ["Review diarization output", "Save draft or complete labels", "Use completed samples for training"] }),
        h(
          "div",
          { className: "summary-grid" },
          h(SummaryCard, { title: "Uploaded Items" }, h("p", null, h("strong", null, summary.total || 0), " file(s)")),
          h(SummaryCard, { title: "Saved For Later" }, h("p", null, h("strong", null, summary.draft || 0), " draft item(s)")),
          h(SummaryCard, { title: "Questions" }, h("p", null, h("strong", null, summary.needs_review || 0), " item(s) need answers")),
          h(SummaryCard, { title: "Completed" }, h("p", null, h("strong", null, summary.completed || 0), " training sample(s)"), h("p", { className: "row-note" }, `Default project: ${defaultProjectName}`))
        )
      ),
      h(
        "article",
        { className: "panel" },
        h(
          "div",
          { className: "panel-head training-labels-head" },
          h(
            "div",
            null,
            h("h2", null, "Uploaded Material To Label"),
            h("p", null, "Open Inspect to label from the review page with audio playback, timing rows, and fine-tuning save actions.")
          ),
	          h(
		            "div",
		            { className: "training-labels-controls" },
		            h(TrainingAudioCacheAll, { rows }),
		            h(FolderFilterControl, { id: "training_labels_folder_filter", value: folderFilter, onChange: setFolderFilter, folders: folderChoices }),
		            h(FileViewLimitControl, {
		              id: "training_labels_row_limit",
		              total: folderFilteredRows.length,
		              shown: visibleRows.length,
		              limit: rowLimit,
		              onLimitChange: setRowLimit,
		            }),
		            h(
		              "label",
	              { className: "checkbox-row training-labels-toggle" },
              h("input", { type: "checkbox", checked: showCompleted, onChange: (event) => setShowCompleted(event.target.checked) }),
              ` Show completed labels${completedCount ? ` (${completedCount})` : ""}`
            )
          )
        ),
        h(TrainingLabelsTable, { rows: visibleRows })
      )
    );
  }

  function projectChoices({ preparedOnly = false } = {}) {
    return (ctx.projects || []).filter((project) => Number(project.sampleCount || 0) > 0 && (!preparedOnly || project.prepared));
  }

  function FineTuneProjectOptions({ projects }) {
    if (!projects.length) {
      return h("option", { value: "" }, "No matching projects yet");
    }
    return projects.map((project) =>
      h(
        "option",
        {
          key: `${project.backend}/${project.slug}`,
          value: project.slug,
          "data-project-backend": project.backend,
        },
        `${backendDisplayName(project.backend)} / ${project.slug} (${project.sampleCount} sample(s))`
      )
    );
  }

  function WorkspaceFileChecklist({ group, name, files, emptyText }) {
    const rows = files || [];
    const [folderFilter, setFolderFilter] = React.useState("all");
    const [rowLimit, setRowLimit] = React.useState(DEFAULT_FILE_VIEW_LIMIT);
    const sortedRows = React.useMemo(
      () => sortFileRowsByFolder(rows, fileViewFolderLabel, (row) => row.name || row.path),
      [rows]
    );
    const folderChoices = React.useMemo(() => fileViewFolderChoices(sortedRows, fileViewFolderLabel), [sortedRows]);
    React.useEffect(() => {
      if (folderFilter !== "all" && !folderChoices.some((folder) => folder.key === folderFilter)) {
        setFolderFilter("all");
      }
    }, [folderFilter, folderChoices]);
    const filteredRows = folderFilter === "all"
      ? sortedRows
      : sortedRows.filter((row) => fileViewFolderLabel(row) === folderFilter);
    const renderedRows = filteredRows.slice(0, Math.min(rowLimit, filteredRows.length));
    const pickerClass = classNames(
      "workspace-file-picker",
      group === "fine-tune-audio" && "workspace-file-picker-audio",
      group === "fine-tune-rttm" && "workspace-file-picker-rttm"
    );
    return h(
      "div",
      { className: pickerClass },
      rows.length
        ? h(
            React.Fragment,
            null,
            h(
              "div",
              { className: "selection-toolbar file-view-toolbar workspace-file-toolbar" },
              h(FolderFilterControl, { id: `${group}_folder_filter`, value: folderFilter, onChange: setFolderFilter, folders: folderChoices }),
              h(FileViewLimitControl, {
                id: `${group}_row_limit`,
                total: filteredRows.length,
                shown: renderedRows.length,
                limit: rowLimit,
                onLimitChange: setRowLimit,
              })
            ),
            renderedRows.map((file) =>
              h(
                "label",
                { key: file.path, className: "checkbox-row file-choice", title: file.path || "" },
                h("input", { type: "checkbox", name, value: file.path, "data-check-group": group }),
                h(
                  "span",
                  { className: "file-choice-text" },
                  h("strong", { className: "file-choice-name" }, file.name || file.path),
                  h("small", { className: "file-choice-path" }, fileViewFolderLabel(file))
                )
              )
            )
          )
        : h("p", { className: "field-status" }, emptyText)
    );
  }

  function FineTuneCreateProjectCard({ preferences }) {
    // Lightweight escape hatch from the upload-first workflow: register a
    // fine-tuning project name (and an optional display name) before any
    // labels exist. Lets the user reserve a slot in the popup's "existing
    // models" list and send Inspect-completed labels at it later.
    return h(
      "section",
      { className: "subpanel", id: "fine-tune-create" },
      h("div", { className: "panel-head" }, h("div", null, h("h2", null, "0. Create Empty Project"), h("p", null, "Reserve a fine-tuned model slot now and feed it labels later."))),
      h(
        "form",
        { method: "post", action: routes.fineTuneCreateProject, "data-loading-message": "Creating fine-tuning project..." },
        h(Field, { id: "create_project_backend", label: "Backend" }, h(SelectInput, { id: "create_project_backend", name: "fine_tuning_backend", defaultValue: preferences.fine_tuning_backend || "pyannote" }, h(Option, { value: "nemo" }, "NeMo"), h(Option, { value: "pyannote" }, "pyannote"))),
        h(Field, { id: "create_project_name", label: "Project name (slug)", note: "Used as the on-disk folder. Letters, numbers, dashes." }, h("input", { id: "create_project_name", name: "project_name", placeholder: "callhome-msdd", required: true })),
        h(Field, { id: "create_project_display", label: "Display name (optional)", note: "Friendly label shown in the popup and project cards. Slug stays the same on disk." }, h("input", { id: "create_project_display", name: "display_name", placeholder: "Callhome MSDD" })),
        h("p", null, h("button", { type: "submit" }, "Create empty project"))
      )
    );
  }

  function FineTuneUploadCard({ preferences, projectNames, trainingSources }) {
    const sources = trainingSources || {};
    const audioFiles = sources.audioFiles || [];
    const rttmFiles = sources.rttmFiles || [];
    const transcriptFiles = sources.transcriptFiles || [];
    const labelHref = navItems.find((item) => item.path === "/training-labels")?.href || "/training-labels";
    // Build the existing-project picker the same way the stitching tab does:
    // each row is a (backend, slug, displayName, sampleCount) tuple, keyed
    // off ``<backend>/<slug>``. Multi-select fans the upload out to every
    // checked target on submit.
    const projectOptions = React.useMemo(
      () => (ctx.projects || [])
        .filter((project) => project && project.slug && project.backend)
        .map((project) => ({
          key: `${project.backend}/${project.slug}`,
          backend: project.backend,
          slug: project.slug,
          label: `${backendDisplayName(project.backend)} / ${project.displayName || project.slug}`,
          sampleCount: project.sampleCount || 0,
        }))
        .sort((a, b) => a.label.toLowerCase().localeCompare(b.label.toLowerCase())),
      [JSON.stringify((ctx.projects || []).map((p) => [p.backend, p.slug, p.displayName, p.sampleCount]))]
    );
    const [trainingTargets, setTrainingTargets] = React.useState(() => new Set());
    const [includeManualTarget, setIncludeManualTarget] = React.useState(() => projectOptions.length === 0);
    React.useEffect(() => {
      const valid = new Set(projectOptions.map((p) => p.key));
      setTrainingTargets((current) => new Set(Array.from(current).filter((k) => valid.has(k))));
      if (projectOptions.length === 0) setIncludeManualTarget(true);
    }, [projectOptions.map((p) => p.key).join("|")]);
    const toggleTarget = (key, checked) =>
      setTrainingTargets((current) => {
        const next = new Set(current);
        if (checked) next.add(key); else next.delete(key);
        return next;
      });
    const totalTargets = trainingTargets.size + (includeManualTarget ? 1 : 0);
    return h(
      "section",
      { className: "subpanel", id: "fine-tune-upload" },
      h("div", { className: "panel-head" }, h("div", null, h("h2", null, "1. Add Training Samples"), h("p", null, "Select SSH workspace audio + matching RTTM labels and pick one or more fine-tune projects to receive them. Each sample is copied into every selected project."))),
      h(
        "form",
        {
          method: "post",
          action: routes.fineTuneUpload,
          "data-loading-message": "Adding training sample batch...",
          onSubmit: (event) => {
            const manualField = event.currentTarget.elements.namedItem("project_name");
            const manualValue = manualField && "value" in manualField ? text(manualField.value).trim() : "";
            if (trainingTargets.size === 0 && (!includeManualTarget || !manualValue)) {
              event.preventDefault();
              window.alert("Pick at least one existing fine-tune project, or fill in the additional project name.");
            }
          },
        },
        Array.from(trainingTargets).map((key) =>
          h("input", { key: `tt-${key}`, type: "hidden", name: "training_targets", value: key, readOnly: true })
        ),
        h(
          "details",
          { className: "details-box compact-details", open: true },
          h("summary", null, "Fine-Tune Targets"),
          h(
            "div",
            null,
            projectOptions.length
              ? h(
                  React.Fragment,
                  null,
                  h(
                    "div",
                    { className: "selection-toolbar compact-toolbar" },
                    h("button", { className: "secondary", type: "button", onClick: () => setTrainingTargets(new Set(projectOptions.map((p) => p.key))) }, "Select all existing"),
                    h("button", { className: "ghost", type: "button", disabled: trainingTargets.size === 0, onClick: () => setTrainingTargets(new Set()) }, "Clear existing"),
                    h("span", { className: "field-status" }, `${totalTargets} target(s) selected`)
                  ),
                  h(
                    "div",
                    { className: "training-target-checklist" },
                    projectOptions.map((project) =>
                      h(
                        "label",
                        { key: project.key, className: "checkbox-row file-choice" },
                        h("input", {
                          type: "checkbox",
                          checked: trainingTargets.has(project.key),
                          onChange: (event) => toggleTarget(project.key, event.target.checked),
                        }),
                        h("span", null, h("strong", null, project.label), h("small", null, `${project.sampleCount} sample(s)`))
                      )
                    )
                  )
                )
              : h("p", { className: "field-status" }, "No existing fine-tuning projects yet. Fill in the additional project name below to create one."),
            h(
              "label",
              { className: "checkbox-row" },
              h("input", { type: "checkbox", checked: includeManualTarget, onChange: (event) => setIncludeManualTarget(event.target.checked) }),
              projectOptions.length ? " Also send to another or new project" : " Send to a new project"
            ),
            includeManualTarget
              ? h(
                  "div",
                  { className: "inline" },
                  h(Field, { id: "fine_tuning_backend", label: "Additional backend" }, h(SelectInput, { id: "fine_tuning_backend", name: "fine_tuning_backend", defaultValue: preferences.fine_tuning_backend || "pyannote" }, h(Option, { value: "pyannote" }, "pyannote"), h(Option, { value: "nemo" }, "NeMo"), h(Option, { value: "both" }, "NeMo + pyannote"))),
                  h(Field, { id: "project_name", label: "Additional project name", note: "Existing slug or a brand-new project." }, h("input", { id: "project_name", name: "project_name", list: "fine_tuning_project_names", placeholder: "callhome-msdd" }))
                )
              : null
          )
        ),
        projectNames.length ? h("datalist", { id: "fine_tuning_project_names" }, projectNames.map((name) => h("option", { key: name, value: name }))) : null,
        h(
          "section",
          { className: "fine-tune-file-section fine-tune-file-section-audio" },
          h("div", { className: "fine-tune-file-section-head" }, h("h3", null, "SSH audio in audio_in/"), h("p", null, `${audioFiles.length} file(s) found`)),
          h("div", { className: "selection-toolbar" }, h("button", { className: "secondary", type: "button", "data-select-group": "fine-tune-audio", "data-select-mode": "all" }, "Select shown"), h("button", { className: "ghost", type: "button", "data-select-group": "fine-tune-audio", "data-select-mode": "none" }, "Clear")),
          h(Field, { id: "server_audio_paths", label: "Audio files" }, h(WorkspaceFileChecklist, { group: "fine-tune-audio", name: "server_audio_paths", files: audioFiles, emptyText: "No audio files are present in audio_in/ yet." })),
          h("p", { className: "field-status" }, h("span", { "data-selection-count": "fine-tune-audio" }, "0"), " audio file(s) selected.")
        ),
        h(
          "section",
          { className: "fine-tune-file-section fine-tune-file-section-rttm" },
          h("div", { className: "fine-tune-file-section-head" }, h("h3", null, "SSH RTTM label files"), h("p", null, `${rttmFiles.length} file(s) found`)),
          h("div", { className: "selection-toolbar" }, h("button", { className: "secondary", type: "button", "data-select-group": "fine-tune-rttm", "data-select-mode": "all" }, "Select shown"), h("button", { className: "ghost", type: "button", "data-select-group": "fine-tune-rttm", "data-select-mode": "none" }, "Clear")),
          h(Field, { id: "server_rttm_paths", label: "RTTM label files" }, h(WorkspaceFileChecklist, { group: "fine-tune-rttm", name: "server_rttm_paths", files: rttmFiles, emptyText: h(React.Fragment, null, "No RTTM files found yet. Use ", h("a", { href: labelHref }, "Training Labels"), " to create labels from any audio_in/ file.") })),
          h("p", { className: "field-status" }, h("span", { "data-selection-count": "fine-tune-rttm" }, "0"), " RTTM file(s) selected.")
        ),
        h("details", { className: "details-box compact-details" }, h("summary", null, "Optional Transcripts"), h("div", null, h("div", { className: "selection-toolbar" }, h("button", { className: "secondary", type: "button", "data-select-group": "fine-tune-transcript", "data-select-mode": "all" }, "Select shown"), h("button", { className: "ghost", type: "button", "data-select-group": "fine-tune-transcript", "data-select-mode": "none" }, "Clear")), h(Field, { id: "server_transcript_paths", label: "SSH transcript files" }, h(WorkspaceFileChecklist, { group: "fine-tune-transcript", name: "server_transcript_paths", files: transcriptFiles, emptyText: "No optional transcript files found." })), h("p", { className: "field-status" }, h("span", { "data-selection-count": "fine-tune-transcript" }, "0"), " transcript file(s) selected."))),
        h(Field, { id: "training_transcript_text", label: "Optional shared transcript text" }, h("textarea", { id: "training_transcript_text", name: "training_transcript_text", placeholder: "Transcript text or annotation notes" })),
        h("p", { className: "footer-note" }, "Batch pairing uses filename stems: ", h("code", null, "001_clip.wav"), " pairs with ", h("code", null, "001_clip.rttm"), " and optional ", h("code", null, "001_clip.txt"), "."),
        h("p", null, h("button", { type: "submit" }, "Add samples"))
      )
    );
  }

  function FineTunePrepareCard({ preferences, sampleProjects }) {
    const nemo = preferences.nemo_fine_tuning || {};
    const pyannote = preferences.pyannote_fine_tuning || {};
    const preferredBackend = text(sampleProjects[0]?.backend || preferences.fine_tuning_backend || "pyannote").toLowerCase() === "nemo" ? "nemo" : "pyannote";
    const nemoBaseModel = nemo.speaker_model || defaults.speakerModel || "";
    const pyannoteBaseModel = pyannote.pretrained_model || defaults.pyannotePretrainedModel || "";
    const baseModelDefault = preferredBackend === "nemo" ? nemoBaseModel : pyannoteBaseModel;
    const disabled = sampleProjects.length === 0;
    return h(
      "section",
      { className: "subpanel" },
      h("div", { className: "panel-head" }, h("div", null, h("h2", null, "2. Prepare / Submit"), h("p", null, "Generate training files, or send the project straight to Slurm."))),
      h(
        "form",
        { method: "post", action: routes.fineTunePrepare, "data-loading-message": "Preparing fine-tuning artifacts..." },
        h(Field, { id: "prepare_project_choice", label: "Project with uploaded samples" }, h("select", { id: "prepare_project_choice", "data-project-name-target": "prepare_project_name", "data-project-backend-target": "prepare_backend" }, h(FineTuneProjectOptions, { projects: sampleProjects }))),
        h(Field, { id: "prepare_backend", label: "Backend" }, h(SelectInput, { id: "prepare_backend", defaultValue: preferredBackend, extra: { "data-base-model-backend": "true" } }, h(Option, { value: "nemo" }, "NeMo"), h(Option, { value: "pyannote" }, "pyannote"))),
        h(Field, { id: "prepare_project_name", label: "Project name" }, h(TextInput, { id: "prepare_project_name", placeholder: "callhome-msdd" })),
        h(
          "details",
          { className: "details-box compact-details" },
          h("summary", null, "Advanced Settings"),
          h("div", null,
            h("div", { className: "inline" }, h(Field, { id: "train_ratio", label: "Train ratio" }, h(TextInput, { id: "train_ratio", defaultValue: nemo.train_ratio || defaults.trainRatio })), h(Field, { id: "devices", label: "Devices" }, h(TextInput, { id: "devices", defaultValue: nemo.devices || "" }))),
            h("div", { className: "inline-3" }, h(Field, { id: "base_window", label: "NeMo base window" }, h(TextInput, { id: "base_window", defaultValue: nemo.base_window || defaults.baseWindow })), h(Field, { id: "base_shift", label: "NeMo base shift" }, h(TextInput, { id: "base_shift", defaultValue: nemo.base_shift || defaults.baseShift })), h(Field, { id: "step_count", label: "NeMo step count" }, h(TextInput, { id: "step_count", defaultValue: nemo.step_count || defaults.stepCount }))),
            h(Field, { id: "config_name", label: "NeMo config name" }, h(TextInput, { id: "config_name", defaultValue: nemo.config_name || defaults.configName })),
            h(Field, { id: "base_model", label: "Base model" }, h(TextInput, { id: "base_model", name: "base_model", defaultValue: baseModelDefault, extra: { "data-base-model-input": "true", "data-nemo-default": nemoBaseModel, "data-pyannote-default": pyannoteBaseModel, "data-active-backend": preferredBackend, title: "NeMo speaker model or pyannote pretrained model for the selected backend" } })),
            h("div", { className: "inline-3" }, h(Field, { id: "pyannote_duration", label: "pyannote chunk duration" }, h(TextInput, { id: "pyannote_duration", defaultValue: pyannote.duration || "" })), h(Field, { id: "pyannote_max_speakers_per_chunk", label: "Max speakers per chunk" }, h(TextInput, { id: "pyannote_max_speakers_per_chunk", defaultValue: pyannote.max_speakers_per_chunk || "" })), h(Field, { id: "pyannote_max_speakers_per_frame", label: "Max speakers per frame" }, h(TextInput, { id: "pyannote_max_speakers_per_frame", defaultValue: pyannote.max_speakers_per_frame || "" }))),
            h("div", { className: "inline" }, h(Field, { id: "max_epochs", label: "Max epochs" }, h(TextInput, { id: "max_epochs", defaultValue: nemo.max_epochs || defaults.maxEpochs })), h(Field, { id: "nemo_root", label: "Optional NeMo root" }, h(TextInput, { id: "nemo_root", placeholder: "path/to/NeMo" }))),
            h("div", { className: "inline-3" }, h(Field, { id: "slurm_partition", label: "Slurm partition" }, h(TextInput, { id: "slurm_partition", defaultValue: nemo.slurm_partition || defaults.slurmPartition })), h(Field, { id: "slurm_time", label: "Slurm time" }, h(TextInput, { id: "slurm_time", defaultValue: nemo.slurm_time || defaults.slurmTime })), h(Field, { id: "slurm_memory", label: "Slurm memory" }, h(TextInput, { id: "slurm_memory", defaultValue: nemo.slurm_memory || defaults.slurmMemory }))),
            h("div", { className: "inline" }, h(Field, { id: "slurm_cpus", label: "Slurm CPUs" }, h(TextInput, { id: "slurm_cpus", defaultValue: nemo.slurm_cpus || defaults.slurmCpus })), h(Field, { id: "slurm_gpus", label: "Slurm GPUs" }, h(TextInput, { id: "slurm_gpus", defaultValue: nemo.slurm_gpus || defaults.slurmGpus })))
          )
        ),
        h(
          "div",
          { className: "button-row" },
          h("button", { className: "secondary", type: "submit", disabled }, "Prepare Only"),
          h(
            "button",
            {
              className: "primary",
              type: "submit",
              disabled,
              formAction: routes.fineTunePrepareLaunch || routes.fineTunePrepare,
              "data-loading-message": "Preparing artifacts and submitting fine-tuning to Slurm...",
            },
            "Prepare + Submit to Sbatch"
          )
        )
      ),
      disabled ? h("p", { className: "field-status" }, "Upload at least one labeled sample before you prepare training artifacts.") : null
    );
  }

  function FineTuneLaunchCard({ preferences, preparedProjects }) {
    const disabled = preparedProjects.length === 0;
    return h(
      "section",
      { className: "subpanel" },
      h("div", { className: "panel-head" }, h("div", null, h("h2", null, "3. Launch Training"), h("p", null, "Choose local launch for a small check or Slurm for longer training."))),
      h(
        "form",
        { method: "post", action: routes.fineTuneLaunch },
        h(Field, { id: "launch_project_choice", label: "Prepared project" }, h("select", { id: "launch_project_choice", "data-project-name-target": "launch_project_name", "data-project-backend-target": "launch_backend" }, h(FineTuneProjectOptions, { projects: preparedProjects }))),
        h(Field, { id: "launch_backend", label: "Backend" }, h(SelectInput, { id: "launch_backend", defaultValue: preferences.fine_tuning_backend || "pyannote" }, h(Option, { value: "nemo" }, "NeMo"), h(Option, { value: "pyannote" }, "pyannote"))),
        h(Field, { id: "launch_project_name", label: "Prepared project name" }, h(TextInput, { id: "launch_project_name", placeholder: "callhome-msdd" })),
        h(Field, { id: "launch_version_name", label: "Trained version name", note: "Optional. Blank becomes '<project> trained version #', based on existing runs." }, h(TextInput, { id: "launch_version_name", placeholder: "campus-vad-cleanup" })),
        h(Field, { id: "launch_nemo_root", label: "Optional NeMo root" }, h(TextInput, { id: "launch_nemo_root", placeholder: "path/to/NeMo" })),
        h(Field, { id: "launch_python_bin", label: "Python binary" }, h(TextInput, { id: "launch_python_bin", defaultValue: defaults.pythonBin || "python3" })),
        h("label", { className: "checkbox-row" }, h("input", { type: "checkbox", name: "launch_local" }), " Force local launch instead of Slurm submission"),
        h("p", null, h("button", { type: "submit", disabled }, "Launch Background Training Run"))
      ),
      disabled ? h("p", { className: "field-status" }, "Prepare a project first. The launch button activates after artifacts exist.") : null
    );
  }

  // Per-project "auto-train when labels complete" toggle. Posts a hidden form
  // on toggle so the change persists in display.json and survives prepare cycles.
  // The pending badge is shown when a queued auto-train is waiting for the
  // currently-running training job to finish before it can take its turn.
  function AutoTrainToggle({ project }) {
    if (!project) return null;
    const enabled = Boolean(project.autoTrain);
    const pending = Boolean(project.autoTrainPending);
    function handleChange(event) {
      const next = Boolean(event.target.checked);
      const fields = {
        project_slug: project.slug,
        backend: project.backend,
      };
      if (next) {
        fields.auto_train_enabled = "1";
      }
      submitHiddenForm(routes.fineTuneAutoTrain, fields);
    }
    return h(
      "div",
      { className: "auto-train-row" },
      h(
        "label",
        { className: "checkbox-row", title: "When ON, completing a label for this project automatically queues prepare + sbatch. Multiple completions while a run is active are coalesced — they all get folded into the next training run." },
        h("input", { type: "checkbox", checked: enabled, onChange: handleChange }),
        " Auto-train when labels complete"
      ),
      pending
        ? h("p", { className: "row-note auto-train-pending" }, "Auto-train queued — waiting for the active run to finish before kicking off the next one.")
        : enabled
          ? h("p", { className: "row-note" }, "Mark a label complete and it will queue a training run automatically.")
          : null
    );
  }

  // Tiny SVG sparkline + summary row for the per-run val_loss series. The
  // points come straight from the parsed pyannote .out file, so this draws
  // whatever the trainer has emitted so far — empty until the first
  // validation epoch lands, then updated by the regular polling loop.
  function RunValLossSpark({ run }) {
    const points = Array.isArray(run?.valLoss) ? run.valLoss : [];
    if (!points.length) {
      return null;
    }
    const values = points
      .map((point) => Number(point && point.val_loss))
      .filter((value) => Number.isFinite(value));
    if (!values.length) {
      return null;
    }
    const minVal = Math.min(...values);
    const maxVal = Math.max(...values);
    const span = maxVal - minVal || 1;
    const width = 140;
    const height = 28;
    const pad = 1.5;
    const stepX = values.length > 1 ? (width - pad * 2) / (values.length - 1) : 0;
    const polyline = values
      .map((value, index) => {
        const x = pad + index * stepX;
        const y = height - pad - ((value - minVal) / span) * (height - pad * 2);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");
    const last = points[points.length - 1] || {};
    const epochsSeen = new Set();
    points.forEach((point) => {
      const epoch = Number(point && point.epoch);
      if (Number.isFinite(epoch)) {
        epochsSeen.add(epoch);
      }
    });
    const latestText = `epoch ${last.epoch || "?"}/${last.max_epochs || "?"}: ${formatNumber(values[values.length - 1], 4)}`;
    const minText = `min ${formatNumber(minVal, 4)}`;
    return h(
      "div",
      { className: "run-val-loss" },
      h(
        "svg",
        {
          className: "run-val-loss-spark",
          width,
          height,
          viewBox: `0 0 ${width} ${height}`,
          role: "img",
          "aria-label": `val_loss across ${values.length} validation point(s)`,
        },
        h("polyline", { points: polyline, fill: "none", stroke: "currentColor", strokeWidth: 1.4 })
      ),
      h(
        "span",
        { className: "row-note" },
        ` val_loss · ${latestText} · ${minText} · ${epochsSeen.size} epoch(s) · ${values.length} pts`
      )
    );
  }

  function FineTuneProjectCard({ project }) {
    const metrics = project.metrics || {};
    const splitText =
      metrics.trainCount || metrics.validationCount
        ? `${metrics.trainCount || 0} train / ${metrics.validationCount || 0} validation (${formatPercent((metrics.actualTrainRatio || 0) * 100)})`
        : "not prepared";
    const primaryMetrics = [
      ["Samples", project.sampleCount],
      ["Speech", formatDuration(metrics.totalSpeechSeconds || 0)],
      ["Coverage", formatPercent((metrics.speechCoverage || 0) * 100)],
      ["Speakers", metrics.uniqueSpeakerLabels || 0],
    ];
    const detailMetrics = [
      ["Audio length", formatDuration(metrics.totalAudioSeconds || 0)],
      ["Overlap", `${formatDuration(metrics.totalOverlapSeconds || 0)} / ${formatPercent((metrics.overlapCoverage || 0) * 100)}`],
      ["Segments", metrics.totalSegments || 0],
      ["Max speakers", metrics.maxConcurrentSpeakers || 0],
      ["Turns/min", formatNumber(metrics.speakerTurnsPerMinute || 0, 2)],
      ["Split", splitText],
      ["Prepared", project.prepared ? "yes" : "no"],
      ["Latest run", project.latestRunText],
    ];
    // The card title is the friendly name the user picked (or the slug
    // by default). The slug stays visible underneath because it's the
    // persistent ID — runs and label_status entries reference it.
    const displayName = project.displayName || project.slug;
    const showSlugSubtitle = displayName !== project.slug;
    return h(
      "article",
      { className: "project-card" },
      h(
        "div",
        { className: "panel-head compact-head" },
        h(
          "div",
          null,
          h("h3", null, `${backendDisplayName(project.backend)} / ${displayName}`),
          showSlugSubtitle
            ? h("p", { className: "row-note" }, `Slug: ${project.slug} (used by runs and label files)`)
            : null
        ),
        h(
          "button",
          {
            className: "ghost",
            type: "button",
            title: "Give this project a friendlier name. The on-disk slug stays the same so labels and runs keep referring to it.",
            onClick: () => promptRenameProject(project),
          },
          "Rename"
        )
      ),
      h(AutoTrainToggle, { project }),
      h(
        "div",
        { className: "project-metrics" },
        primaryMetrics.map(([label, value]) =>
          h("p", { key: label }, h("span", null, label), h("strong", null, value))
        )
      ),
      h("progress", { value: project.completedSteps, max: 3 }),
      h("p", { className: "progress-meta" }, `${project.completedSteps} of 3 stages complete. Upload-ready projects can prepare and submit in one click.`),
      h(
        "details",
        { className: "details-box compact-details project-details" },
        h("summary", null, "More project details"),
        h(
          "div",
          null,
          h(
            "div",
            { className: "project-metrics detail-metrics" },
            detailMetrics.map(([label, value]) =>
              h("p", { key: label }, h("span", null, label), h("strong", null, value))
            )
          ),
        project.recentRuns?.length
          ? h(
              "ul",
              { className: "compact-list" },
              project.recentRuns.map((run) => {
                const activeRun = ["running", "submitted"].includes(text(run.status).toLowerCase());
                return h(
                  "li",
                  { key: run.runName },
                  h("strong", null, run.displayName || run.versionName),
                  `: ${run.status}`,
                  run.displayName && run.versionName && run.displayName !== run.versionName
                    ? h("span", { className: "row-note" }, ` (v: ${run.versionName})`)
                    : null,
                  run.baseModel
                    ? h("span", { className: "row-note" }, ` · base: ${run.baseModel}`)
                    : null,
                  run.modelDeleted
                    ? h("span", { className: "row-note" }, " · model files deleted")
                    : null,
                  run.runDir
                    ? h(
                        "span",
                        { className: "run-inline-actions" },
                        h(
                          "button",
                          {
                            className: "ghost",
                            type: "button",
                            onClick: () => promptRenameRun(run),
                            title: "Give this training run a friendlier name. Useful when you've got several versions and need to remember which one was the best.",
                          },
                          "Rename"
                        ),
                        run.modelAvailable
                          ? h(
                              "button",
                              {
                                className: "ghost danger",
                                type: "button",
                                disabled: activeRun,
                                onClick: () => confirmDeleteFineTunedModel(run),
                                title: activeRun ? "Wait for this run to finish before deleting model artifacts." : "Delete checkpoint/model files for this fine-tuned version while keeping logs.",
                              },
                              "Delete Model"
                            )
                          : null
                      )
                    : null,
                  h(RunValLossSpark, { run })
                );
              })
            )
          : h("p", { className: "row-note" }, "No training runs yet."),
          project.warnings?.length ? h("ul", null, project.warnings.map((warning, index) => h("li", { key: index }, warning))) : h("p", { className: "row-note" }, "No preparation warnings.")
        )
      ),
      h(
        "div",
        { className: "project-actions" },
        Number(project.sampleCount || 0) > 0 && routes.fineTunePrepareLaunch
          ? h(
              "form",
              {
                method: "post",
                action: routes.fineTunePrepareLaunch,
                className: "inline-form quick-train-form",
                "data-loading-message": "Preparing artifacts and submitting fine-tuning to Slurm...",
              },
              h("input", { type: "hidden", name: "project_slug", value: project.slug }),
              h("input", { type: "hidden", name: "backend", value: project.backend }),
              h(
                "button",
                {
                  className: "primary",
                  type: "submit",
                  title: "Prepare this project with saved defaults and immediately submit a Slurm-preferred training run.",
                },
                "Prepare + Submit to Sbatch"
              )
            )
          : null,
        h("button", { className: "secondary", type: "button", "data-fill-project-name": project.slug, "data-fill-project-backend": project.backend, "data-fill-name-target": "project_name", "data-fill-backend-target": "fine_tuning_backend", "data-fill-focus-target": "project_name" }, "Use For Upload"),
        h("button", { className: "secondary", type: "button", "data-fill-project-name": project.slug, "data-fill-project-backend": project.backend, "data-fill-name-target": "prepare_project_name", "data-fill-backend-target": "prepare_backend", "data-fill-focus-target": "prepare_project_name" }, "Use For Prepare"),
        h("button", { className: "secondary", type: "button", "data-fill-project-name": project.slug, "data-fill-project-backend": project.backend, "data-fill-name-target": "launch_project_name", "data-fill-backend-target": "launch_backend", "data-fill-focus-target": "launch_project_name" }, "Use For Launch")
      ),
      h(LinkList, { links: project.links })
    );
  }

  function FindingsCalculator({ summary, projects }) {
    const [values, setValues] = React.useState(() => ({
      referenceSpeech: summary.total_speech_seconds ? String(summary.total_speech_seconds) : "",
      missedSpeech: "",
      falseAlarm: "",
      speakerConfusion: "",
      baselineDer: "",
      tunedDer: "",
    }));
    // The score-against-labels block lets you point at a reference RTTM
    // (the labeled ground truth) and a hypothesis RTTM (the model output)
    // and have the four DER inputs filled in for you. Way faster than
    // typing them in by hand once you've got a real run to compare.
    const [scoring, setScoring] = React.useState({
      referencePath: "",
      hypothesisPath: "",
      status: "idle", // idle | loading | error | success
      error: "",
      lastResult: null,
    });
    const updateValue = (key) => (event) => setValues((current) => ({ ...current, [key]: event.target.value }));
    const updateScoringValue = (key) => (event) => setScoring((current) => ({ ...current, [key]: event.target.value }));
    const runScoring = React.useCallback(() => {
      const reference = scoring.referencePath.trim();
      const hypothesis = scoring.hypothesisPath.trim();
      if (!reference || !hypothesis) {
        setScoring((current) => ({ ...current, status: "error", error: "Need both a reference and a hypothesis RTTM path." }));
        return;
      }
      const url = routes.fineTuneScoreRun;
      if (!url) {
        setScoring((current) => ({ ...current, status: "error", error: "Score endpoint isn't configured for this build." }));
        return;
      }
      setScoring((current) => ({ ...current, status: "loading", error: "" }));
      const params = new URLSearchParams({ reference, hypothesis });
      window
        .fetch(`${url}?${params.toString()}`, { cache: "no-store", headers: { Accept: "application/json" } })
        .then((response) => response.json().then((payload) => ({ ok: response.ok, payload })))
        .then(({ ok, payload }) => {
          if (!ok) {
            const message = payload && payload.error ? payload.error : "Score request failed.";
            setScoring((current) => ({ ...current, status: "error", error: message }));
            return;
          }
          // Auto-fill the four manual inputs below so the rest of the
          // calculator (baseline vs fine-tuned comparison, etc.) just works
          // off a real run instead of hand-typed numbers.
          setValues((current) => ({
            ...current,
            referenceSpeech: String(payload.reference_speech_seconds ?? ""),
            missedSpeech: String(payload.miss_seconds ?? ""),
            falseAlarm: String(payload.false_alarm_seconds ?? ""),
            speakerConfusion: String(payload.confusion_seconds ?? ""),
          }));
          setScoring((current) => ({ ...current, status: "success", error: "", lastResult: payload }));
        })
        .catch((err) => {
          setScoring((current) => ({ ...current, status: "error", error: String((err && err.message) || err) }));
        });
    }, [scoring.referencePath, scoring.hypothesisPath]);
    const referenceSpeech = numericValue(values.referenceSpeech);
    const missedSpeech = numericValue(values.missedSpeech);
    const falseAlarm = numericValue(values.falseAlarm);
    const speakerConfusion = numericValue(values.speakerConfusion);
    const der = referenceSpeech > 0 ? ((missedSpeech + falseAlarm + speakerConfusion) / referenceSpeech) * 100 : null;
    const tunedDer = values.tunedDer === "" ? der : numericValue(values.tunedDer);
    const baselineDer = values.baselineDer === "" ? null : numericValue(values.baselineDer);
    const absoluteReduction = baselineDer !== null && tunedDer !== null ? baselineDer - tunedDer : null;
    const relativeReduction = baselineDer && tunedDer !== null ? ((baselineDer - tunedDer) / baselineDer) * 100 : null;
    const bestProject = (projects || [])
      .slice()
      .sort((left, right) => Number(right.metrics?.totalSpeechSeconds || 0) - Number(left.metrics?.totalSpeechSeconds || 0))[0];
    return h(
      "article",
      { className: "panel", id: "fine-tuning-findings" },
      h("div", { className: "panel-head" }, h("div", null, h("h2", null, "Findings Calculator"), h("p", null, "Use these values when summarizing dataset size, DER, and model improvement."))),
      h(
        "div",
        { className: "summary-grid" },
        h(SummaryCard, { title: "Labeled Speech" }, h("p", null, h("strong", null, formatDuration(summary.total_speech_seconds || 0))), h("p", { className: "row-note" }, `${summary.total_samples || 0} sample(s), ${summary.total_segments || 0} segment(s)`)),
        h(SummaryCard, { title: "Speaker Labels" }, h("p", null, h("strong", null, summary.unique_speaker_labels || 0)), h("p", { className: "row-note" }, `${formatPercent((summary.speech_coverage || 0) * 100)} speech coverage across labeled audio.`)),
        h(SummaryCard, { title: "Overlap" }, h("p", null, h("strong", null, formatDuration(summary.total_overlap_seconds || 0))), h("p", { className: "row-note" }, `${formatPercent((summary.overlap_coverage || 0) * 100)} of active speech, ${summary.max_concurrent_speakers || 0} max concurrent speaker(s).`)),
        h(SummaryCard, { title: "Largest Project" }, h("p", null, h("strong", null, bestProject ? `${backendDisplayName(bestProject.backend)} / ${bestProject.displayName || bestProject.slug}` : "none")), h("p", { className: "row-note" }, bestProject ? formatDuration(bestProject.metrics?.totalSpeechSeconds || 0) : "No project samples yet."))
      ),
      h(
        "section",
        { className: "subpanel", style: { marginBottom: "16px" } },
        h("h3", null, "Score a model against your labels"),
        h(
          "p",
          { className: "row-note" },
          "Paste two RTTM paths from this project — the reference (your hand-labeled ground truth) and the hypothesis (whatever a diarization run produced). I'll compute DER, JER, and the miss / false-alarm / confusion split, then drop them into the inputs below."
        ),
        h(Field, { id: "score_reference_rttm", label: "Reference RTTM (labels)" },
          h("input", {
            id: "score_reference_rttm",
            type: "text",
            placeholder: "fine_tuning/projects/nemo/<project>/rttm/<clip>.rttm",
            value: scoring.referencePath,
            onChange: updateScoringValue("referencePath"),
          })
        ),
        h(Field, { id: "score_hypothesis_rttm", label: "Hypothesis RTTM (model output)" },
          h("input", {
            id: "score_hypothesis_rttm",
            type: "text",
            placeholder: "outputs/diarization_runs/<run>/<clip>.rttm",
            value: scoring.hypothesisPath,
            onChange: updateScoringValue("hypothesisPath"),
          })
        ),
        h(
          "div",
          { className: "button-row" },
          h(
            "button",
            { type: "button", onClick: runScoring, disabled: scoring.status === "loading" },
            scoring.status === "loading" ? "Scoring..." : "Score now"
          )
        ),
        scoring.status === "error"
          ? h("p", { className: "field-error" }, scoring.error || "Something went wrong.")
          : null,
        scoring.status === "success" && scoring.lastResult
          ? h(
              "div",
              { className: "row-note", style: { marginTop: "8px" } },
              h("p", null,
                h("strong", null, `DER: ${formatPercent(scoring.lastResult.der * 100, 2)}`),
                ` · JER ${formatPercent(scoring.lastResult.jer * 100, 2)}`,
                ` · ref ${formatNumber(scoring.lastResult.reference_speaker_count, 0)} speakers, hyp ${formatNumber(scoring.lastResult.hypothesis_speaker_count, 0)}.`
              ),
              h("p", null, `Filled in below: missed ${formatNumber(scoring.lastResult.miss_seconds, 2)}s · false alarm ${formatNumber(scoring.lastResult.false_alarm_seconds, 2)}s · confusion ${formatNumber(scoring.lastResult.confusion_seconds, 2)}s.`)
            )
          : null
      ),
      h(
        "div",
        { className: "split-grid" },
        h(
          "section",
          { className: "subpanel" },
          h("h3", null, "DER Inputs"),
          h(Field, { id: "calc_reference_speech", label: "Reference speech seconds" }, h("input", { id: "calc_reference_speech", type: "number", min: "0", step: "0.001", value: values.referenceSpeech, onChange: updateValue("referenceSpeech") })),
          h(Field, { id: "calc_missed_speech", label: "Missed speech seconds" }, h("input", { id: "calc_missed_speech", type: "number", min: "0", step: "0.001", value: values.missedSpeech, onChange: updateValue("missedSpeech") })),
          h(Field, { id: "calc_false_alarm", label: "False alarm seconds" }, h("input", { id: "calc_false_alarm", type: "number", min: "0", step: "0.001", value: values.falseAlarm, onChange: updateValue("falseAlarm") })),
          h(Field, { id: "calc_speaker_confusion", label: "Speaker confusion seconds" }, h("input", { id: "calc_speaker_confusion", type: "number", min: "0", step: "0.001", value: values.speakerConfusion, onChange: updateValue("speakerConfusion") }))
        ),
        h(
          "section",
          { className: "subpanel" },
          h("h3", null, "Presentation Outputs"),
          h("p", null, h("strong", null, "DER: "), der === null ? "enter reference speech" : formatPercent(der, 2)),
          h("p", { className: "row-note" }, "DER = (missed speech + false alarm + speaker confusion) / reference speech."),
          h("p", null, h("strong", null, "Diarization correctness proxy: "), der === null ? "n/a" : formatPercent(Math.max(0, 100 - der), 2)),
          h("p", null, h("strong", null, "Total audio: "), formatDuration(summary.total_audio_seconds || 0)),
          h("p", null, h("strong", null, "Active speech: "), formatDuration(summary.total_active_speech_seconds || 0)),
          h("p", null, h("strong", null, "Non-speech: "), formatDuration(summary.total_non_speech_seconds || 0)),
          h("p", null, h("strong", null, "Speaker turns/min: "), formatNumber(summary.speaker_turns_per_minute || 0, 2)),
          h("p", null, h("strong", null, "pyannote max speakers/frame target: "), summary.max_concurrent_speakers || 1),
          h("div", { className: "inline" }, h(Field, { id: "calc_baseline_der", label: "Baseline DER %" }, h("input", { id: "calc_baseline_der", type: "number", min: "0", step: "0.001", value: values.baselineDer, onChange: updateValue("baselineDer") })), h(Field, { id: "calc_tuned_der", label: "Fine-tuned DER %" }, h("input", { id: "calc_tuned_der", type: "number", min: "0", step: "0.001", value: values.tunedDer, onChange: updateValue("tunedDer"), placeholder: der === null ? "" : formatNumber(der, 2) }))),
          h("p", null, h("strong", null, "Absolute DER reduction: "), absoluteReduction === null ? "n/a" : `${formatNumber(absoluteReduction, 2)} points`),
          h("p", null, h("strong", null, "Relative DER reduction: "), relativeReduction === null ? "n/a" : formatPercent(relativeReduction, 2))
        )
      )
    );
  }

  // Maps DER (already scaled to 0-100 percent) to a coarse quality bucket so
  // the CSS can paint a green/yellow/red pill. Thresholds are by-eye, but they
  // help spot outliers way faster than scanning a column of numbers.
  function derQualityClass(percent) {
    if (!Number.isFinite(percent)) return "";
    if (percent <= 12) return "good";
    if (percent <= 25) return "warn";
    return "bad";
  }

  function describeRunOption(run) {
    if (!run) return "";
    const kindBadge = run.modelKind === "fine_tuned" ? " [fine-tuned]" : run.modelKind === "default" ? " [base]" : "";
    return `${run.modelLabel || run.backendLabel || run.backend}${kindBadge} — ${run.name} · ${run.fileCount} file${run.fileCount === 1 ? "" : "s"}`;
  }

  function DerCalculatorPanel() {
    // Model: pick a reference (hand labels OR a diarization run), then pick
    // one or more "hypothesis" runs to score against it. Same control flow
    // covers single-model DER, side-by-side DER vs labels, and pairwise
    // model-vs-model comparisons. Any saved diarization run — base or
    // fine-tuned — can be the reference or a hypothesis.
    const diarization = ctx.diarization || {};
    const runs = Array.isArray(diarization.runsForCompare) ? diarization.runsForCompare : [];
    const evaluable = Array.isArray(diarization.evaluableFiles) ? diarization.evaluableFiles : [];
    const labelHref = navItems.find((item) => item.path === "/training-labels")?.href || "/training-labels";

    const initialReferenceSource = evaluable.length > 0 ? "labels" : "run";
    const [referenceSource, setReferenceSource] = React.useState(initialReferenceSource);
    const [referenceRunPath, setReferenceRunPath] = React.useState(runs[0]?.path || "");
    const [selectedModelPaths, setSelectedModelPaths] = React.useState(() => {
      // Default selection: the first run (or two, when more than one is
      // available) so the panel is usable in one click. We exclude whatever
      // path is sitting in the reference dropdown.
      const initial = new Set();
      const refPath = runs[0]?.path || "";
      runs.forEach((run, index) => {
        if (run.path === refPath && initialReferenceSource === "run") return;
        if (initial.size < 2) initial.add(run.path);
      });
      return initial;
    });
    const [fileFilter, setFileFilter] = React.useState("");
    const [folderFilter, setFolderFilter] = React.useState("all");
    // Default off because the matched list is the headline output now;
    // toggling this on is for "what's missing diarization?" debugging.
    const [showSkipped, setShowSkipped] = React.useState(false);
    const [status, setStatus] = React.useState({ state: "idle", error: "", payload: null });

    const usingLabelsReference = referenceSource === "labels";
    const referenceRun = !usingLabelsReference ? runs.find((row) => row.path === referenceRunPath) || null : null;

    // Drop the chosen reference run from the model picker so the user can't
    // accidentally compare a run to itself.
    const eligibleModelRuns = runs.filter((run) => usingLabelsReference || run.path !== referenceRunPath);

    React.useEffect(() => {
      // When the reference run changes, scrub it from the selection set so
      // the submit doesn't quietly drop it server-side.
      setSelectedModelPaths((current) => {
        if (usingLabelsReference) return current;
        if (!current.has(referenceRunPath)) return current;
        const next = new Set(current);
        next.delete(referenceRunPath);
        return next;
      });
    }, [usingLabelsReference, referenceRunPath]);

    React.useEffect(() => {
      // Switching to labels mode shouldn't keep stale model picks if labels
      // can't reach those files; the file effect below will further trim.
      if (referenceSource === "labels" && evaluable.length === 0) {
        setReferenceSource("run");
      }
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [evaluable.length]);

    function describeRunInline(run) {
      if (!run) return "";
      const kindBadge = run.modelKind === "fine_tuned" ? " [fine-tuned]" : run.modelKind === "default" ? " [base]" : "";
      return `${run.modelLabel || run.backendLabel || run.backend}${kindBadge} — ${run.name} · ${run.fileCount} file${run.fileCount === 1 ? "" : "s"}`;
    }

    const selectedModelRuns = eligibleModelRuns.filter((run) => selectedModelPaths.has(run.path));
    const referenceRunFiles = !usingLabelsReference && referenceRun ? new Set(referenceRun.audioFiles || []) : null;

    // Candidate files = all completed-label audios (labels mode) OR every
    // audio that the reference run has an SRT for (model-as-reference mode).
    // Each candidate is annotated with which selected models actually have
    // an SRT so the file picker can grey out fully-unscoreable files.
    const candidateEntries = (usingLabelsReference
      ? evaluable.map((entry) => ({ name: entry.name, fileName: entry.fileName }))
      : (referenceRun?.audioFiles || []).map((name) => ({
          name,
          fileName: name.includes("/") ? name.split("/").pop() : name,
        }))
    ).map((entry) => {
      const presentInModels = selectedModelRuns.filter((run) => (run.audioFiles || []).includes(entry.name));
      return {
        ...entry,
        modelHits: presentInModels.length,
        modelTotal: selectedModelRuns.length,
      };
    });

    const filterTokens = fileFilter.trim().toLowerCase().split(/\s+/).filter(Boolean);
    const matchesFilter = (entry) => {
      if (!filterTokens.length) return true;
      const haystack = `${entry.name || ""} ${entry.fileName || ""}`.toLowerCase();
      return filterTokens.every((token) => haystack.includes(token));
    };

    // Folder choices come from the same fileViewFolderLabel helper the SSH
    // pickers use, so the user sees the same folder names (JIBOKids,
    // JIBOKids/blocks, etc.) here too.
    const folderChoices = React.useMemo(
      () => fileViewFolderChoices(candidateEntries, fileViewFolderLabel),
      [candidateEntries.map((entry) => entry.name).join("|")]
    );
    React.useEffect(() => {
      if (folderFilter !== "all" && !folderChoices.some((folder) => folder.key === folderFilter)) {
        setFolderFilter("all");
      }
    }, [folderFilter, folderChoices]);

    const matchesFolder = (entry) =>
      folderFilter === "all" || fileViewFolderLabel(entry) === folderFilter;

    const visibleEntries = candidateEntries
      .filter(matchesFilter)
      .filter(matchesFolder);
    const availableCount = candidateEntries.filter((entry) => entry.modelHits > 0).length;
    // Scoring auto-targets any matched file inside the active filter scope.
    // The user picks reference + model(s) once and the panel does the
    // pairing; no more checkbox-per-file ritual every time.
    const scoreableInScope = visibleEntries.filter((entry) => entry.modelHits > 0);

    function toggleModel(path) {
      setSelectedModelPaths((current) => {
        const next = new Set(current);
        if (next.has(path)) {
          next.delete(path);
        } else {
          next.add(path);
        }
        return next;
      });
    }

    function selectAllModels() {
      setSelectedModelPaths(new Set(eligibleModelRuns.map((run) => run.path)));
    }

    function clearAllModels() {
      setSelectedModelPaths(new Set());
    }

    function runScoring() {
      if (!routes.fineTuneCompareRuns) {
        setStatus({ state: "error", error: "Compare endpoint isn't configured for this build.", payload: null });
        return;
      }
      if (!usingLabelsReference && !referenceRunPath) {
        setStatus({ state: "error", error: "Pick a model run to use as the reference.", payload: null });
        return;
      }
      if (selectedModelPaths.size === 0) {
        setStatus({ state: "error", error: "Pick at least one model run to compare against the reference.", payload: null });
        return;
      }
      if (scoreableInScope.length === 0) {
        setStatus({
          state: "error",
          error: "No matched files in scope. Run diarization on these audios with the picked model(s) first, or widen the file/folder filter.",
          payload: null,
        });
        return;
      }
      const params = new URLSearchParams();
      params.set("reference_source", referenceSource);
      if (!usingLabelsReference) params.set("reference_run", referenceRunPath);
      Array.from(selectedModelPaths).forEach((path) => params.append("model", path));
      scoreableInScope.forEach((entry) => params.append("audio", entry.name));
      setStatus({ state: "loading", error: "", payload: null });
      window
        .fetch(`${routes.fineTuneCompareRuns}?${params.toString()}`, {
          cache: "no-store",
          headers: { Accept: "application/json" },
        })
        .then((response) => response.json().then((payload) => ({ ok: response.ok, payload })))
        .then(({ ok, payload }) => {
          if (!ok) {
            const message = payload && payload.error ? payload.error : "Scoring request failed.";
            setStatus({ state: "error", error: message, payload: null });
            return;
          }
          setStatus({ state: "success", error: "", payload });
        })
        .catch((err) => {
          setStatus({ state: "error", error: String((err && err.message) || err), payload: null });
        });
    }

    function modelKindBadge(run) {
      if (!run) return null;
      if (run.modelKind === "fine_tuned") return h("span", { className: "der-kind-pill der-kind-tuned" }, "fine-tuned");
      if (run.modelKind === "default") return h("span", { className: "der-kind-pill der-kind-base" }, "base");
      return null;
    }

    function modelHeading(run) {
      if (!run) return "Model";
      return h(
        React.Fragment,
        null,
        h("strong", null, run.modelLabel || run.backendLabel || run.backend),
        modelKindBadge(run),
        h("span", { className: "der-run-name" }, run.name)
      );
    }

    function derPill(percent) {
      const display = Number.isFinite(percent) ? `${formatNumber(percent, 2)}%` : "n/a";
      return h("span", { className: classNames("der-pill", derQualityClass(percent)) }, display);
    }

    function metricsBlock(metrics) {
      if (!metrics) return h("div", { className: "der-metric-block empty" }, "Not scored");
      if (metrics.error) return h("div", { className: "der-metric-block error" }, metrics.error);
      const derPercent = (metrics.der || 0) * 100;
      const jerPercent = (metrics.jer || 0) * 100;
      return h(
        "div",
        { className: "der-metric-block" },
        h(
          "div",
          { className: "der-metric-headline" },
          h("span", { className: "der-metric-label" }, "DER"),
          derPill(derPercent)
        ),
        h(
          "dl",
          { className: "der-metric-grid" },
          h("dt", null, "JER"),
          h("dd", null, formatPercent(jerPercent, 2)),
          h("dt", null, "Miss"),
          h("dd", null, `${formatNumber(metrics.miss_seconds || 0, 2)} s`),
          h("dt", null, "False alarm"),
          h("dd", null, `${formatNumber(metrics.false_alarm_seconds || 0, 2)} s`),
          h("dt", null, "Confusion"),
          h("dd", null, `${formatNumber(metrics.confusion_seconds || 0, 2)} s`),
          h("dt", null, "Speakers"),
          h("dd", null, `ref ${metrics.reference_speaker_count} · hyp ${metrics.hypothesis_speaker_count} (Δ ${metrics.speaker_count_diff})`)
        )
      );
    }

    function summaryCard(modelEntry, avg, bestDer) {
      const headerLabel = modelEntry.modelLabel || modelEntry.backendLabel || modelEntry.name;
      if (!avg) {
        return h(
          "div",
          { className: "der-summary-card empty", key: modelEntry.key },
          h("p", { className: "der-summary-label hyp" }, "Compared model"),
          h("p", { className: "der-summary-run" }, headerLabel),
          h("p", { className: "row-note" }, modelEntry.name),
          h("p", { className: "row-note" }, "No files scored.")
        );
      }
      const der = avg.weighted_der === null || avg.weighted_der === undefined ? null : avg.weighted_der * 100;
      const jer = avg.macro_jer * 100;
      const delta = bestDer !== null && bestDer !== undefined && der !== null ? der - bestDer : null;
      const isBest = der !== null && bestDer !== null && Math.abs(der - bestDer) < 0.001;
      return h(
        "div",
        { className: classNames("der-summary-card", isBest && "is-best"), key: modelEntry.key },
        h(
          "p",
          { className: "der-summary-label hyp" },
          "Compared model",
          isBest ? h("span", { className: "der-best-badge" }, "Best DER") : null
        ),
        h("p", { className: "der-summary-run" }, headerLabel),
        h("p", { className: "row-note" }, modelEntry.name),
        h("div", { className: "der-summary-headline" }, derPill(der)),
        h(
          "dl",
          { className: "der-summary-grid" },
          h("dt", null, "Macro JER"),
          h("dd", null, formatPercent(jer, 2)),
          h("dt", null, "Files scored / skipped"),
          h("dd", null, `${avg.files_scored} / ${avg.files_skipped}`),
          h("dt", null, "Reference speech"),
          h("dd", null, formatDuration(avg.reference_speech_seconds || 0)),
          h("dt", null, "Miss / FA / Conf"),
          h("dd", null, `${formatNumber(avg.miss_seconds || 0, 2)} / ${formatNumber(avg.false_alarm_seconds || 0, 2)} / ${formatNumber(avg.confusion_seconds || 0, 2)} s`)
        ),
        delta !== null && !isBest
          ? h(
              "p",
              { className: classNames("der-summary-delta", "bad") },
              `${formatNumber(delta, 2)} pts behind best`
            )
          : null
      );
    }

    function bestModelKey(metricsByKey, modelEntries) {
      let bestKey = null;
      let bestDer = Infinity;
      modelEntries.forEach((entry) => {
        const m = metricsByKey?.[entry.key];
        if (!m || m.error) return;
        if (m.der < bestDer) {
          bestDer = m.der;
          bestKey = entry.key;
        }
      });
      return { bestKey, bestDer: bestKey ? bestDer : null };
    }

    const result = status.payload;
    const filesRows = result?.files || [];
    const averages = result?.averages || {};
    const responseModels = result?.models || [];
    const responseModelLookup = new Map(responseModels.map((entry) => [entry.key, entry]));
    const responseModelMeta = responseModels.map((entry) => {
      const run = runs.find((r) => r.path === entry.path) || null;
      return {
        ...entry,
        modelLabel: run?.modelLabel || "",
        modelKind: run?.modelKind || "",
        backendLabel: run?.backendLabel || "",
      };
    });

    // Build a "best weighted DER among the result's models" so summary cards
    // can call out the leader and show the gap to it.
    let bestOverallDer = null;
    let bestOverallKey = null;
    responseModels.forEach((entry) => {
      const avg = averages[entry.key];
      if (!avg || avg.weighted_der === null || avg.weighted_der === undefined) return;
      const der = avg.weighted_der * 100;
      if (bestOverallDer === null || der < bestOverallDer) {
        bestOverallDer = der;
        bestOverallKey = entry.key;
      }
    });

    return h(
      "article",
      { className: "panel", id: "fine-tune-der" },
      h(
        "div",
        { className: "panel-head" },
        h(
          "div",
          null,
          h("h2", null, "DER Calculator"),
          h(
            "p",
            null,
            "Pick a reference (hand-labeled RTTMs or another diarization run), then check off one or more model runs to compare against it. Any saved run is fair game — base NeMo, base pyannote, fine-tuned checkpoints, or a mix. Per-file and weighted-overall DER, JER, miss / false alarm / speaker confusion are reported for each model."
          )
        )
      ),
      runs.length === 0
        ? h("p", { className: "row-note" }, "No diarization runs with SRT outputs are available yet. Run diarization first, then come back here.")
        : h(
              React.Fragment,
              null,
              // ---- REFERENCE section ----
              h(
                "section",
                { className: "subpanel der-reference-section" },
                h(
                  "div",
                  { className: "panel-head compact-head" },
                  h(
                    "div",
                    null,
                    h(
                      "h3",
                      null,
                      h("span", { className: "der-role-pill ref" }, "REFERENCE"),
                      " Ground truth to score against"
                    ),
                    h("p", { className: "row-note" }, "Lower DER vs. this reference = closer to the reference. With hand labels this is true diarization error; with another model as reference it's an agreement metric.")
                  )
                ),
                h(
                  "div",
                  { className: "der-reference-options" },
                  h(
                    "label",
                    { className: classNames("der-reference-option", usingLabelsReference && "is-selected", evaluable.length === 0 && "is-disabled") },
                    h("input", {
                      type: "radio",
                      name: "der-reference",
                      value: "labels",
                      checked: usingLabelsReference,
                      disabled: evaluable.length === 0,
                      onChange: () => setReferenceSource("labels"),
                    }),
                    h("span", { className: "der-reference-title" }, "Hand-labeled RTTMs (true DER)"),
                    h(
                      "span",
                      { className: "der-reference-detail" },
                      evaluable.length > 0
                        ? `${evaluable.length} file${evaluable.length === 1 ? "" : "s"} with completed labels available.`
                        : h(
                            React.Fragment,
                            null,
                            "No completed training labels yet. ",
                            h("a", { href: labelHref }, "Open Training Labels"),
                            " to finish at least one label."
                          )
                    )
                  ),
                  h(
                    "label",
                    { className: classNames("der-reference-option", !usingLabelsReference && "is-selected") },
                    h("input", {
                      type: "radio",
                      name: "der-reference",
                      value: "run",
                      checked: !usingLabelsReference,
                      onChange: () => setReferenceSource("run"),
                    }),
                    h("span", { className: "der-reference-title" }, "A diarization run (model-as-reference / agreement)"),
                    h("span", { className: "der-reference-detail" }, "No labels needed. The chosen run's SRTs are treated as ground truth; results show how far the other models drift from it.")
                  )
                ),
                !usingLabelsReference
                  ? h(
                      "div",
                      { className: "der-reference-picker" },
                      h(
                        Field,
                        { id: "der_reference_run", label: "Reference run", note: referenceRun ? `${referenceRun.modelLabel || referenceRun.backendLabel} · ${referenceRun.fileCount} file${referenceRun.fileCount === 1 ? "" : "s"} · last run ${referenceRun.lastRun || "unknown"}` : null },
                        h(
                          "select",
                          { id: "der_reference_run", value: referenceRunPath, onChange: (event) => setReferenceRunPath(event.target.value) },
                          runs.map((run) => h("option", { key: run.path, value: run.path }, describeRunOption(run)))
                        )
                      )
                    )
                  : null,
                usingLabelsReference
                  ? h(
                      "p",
                      { className: "der-reference-summary" },
                      h("span", { className: "der-role-pill ref small" }, "REFERENCE"),
                      ` Hand-labeled RTTMs · ${evaluable.length} file${evaluable.length === 1 ? "" : "s"} ready.`
                    )
                  : referenceRun
                    ? h(
                        "p",
                        { className: "der-reference-summary" },
                        h("span", { className: "der-role-pill ref small" }, "REFERENCE"),
                        ` ${referenceRun.modelLabel || referenceRun.backendLabel} — `,
                        h("span", { className: "mono" }, referenceRun.name)
                      )
                    : null
              ),
              // ---- COMPARED MODELS section ----
              h(
                "section",
                { className: "subpanel der-models-section" },
                h(
                  "div",
                  { className: "panel-head compact-head" },
                  h(
                    "div",
                    null,
                    h(
                      "h3",
                      null,
                      h("span", { className: "der-role-pill hyp" }, "COMPARED"),
                      ` Models to score against the reference (${selectedModelPaths.size} of ${eligibleModelRuns.length} selected)`
                    ),
                    h("p", { className: "row-note" }, "Pick one or more runs. Each gets its own column in the results. The reference run itself is hidden from this list so you can't accidentally compare it to itself.")
                  ),
                  h(
                    "div",
                    { className: "button-row" },
                    h("button", { className: "secondary", type: "button", onClick: selectAllModels }, "Select All"),
                    h("button", { className: "ghost", type: "button", onClick: clearAllModels }, "Clear")
                  )
                ),
                h(
                  "ul",
                  { className: "der-model-list" },
                  eligibleModelRuns.length === 0
                    ? h("li", { className: "der-file-empty" }, "No other runs available. You need at least one diarization run that isn't the reference.")
                    : eligibleModelRuns.map((run) => {
                        const checked = selectedModelPaths.has(run.path);
                        return h(
                          "li",
                          { key: run.path, className: classNames("der-model-row", checked && "is-checked") },
                          h(
                            "label",
                            { className: "checkbox-row" },
                            h("input", { type: "checkbox", checked, onChange: () => toggleModel(run.path) }),
                            h(
                              "span",
                              { className: "der-model-info" },
                              h(
                                "span",
                                { className: "der-model-title" },
                                h("strong", null, run.modelLabel || run.backendLabel || run.backend),
                                modelKindBadge(run)
                              ),
                              h("span", { className: "der-model-subtitle" }, `${run.name} · ${run.fileCount} file${run.fileCount === 1 ? "" : "s"} · last run ${run.lastRun || "unknown"}`)
                            )
                          )
                        );
                      })
                )
              ),
              // ---- MATCHES section ----
              // The panel auto-pairs the reference with whatever each picked
              // model has already diarized — no per-file checkbox needed.
              // Rows are read-only so the user can verify the matches before
              // hitting Score, and skipped-files can be expanded for a
              // sanity check.
              h(
                "section",
                { className: "subpanel" },
                h(
                  "div",
                  { className: "panel-head compact-head" },
                  h(
                    "div",
                    null,
                    h("h3", null, "Matched files"),
                    h(
                      "p",
                      { className: "row-note" },
                      `${scoreableInScope.length} ready to score in current filter · ${availableCount} matched in total · ${candidateEntries.length - availableCount} still need diarization on at least one picked model`
                    )
                  )
                ),
                // Hint banner: when none of the files in scope are present in
                // any chosen run, scoring is blocked; point the user at the
                // diarization tab so they can run inference first.
                availableCount === 0 && candidateEntries.length > 0 && selectedModelPaths.size > 0
                  ? h(
                      "p",
                      { className: "field-status der-no-runs-hint" },
                      "None of these files have been diarized by the selected run(s). ",
                      h("a", { href: "/diarization" }, "Run diarization first"),
                      ", then come back."
                    )
                  : null,
                h(
                  "div",
                  { className: "selection-toolbar compact-toolbar der-file-toolbar" },
                  folderChoices.length
                    ? h(FolderFilterControl, {
                        id: "der_calc_folder_filter",
                        value: folderFilter,
                        onChange: setFolderFilter,
                        folders: folderChoices,
                      })
                    : null,
                  h(
                    "label",
                    { className: "checkbox-row compact-checkbox" },
                    h("input", {
                      type: "checkbox",
                      checked: showSkipped,
                      onChange: (event) => setShowSkipped(event.target.checked),
                    }),
                    " Show files still missing diarization"
                  )
                ),
                h(
                  "div",
                  { className: "der-file-filter" },
                  h("input", {
                    type: "search",
                    placeholder: "Filter files…",
                    value: fileFilter,
                    onChange: (event) => setFileFilter(event.target.value),
                    "aria-label": "Filter audio files",
                  })
                ),
                h(
                  "ul",
                  { className: "der-file-list" },
                  visibleEntries.length === 0
                    ? h(
                        "li",
                        { className: "der-file-empty" },
                        usingLabelsReference
                          ? "No completed-label files match the current filters."
                          : "No files match — pick a reference run with files and at least one comparison model."
                      )
                    : visibleEntries
                        .filter((entry) => showSkipped || entry.modelHits > 0)
                        .map((entry) => {
                          const inReference = usingLabelsReference || (referenceRunFiles ? referenceRunFiles.has(entry.name) : false);
                          const available = entry.modelHits > 0 && (usingLabelsReference || inReference);
                          // Status colour-coded so coverage reads at a glance:
                          // green = matched, amber = partial across multiple
                          // compared models, red = blocked / needs diarization.
                          let statusKind = "ok";
                          let statusText = entry.modelTotal > 1 ? `${entry.modelHits}/${entry.modelTotal} models` : "Matched";
                          if (!available) {
                            statusKind = "blocked";
                            statusText = !inReference && !usingLabelsReference ? "Not in reference" : "No diarization yet";
                          } else if (entry.modelTotal > 1 && entry.modelHits < entry.modelTotal) {
                            statusKind = "partial";
                          }
                          return h(
                            "li",
                            { key: entry.name, className: classNames("der-file-row", !available && "is-missing", `der-file-row--${statusKind}`) },
                            h(
                              "div",
                              { className: "der-file-row-info" },
                              h("span", { className: "der-file-name" }, entry.fileName || entry.name),
                              h("span", { className: classNames("der-file-status", `der-file-status--${statusKind}`) }, statusText)
                            )
                          );
                        })
                )
              ),
              // ---- ACTION row ----
              h(
                "div",
                { className: "button-row" },
                h(
                  "button",
                  {
                    className: "primary",
                    type: "button",
                    onClick: runScoring,
                    disabled: status.state === "loading" || scoreableInScope.length === 0 || selectedModelPaths.size === 0,
                  },
                  status.state === "loading"
                    ? "Scoring…"
                    : selectedModelPaths.size === 1
                      ? `Calculate DER (${scoreableInScope.length} file${scoreableInScope.length === 1 ? "" : "s"})`
                      : `Compare ${selectedModelPaths.size} models (${scoreableInScope.length} file${scoreableInScope.length === 1 ? "" : "s"})`
                ),
                status.state === "error" ? h("span", { className: "form-status error" }, status.error) : null,
                status.state === "success" ? h("span", { className: "form-status success" }, `Scored ${filesRows.length} file(s).`) : null
              ),
              // ---- RESULTS ----
              status.state === "success" && result
                ? (() => {
                    const isPairwiseResult = result.reference?.source === "run";
                    return h(
                      "section",
                      { className: "subpanel der-results", style: { marginTop: "12px" } },
                      h("h3", null, "Results"),
                      // Reference banner — sits above the cards so the
                      // "what we scored against" question always has a
                      // one-line answer that's hard to miss.
                      h(
                        "div",
                        { className: "der-reference-banner" },
                        h("span", { className: "der-role-pill ref" }, "REFERENCE"),
                        isPairwiseResult
                          ? h(
                              React.Fragment,
                              null,
                              h("strong", null, "Diarization run"),
                              " — ",
                              h("span", { className: "mono" }, result.reference?.run?.name || ""),
                              h("p", { className: "row-note" }, "Numbers measure agreement against this run, not error against ground truth.")
                            )
                          : h(
                              React.Fragment,
                              null,
                              h("strong", null, "Hand-labeled RTTMs"),
                              h("p", { className: "row-note" }, "True DER against the labels you saved on the Training Labels page.")
                            )
                      ),
                      h(
                        "div",
                        { className: "der-summary-grid-cards", style: { gridTemplateColumns: `repeat(auto-fit, minmax(220px, 1fr))` } },
                        responseModelMeta.map((entry) => summaryCard(entry, averages[entry.key], bestOverallDer))
                      ),
                      h(
                        "div",
                        { className: "table-scroll" },
                        h(
                          "table",
                          { className: "der-results-table" },
                          h(
                            "thead",
                            null,
                            h(
                              "tr",
                              null,
                              h("th", null, "Audio"),
                              responseModelMeta.map((entry) =>
                                h(
                                  "th",
                                  { key: `head-${entry.key}` },
                                  h(
                                    "div",
                                    { className: "der-column-head" },
                                    h("span", { className: "der-role-pill hyp small" }, "HYP"),
                                    modelHeading(entry)
                                  )
                                )
                              ),
                              responseModelMeta.length > 1 ? h("th", null, "Best DER") : null
                            )
                          ),
                          h(
                            "tbody",
                            null,
                            filesRows.map((row, index) => {
                              if (row.error) {
                                const colspan = 1 + responseModelMeta.length + (responseModelMeta.length > 1 ? 1 : 0);
                                return h(
                                  "tr",
                                  { key: `${row.audio}-${index}`, className: "der-row error" },
                                  h("td", null, h("strong", null, row.audio)),
                                  h("td", { className: "score-error", colSpan: colspan - 1 }, row.error)
                                );
                              }
                              const { bestKey } = bestModelKey(row.metrics, responseModelMeta);
                              const refDescription = isPairwiseResult
                                ? `Reference SRT: ${row.reference_rttm}`
                                : `Reference RTTM: ${row.reference_rttm}`;
                              const bestEntry = bestKey ? responseModelLookup.get(bestKey) : null;
                              const bestLabel = bestEntry ? (bestEntry.name || bestEntry.key) : null;
                              return h(
                                "tr",
                                { key: `${row.audio}-${index}`, className: "der-row" },
                                h(
                                  "td",
                                  { className: "der-audio-cell" },
                                  h("strong", null, row.audio),
                                  h("p", { className: "row-note" }, refDescription)
                                ),
                                responseModelMeta.map((entry) =>
                                  h(
                                    "td",
                                    { key: `cell-${entry.key}-${index}`, className: classNames(bestKey === entry.key && "is-best-cell") },
                                    metricsBlock(row.metrics?.[entry.key])
                                  )
                                ),
                                responseModelMeta.length > 1
                                  ? h(
                                      "td",
                                      { className: "der-winner-cell" },
                                      bestLabel
                                        ? h("span", { className: "der-winner a", title: bestLabel }, bestLabel.length > 28 ? `${bestLabel.slice(0, 25)}…` : bestLabel)
                                        : h("span", { className: "der-winner none" }, "—")
                                    )
                                  : null
                              );
                            })
                          )
                        )
                      )
                    );
                  })()
                : null
            )
    );
  }

  function FineTuningPage() {
    const preferences = ctx.preferences || {};
    const projects = ctx.projects || [];
    const summary = ctx.fineTuningSummary || {};
    const labelSummary = ctx.trainingLabels?.summary || {};
    const labelHref = navItems.find((item) => item.path === "/training-labels")?.href || "/training-labels";
    const sampleProjects = projectChoices();
    const preparedProjects = projectChoices({ preparedOnly: true });
    const projectNames = Array.from(new Set(projects.map((project) => project.slug))).sort();
    return h(
      React.Fragment,
      null,
      h(
        "article",
        { className: "panel" },
        h("div", { className: "panel-head" }, h("div", null, h("h2", null, "Fine-Tuning"), h("p", null, "Use completed labels to prepare and launch training."))),
        h("div", { className: "summary-grid" }, h(SummaryCard, { title: "Labels" }, h("p", null, h("strong", null, labelSummary.completed || 0), " complete"), h("p", { className: "row-note" }, `${labelSummary.draft || 0} draft, ${labelSummary.needs_review || 0} need review`)), h(SummaryCard, { title: "Projects" }, h("p", null, h("strong", null, summary.projects_with_samples || 0), " with samples"), h("p", { className: "row-note" }, `${summary.prepared_projects || 0} prepared`)), h(SummaryCard, { title: "Labeled Speech" }, h("p", null, h("strong", null, formatDuration(summary.total_speech_seconds || 0))), h("p", { className: "row-note" }, `${summary.total_segments || 0} segment(s)`)), h(SummaryCard, { title: "Runs" }, h("p", null, h("strong", null, summary.active_runs || 0), " active"), h("p", { className: "row-note" }, "Submitted or running"))),
        h("div", { className: "mini-links" }, h("a", { href: labelHref }, "Open Training Labels"), h("a", { href: "#fine-tune-workflow" }, "Training Workflow"), h("a", { href: "#fine-tune-projects" }, "Projects"))
      ),
      h(
        "article",
        { className: "panel", id: "fine-tune-workflow" },
        h("div", { className: "panel-head" }, h("div", null, h("h2", null, "Training Workflow"), h("p", null, "Only the actions needed to build and run a fine-tuning project."))),
        h(FineTuneCreateProjectCard, { preferences }),
        h("div", { className: "three-grid" }, h(FineTuneUploadCard, { preferences, projectNames, trainingSources: ctx.trainingSources || {} }), h(FineTunePrepareCard, { preferences, sampleProjects }), h(FineTuneLaunchCard, { preferences, preparedProjects }))
      ),
      h(DerCalculatorPanel),
      h(
        "article",
        { className: "panel", id: "fine-tune-projects" },
        h("div", { className: "panel-head" }, h("div", null, h("h2", null, "Fine-Tuning Projects"), h("p", null, "Each card summarizes preparation state, latest run state, and generated artifacts."))),
        h(
          "div",
          { className: "project-grid" },
          projects.length
            ? projects.map((project) => h(FineTuneProjectCard, { key: `${project.backend}/${project.slug}`, project }))
            : h(
                "article",
                { className: "project-card empty-card" },
                h("h3", null, "No fine-tuning projects yet"),
                h("p", null, "Once you upload a labeled audio + RTTM pair above, the project will show up here with sample counts, speech coverage, and run status."),
                h("p", { className: "row-note" }, "Use the Training Workflow cards above to upload, prepare, and launch.")
              )
        )
      )
    );
  }

  function Page() {
    switch (state.currentPath) {
      case "/uploads":
        return h(UploadsPage);
      case "/stitching":
        return h(StitchingPage);
      case "/training-labels":
        return h(TrainingLabelsPage);
      case "/youtube":
        return h(YoutubePage);
      case "/diarization":
        return h(DiarizationPage);
      case "/fine-tuning":
        return h(FineTuningPage);
      default:
        return h(OverviewPage);
    }
  }

  function updateSelectionCounter(group, scope) {
    const container = scope || document;
    const boxes = container.querySelectorAll(`input[type="checkbox"][data-check-group="${group}"]`);
    const checked = Array.from(boxes).filter((box) => box instanceof HTMLInputElement && box.checked).length;
    container.querySelectorAll(`[data-selection-count="${group}"]`).forEach((node) => {
      node.textContent = String(checked);
    });
    container.querySelectorAll(`button[data-selection-submit="${group}"]`).forEach((button) => {
      if (button instanceof HTMLButtonElement) {
        button.disabled = checked === 0;
      }
    });
  }

  function selectedValuesForSelect(select) {
    if (!(select instanceof HTMLSelectElement)) {
      return [];
    }
    if (select.multiple) {
      return Array.from(select.selectedOptions || []).map((option) => option.value).filter(Boolean);
    }
    return select.value ? [select.value] : [];
  }

  function readyValueForSelection(box, selectionKeys) {
    const keys = Array.isArray(selectionKeys) ? selectionKeys.filter(Boolean) : [selectionKeys].filter(Boolean);
    if (keys.length) {
      try {
        const byModel = JSON.parse(box.getAttribute("data-ready-by-model") || "{}");
        const modelValues = keys
          .filter((key) => Object.prototype.hasOwnProperty.call(byModel, key))
          .map((key) => byModel[key]);
        if (modelValues.length) {
          return modelValues.some((value) => value === "yes") ? "yes" : "no";
        }
      } catch (_error) {
        return box.getAttribute("data-ready") || "yes";
      }
    }
    return box.getAttribute("data-ready") || "yes";
  }

  // Same answer as closest("form"), but `node.form` is the native handle
  // and skips the tree walk on every input/select/textarea we visit.
  function formIdentityForControl(node) {
    const form = node.form || node.closest("form");
    return formIdentityForForm(form);
  }

  function formIdentityForForm(form) {
    return text(form?.getAttribute("data-form-key") || form?.id || form?.getAttribute("action") || "", "").trim();
  }

  // Snapshot/restore call this once per control per refresh — memoize the
  // per-form key so a page with hundreds of inputs doesn't pay it each time.
  function formControlKey(node, formKeyCache) {
    let formKey;
    if (formKeyCache) {
      const form = node.form || node.closest("form") || null;
      const cached = form ? formKeyCache.get(form) : formKeyCache.get(null);
      if (cached !== undefined) {
        formKey = cached;
      } else {
        formKey = formIdentityForForm(form);
        formKeyCache.set(form, formKey);
      }
    } else {
      formKey = formIdentityForControl(node);
    }
    if (node.id) {
      return `${formKey}:id:${node.id}`;
    }
    if (node instanceof HTMLInputElement) {
      const checkGroup = node.getAttribute("data-check-group");
      if (checkGroup) {
        return `${formKey}:check:${checkGroup}:${node.value}`;
      }
    }
    const name = text(node.getAttribute("name") || "", "").trim();
    if (!name) {
      return "";
    }
    if (node instanceof HTMLInputElement && (node.type === "checkbox" || node.type === "radio")) {
      return `${formKey}:choice:${name}:${node.value}`;
    }
    return `${formKey}:name:${name}`;
  }

  // Grabs every control's value before a re-render so we can put it back
  // afterward. Inner loop avoids forEach overhead and reuses the form-key
  // memo since this fires on every poll fingerprint change.
  function snapshotFormState() {
    const snapshot = {};
    const formKeyCache = new Map();
    const nodes = document.querySelectorAll("input, select, textarea");
    for (let i = 0; i < nodes.length; i += 1) {
      const node = nodes[i];
      if (!(node instanceof HTMLInputElement || node instanceof HTMLSelectElement || node instanceof HTMLTextAreaElement)) {
        continue;
      }
      if (node instanceof HTMLInputElement && node.type === "file") {
        continue;
      }
      const key = formControlKey(node, formKeyCache);
      if (!key) {
        continue;
      }
      if (node instanceof HTMLInputElement && (node.type === "checkbox" || node.type === "radio")) {
        snapshot[key] = { checked: node.checked };
        continue;
      }
      if (node instanceof HTMLSelectElement && node.multiple) {
        snapshot[key] = {
          values: Array.from(node.selectedOptions || []).map((option) => option.value),
        };
        continue;
      }
      snapshot[key] = { value: node.value };
    }
    return snapshot;
  }

  // Pair of snapshotFormState — pushes saved values back into the new DOM
  // after React reconciles. Same memo trick to keep the hot path quiet.
  function restoreFormState(snapshot) {
    if (!snapshot) {
      return;
    }
    const formKeyCache = new Map();
    const nodes = document.querySelectorAll("input, select, textarea");
    for (let i = 0; i < nodes.length; i += 1) {
      const node = nodes[i];
      if (!(node instanceof HTMLInputElement || node instanceof HTMLSelectElement || node instanceof HTMLTextAreaElement)) {
        continue;
      }
      if (node instanceof HTMLInputElement && node.type === "file") {
        continue;
      }
      const saved = snapshot[formControlKey(node, formKeyCache)];
      if (!saved) {
        continue;
      }
      if (node instanceof HTMLInputElement && (node.type === "checkbox" || node.type === "radio")) {
        node.checked = Boolean(saved.checked);
        continue;
      }
      if (node instanceof HTMLSelectElement && node.multiple) {
        const selectedValues = new Set(saved.values || []);
        const options = node.options || [];
        for (let j = 0; j < options.length; j += 1) {
          options[j].selected = selectedValues.has(options[j].value);
        }
        continue;
      }
      if (Object.prototype.hasOwnProperty.call(saved, "value")) {
        if (node instanceof HTMLSelectElement) {
          let hasOption = false;
          const options = node.options || [];
          for (let j = 0; j < options.length; j += 1) {
            if (options[j].value === saved.value) {
              hasOption = true;
              break;
            }
          }
          if (!hasOption) {
            continue;
          }
        }
        node.value = saved.value;
      }
    }
  }

  function updateFileSummary(input) {
    const summaryId = input.getAttribute("data-file-summary");
    if (!summaryId) {
      return;
    }
    const summary = document.getElementById(summaryId);
    if (!summary) {
      return;
    }
    const files = Array.from(input.files || []);
    const totalBytes = files.reduce((sum, file) => sum + (Number(file.size) || 0), 0);
    if (files.length === 0) {
      summary.textContent = "No file selected yet.";
    } else if (files.length === 1) {
      summary.textContent = `Loaded: ${files[0].name} (${formatBytes(totalBytes)})`;
    } else {
      summary.textContent = `Loaded ${files.length} files (${formatBytes(totalBytes)} total). First file: ${files[0].name}`;
    }
  }

  function syncProjectSelector(select) {
    const selected = select.options[select.selectedIndex];
    if (!selected) {
      return;
    }
    const nameTarget = document.getElementById(select.getAttribute("data-project-name-target") || "");
    const backendTarget = document.getElementById(select.getAttribute("data-project-backend-target") || "");
    if (nameTarget) {
      nameTarget.value = selected.value || "";
    }
    const backend = selected.getAttribute("data-project-backend");
    if (backendTarget && backend) {
      backendTarget.value = backend;
      syncFineTuneBaseModelInput(backendTarget);
    }
  }

  function syncFineTuneBaseModelInput(backendTarget, { force = false } = {}) {
    if (!(backendTarget instanceof HTMLSelectElement)) {
      return;
    }
    const form = backendTarget.closest("form") || document;
    const baseModelInput = form.querySelector("input[data-base-model-input]");
    if (!(baseModelInput instanceof HTMLInputElement)) {
      return;
    }
    const backend = text(backendTarget.value).toLowerCase() === "nemo" ? "nemo" : "pyannote";
    const previousBackend = text(baseModelInput.getAttribute("data-active-backend")).toLowerCase();
    const nextDefault = backend === "nemo"
      ? baseModelInput.getAttribute("data-nemo-default") || ""
      : baseModelInput.getAttribute("data-pyannote-default") || "";
    const previousDefault = previousBackend === "nemo"
      ? baseModelInput.getAttribute("data-nemo-default") || ""
      : baseModelInput.getAttribute("data-pyannote-default") || "";
    const userEdited = baseModelInput.getAttribute("data-user-edited") === "true";
    if (force || !userEdited || !baseModelInput.value || baseModelInput.value === previousDefault) {
      baseModelInput.value = nextDefault;
      baseModelInput.setAttribute("data-user-edited", "false");
    }
    baseModelInput.setAttribute("data-active-backend", backend);
  }

  function showLoadingVisual(message) {
    const visual = document.getElementById("loading-visual");
    if (!visual) {
      return;
    }
    const textNode = visual.querySelector("[data-loading-text]");
    if (textNode) {
      textNode.textContent = message || "Submitting request...";
    }
    visual.hidden = false;
  }

  function uploadProgressNodes(form) {
    const progressId = form?.getAttribute("data-upload-progress") || "";
    const tracker = progressId ? document.getElementById(progressId) : null;
    return {
      tracker,
      bar: tracker?.querySelector("[data-upload-progress-bar]") || null,
      status: tracker?.querySelector("[data-upload-progress-status]") || null,
      percent: tracker?.querySelector("[data-upload-progress-percent]") || null,
      meta: tracker?.querySelector("[data-upload-progress-meta]") || null,
    };
  }

  function setUploadProgress(form, { loaded = 0, total = 0, status = "", meta = "", percent = null } = {}) {
    const nodes = uploadProgressNodes(form);
    if (!nodes.tracker) {
      return;
    }
    nodes.tracker.hidden = false;
    const safeTotal = Math.max(Number(total) || 0, 0);
    const safeLoaded = Math.max(Number(loaded) || 0, 0);
    const computedPercent = percent === null && safeTotal > 0 ? Math.min(100, Math.max(0, (safeLoaded / safeTotal) * 100)) : percent;
    if (nodes.status) {
      nodes.status.textContent = status || "Uploading media files";
    }
    if (nodes.meta) {
      nodes.meta.textContent = meta || "Estimating time remaining...";
    }
    if (nodes.percent) {
      nodes.percent.textContent = computedPercent === null ? "Estimating" : `${formatNumber(computedPercent, 0)}%`;
    }
    if (nodes.bar instanceof HTMLProgressElement) {
      nodes.bar.max = 100;
      if (computedPercent === null) {
        nodes.bar.removeAttribute("value");
      } else {
        nodes.bar.value = Math.min(100, Math.max(0, computedPercent));
      }
    }
  }

  function selectedUploadFiles(form) {
    return Array.from(form.querySelectorAll("input[type='file']"))
      .filter((input) => input instanceof HTMLInputElement && !input.disabled)
      .flatMap((input) => Array.from(input.files || []));
  }

  function resetSubmittingForm(form) {
    if (!form) {
      return;
    }
    form.dataset.submitting = "";
    form.removeAttribute("aria-busy");
    form.querySelectorAll("button[type='submit'], input[type='submit']").forEach((button) => {
      if (button instanceof HTMLButtonElement || button instanceof HTMLInputElement) {
        button.disabled = false;
      }
    });
  }

  function submitFormWithUploadProgress(form, submitter) {
    if (!form || typeof XMLHttpRequest === "undefined") {
      return Promise.reject(new Error("Upload progress is unavailable."));
    }
    const method = text(form.method || "post", "post").toUpperCase();
    if (method === "GET") {
      return Promise.reject(new Error("Upload progress requires a POST form."));
    }
    const action = form.action || window.location.href;
    const formData = new FormData(form);
    if (submitter && submitter.name) {
      formData.set(submitter.name, submitter.value || "");
    }
    const files = selectedUploadFiles(form);
    const totalSelectedBytes = files.reduce((sum, file) => sum + (Number(file.size) || 0), 0);
    const startedAt = Date.now();
    let processingTimer = null;
    setUploadProgress(form, {
      loaded: 0,
      total: totalSelectedBytes,
      status: `Preparing ${files.length || "selected"} file(s)`,
      meta: totalSelectedBytes ? `${formatBytes(totalSelectedBytes)} selected.` : "Waiting for the browser to report upload size.",
      percent: totalSelectedBytes ? 0 : null,
    });

    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      const clearProcessingTimer = () => {
        if (processingTimer) {
          window.clearInterval(processingTimer);
          processingTimer = null;
        }
      };
      const rejectOnce = (error) => {
        clearProcessingTimer();
        reject(error);
      };
      xhr.open(method, action, true);
      xhr.withCredentials = true;
      xhr.setRequestHeader("Accept", "text/html,*/*");
      xhr.upload.addEventListener("progress", (event) => {
        const elapsedSeconds = Math.max((Date.now() - startedAt) / 1000, 0.001);
        const speed = event.loaded / elapsedSeconds;
        if (event.lengthComputable) {
          const remainingSeconds = speed > 0 ? Math.max((event.total - event.loaded) / speed, 0) : 0;
          setUploadProgress(form, {
            loaded: event.loaded,
            total: event.total,
            status: `Uploading ${formatBytes(event.loaded)} of ${formatBytes(event.total)}`,
            meta: speed > 0 ? `${formatBytes(speed)}/s - about ${formatDuration(remainingSeconds)} remaining` : "Estimating time remaining...",
          });
        } else {
          setUploadProgress(form, {
            loaded: event.loaded,
            total: totalSelectedBytes,
            status: `Uploading ${formatBytes(event.loaded)}`,
            meta: "The browser has not reported the final upload size yet.",
            percent: null,
          });
        }
      });
      xhr.upload.addEventListener("load", () => {
        const processingStartedAt = Date.now();
        setUploadProgress(form, {
          loaded: totalSelectedBytes,
          total: totalSelectedBytes || 1,
          status: "Upload received. Converting files...",
          meta: "The server is converting accepted media to WAV and refreshing the library.",
          percent: 100,
        });
        processingTimer = window.setInterval(() => {
          setUploadProgress(form, {
            loaded: totalSelectedBytes,
            total: totalSelectedBytes || 1,
            status: "Upload received. Converting files...",
            meta: `Server processing for ${formatDuration((Date.now() - processingStartedAt) / 1000)}.`,
            percent: 100,
          });
        }, 1000);
      });
      xhr.addEventListener("load", () => {
        clearProcessingTimer();
        if (xhr.status >= 200 && xhr.status < 400) {
          setUploadProgress(form, {
            loaded: totalSelectedBytes,
            total: totalSelectedBytes || 1,
            status: "Upload complete. Refreshing Media Library...",
            meta: "Loading the updated tracker.",
            percent: 100,
          });
          window.location.replace(xhr.responseURL || action);
          resolve();
          return;
        }
        reject(new Error(`Upload failed with ${xhr.status}`));
      });
      xhr.addEventListener("error", () => rejectOnce(new Error("Upload failed before the server responded.")));
      xhr.addEventListener("abort", () => rejectOnce(new Error("Upload was canceled.")));
      xhr.send(formData);
    });
  }

  function submitFormWithHistoryReplace(form, submitter) {
    if (!form || typeof window.fetch !== "function") {
      return Promise.reject(new Error("Fetch submit is unavailable."));
    }
    const method = text(form.method || "post", "post").toUpperCase();
    const action = form.action || window.location.href;
    const formData = new FormData(form);
    if (submitter && submitter.name) {
      formData.set(submitter.name, submitter.value || "");
    }
    return window
      .fetch(action, {
        method,
        body: method === "GET" ? null : formData,
        credentials: "same-origin",
        redirect: "follow",
        headers: { Accept: "text/html,*/*" },
      })
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Form submit failed with ${response.status}`);
        }
        window.location.replace(response.url || action);
      });
  }

  // Runs once after every full re-render. Three small DOM sync passes —
  // file summaries, selection counters, project-name selects.
  function postRenderSync() {
    const fileInputs = document.querySelectorAll("input[type='file'][data-file-summary]");
    for (let i = 0; i < fileInputs.length; i += 1) {
      const input = fileInputs[i];
      if (input instanceof HTMLInputElement) {
        updateFileSummary(input);
      }
    }
    // updateSelectionCounter rescans the form for the group, so calling it
    // once per (form, group) is enough — even when several counter nodes
    // share the group within one form.
    const selectionNodes = document.querySelectorAll("[data-selection-count]");
    const seenScopes = new WeakMap();
    for (let i = 0; i < selectionNodes.length; i += 1) {
      const node = selectionNodes[i];
      const group = node.getAttribute("data-selection-count");
      if (!group) {
        continue;
      }
      const scope = node.closest("form") || document;
      let groupsForScope = seenScopes.get(scope);
      if (groupsForScope && groupsForScope.has(group)) {
        continue;
      }
      if (!groupsForScope) {
        groupsForScope = new Set();
        seenScopes.set(scope, groupsForScope);
      }
      groupsForScope.add(group);
      updateSelectionCounter(group, scope);
    }
    const projectSelects = document.querySelectorAll("select[data-project-name-target]");
    for (let i = 0; i < projectSelects.length; i += 1) {
      const select = projectSelects[i];
      if (select instanceof HTMLSelectElement) {
        syncProjectSelector(select);
      }
    }
  }

  function snapshotOpenDialogs() {
    return Array.from(document.querySelectorAll("dialog[open][id]"))
      .map((dialog) => dialog.id)
      .filter(Boolean);
  }

  function restoreOpenDialogs(dialogIds) {
    dialogIds.forEach((dialogId) => {
      const dialog = document.getElementById(dialogId);
      if (!(dialog instanceof HTMLDialogElement) || dialog.open) {
        return;
      }
      try {
        dialog.showModal();
      } catch (_error) {
        // Ignore unsupported or disconnected dialogs during a refresh cycle.
      }
    });
  }

  function hasRefreshBlockingForm() {
    return Boolean(document.querySelector("dialog[open] form[data-pause-refresh='true']"));
  }

  function currentRefreshInterval() {
    const refreshInterval = Number((ctx.tracking || {}).refreshIntervalMs);
    return refreshInterval > 0 ? refreshInterval : LIVE_TRACKING_INTERVAL_MS;
  }

  function currentIdleRefreshInterval() {
    const refreshInterval = Number((ctx.tracking || {}).idleRefreshIntervalMs);
    return refreshInterval > 0 ? refreshInterval : IDLE_TRACKING_INTERVAL_MS;
  }

  function currentDiarizationModelKey(overrides) {
    const overrideValue = overrides && overrides.diarization_model_key;
    if (overrideValue) {
      return text(overrideValue).trim();
    }
    const modelSelect = document.getElementById("diarization_model_keys") || document.getElementById("diarization_model_key");
    if (modelSelect instanceof HTMLSelectElement) {
      const selected = selectedValuesForSelect(modelSelect)[0] || "";
      if (text(selected).trim()) {
        return text(selected).trim();
      }
    }
    return text(
      (ctx.diarization && ctx.diarization.selectedModelKey) ||
        (ctx.preferences && ctx.preferences.default_diarization_model_key) ||
        (ctx.preferences && ctx.preferences.default_backend) ||
        "nemo"
    ).trim();
  }

  function activeTrackingShouldRefresh(payload) {
    const livePayload = payload || {};
    const tracking = livePayload.tracking || ctx.tracking || {};
    const diarizationStatus = text(
      (livePayload.diarization && livePayload.diarization.latestRun && livePayload.diarization.latestRun.status) ||
        (ctx.diarization && ctx.diarization.latestRun && ctx.diarization.latestRun.status) ||
        ""
    ).toLowerCase();
    const youtubeStatus = text(
      (livePayload.youtube && livePayload.youtube.latestRun && livePayload.youtube.latestRun.status) ||
        (ctx.youtube && ctx.youtube.latestRun && ctx.youtube.latestRun.status) ||
        ""
    ).toLowerCase();
    const activeRunCount = Number(tracking.activeRunCount || 0);
    return (
      Boolean(tracking.shouldRefresh) ||
      activeRunCount > 0 ||
      ACTIVE_RUN_STATUSES.includes(diarizationStatus) ||
      ACTIVE_RUN_STATUSES.includes(youtubeStatus)
    );
  }

  function nextTrackingDelay(payload) {
    return activeTrackingShouldRefresh(payload) ? currentRefreshInterval() : currentIdleRefreshInterval();
  }

  function pageStateUrl(overrides) {
    if (!routes.pageState) {
      return "";
    }
    const params = new URLSearchParams();
    const nextPath = text((overrides && overrides.path) || state.currentPath || "/");
    params.set("path", nextPath);
    if (nextPath === "/diarization") {
      const diarizationModelKey = currentDiarizationModelKey(overrides);
      if (diarizationModelKey) {
        params.set("diarization_model_key", diarizationModelKey);
      }
    }
    const separator = routes.pageState.includes("?") ? "&" : "?";
    return `${routes.pageState}${separator}${params.toString()}`;
  }

  function clearTrackingTimer() {
    if (refreshTimer) {
      window.clearTimeout(refreshTimer);
      refreshTimer = null;
    }
  }

  function scheduleTrackingPoll(delay) {
    clearTrackingTimer();
    if (trackingStopped) {
      return;
    }
    const delayMs = delay === undefined ? nextTrackingDelay() : delay;
    refreshTimer = window.setTimeout(runTrackingPoll, Math.max(0, delayMs));
  }

  function restartTrackingPolling(immediate) {
    clearTrackingTimer();
    if (trackingStopped) {
      return;
    }
    currentFingerprint = text((ctx.tracking || {}).fingerprint || currentFingerprint);
    scheduleTrackingPoll(immediate ? 0 : nextTrackingDelay());
  }

  function refreshCurrentPageState(overrides) {
    const url = pageStateUrl(overrides);
    if (!url || typeof window.fetch !== "function") {
      return Promise.reject(new Error("Page-state refresh is unavailable."));
    }
    return window
      .fetch(url, {
        cache: "no-store",
        headers: { Accept: "application/json" },
      })
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Page-state request failed with ${response.status}`);
        }
        return response.json();
      });
  }

  function applyState(nextState, options) {
    const settings = options || {};
    const openDialogIds = settings.preserveDialogs ? snapshotOpenDialogs() : [];
    const formState = settings.preserveFormState ? snapshotFormState() : null;
    syncDerivedState(nextState || initialState);
    root.render(h(Dashboard));
    const afterRender = () => {
      restoreFormState(formState);
      postRenderSync();
      if (openDialogIds.length) {
        restoreOpenDialogs(openDialogIds);
      }
      restartTrackingPolling(Boolean(settings.immediatePoll));
    };
    if (typeof window.requestAnimationFrame === "function") {
      window.requestAnimationFrame(afterRender);
    } else {
      window.setTimeout(afterRender, 0);
    }
  }

  function runTrackingPoll() {
    refreshTimer = null;
    if (trackingStopped) {
      return;
    }
    // Skip the network round-trip when the user has the tab in the background
    // — modern browsers throttle hidden timers anyway, but the dashboard's
    // per-second cadence still spends bandwidth and CPU on a tab the user is
    // not looking at. The visibilitychange listener below kicks an immediate
    // poll the moment they come back, so freshness is preserved.
    if (typeof document !== "undefined" && document.hidden) {
      scheduleTrackingPoll(currentIdleRefreshInterval());
      return;
    }
    if (hasRefreshBlockingForm()) {
      scheduleTrackingPoll(currentIdleRefreshInterval());
      return;
    }
    if (!routes.tracking || typeof window.fetch !== "function") {
      scheduleTrackingPoll();
      return;
    }
    window
      .fetch(routes.tracking, {
        cache: "no-store",
        headers: { Accept: "application/json" },
      })
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Tracking request failed with ${response.status}`);
        }
        return response.json();
      })
      .then((payload) => {
        if (trackingStopped) {
          return;
        }
        const liveTracking = payload && payload.tracking ? payload.tracking : {};
        const nextFingerprint = text(liveTracking.fingerprint || payload?.fingerprint || "");
        if (currentFingerprint && nextFingerprint && nextFingerprint !== currentFingerprint) {
          return refreshCurrentPageState()
            .then((nextState) => {
              if (!trackingStopped) {
                applyState(nextState, { preserveDialogs: true, preserveFormState: true });
              }
            })
            .catch(() => {
              if (!trackingStopped) {
                restartTrackingPolling();
              }
            });
        }
        if (nextFingerprint && !currentFingerprint) {
          currentFingerprint = nextFingerprint;
        }
        scheduleTrackingPoll(nextTrackingDelay(payload));
      })
      .catch(() => {
        if (!trackingStopped) {
          scheduleTrackingPoll(currentIdleRefreshInterval());
        }
      });
  }

  function Dashboard() {
    React.useEffect(() => {
      function handleChange(event) {
        const target = event.target;
        if (target instanceof HTMLInputElement) {
          const checkboxGroup = target.getAttribute("data-check-group");
          if (checkboxGroup) {
            updateSelectionCounter(checkboxGroup, target.form || document);
          }
          if (target.hasAttribute("data-base-model-input")) {
            target.setAttribute("data-user-edited", "true");
          }
          if (target.type === "file") {
            updateFileSummary(target);
          }
        }
        if (target instanceof HTMLSelectElement && target.hasAttribute("data-project-name-target")) {
          syncProjectSelector(target);
        }
        if (target instanceof HTMLSelectElement && target.hasAttribute("data-base-model-backend")) {
          syncFineTuneBaseModelInput(target);
        }
        if (target instanceof HTMLSelectElement && target.id === "uploads_model_filter") {
          uiState.uploadsModelFilter = target.value;
        }
        if (target instanceof HTMLSelectElement && (target.id === "diarization_model_keys" || target.id === "diarization_model_key")) {
          refreshCurrentPageState({ diarization_model_key: selectedValuesForSelect(target)[0] || target.value })
            .then((nextState) => {
              if (!trackingStopped) {
                applyState(nextState, { preserveDialogs: true, preserveFormState: true });
              }
            })
            .catch(() => {
              if (!trackingStopped) {
                restartTrackingPolling();
              }
            });
        }
      }

      function handleSubmit(event) {
        const form = event.target;
        if (!(form instanceof HTMLFormElement)) {
          return;
        }
        if (event.defaultPrevented) {
          return;
        }
        if (form.hasAttribute("data-training-label-editor")) {
          syncTrainingLabelEditorForm(form);
        }
        if (form.dataset.submitting === "true") {
          event.preventDefault();
          return;
        }
        form.dataset.submitting = "true";
        form.setAttribute("aria-busy", "true");
        const submitter = event.submitter instanceof HTMLElement ? event.submitter : null;
        const message =
          submitter?.getAttribute("data-loading-message") ||
          form.getAttribute("data-loading-message") ||
          "Submitting request...";
        showLoadingVisual(message);
        if (form.hasAttribute("data-upload-progress") && typeof XMLHttpRequest !== "undefined") {
          event.preventDefault();
          submitFormWithUploadProgress(form, submitter).catch((error) => {
            resetSubmittingForm(form);
            setUploadProgress(form, {
              status: "Upload failed.",
              meta: error?.message || "Check the server and try again.",
              percent: 0,
            });
            window.alert("The upload could not be completed. Check the server and try again.");
          });
        } else if (form.getAttribute("data-replace-submit") === "true" && typeof window.fetch === "function") {
          event.preventDefault();
          submitFormWithHistoryReplace(form, submitter).catch(() => {
            resetSubmittingForm(form);
            window.alert("The form could not be submitted. Check the server and try again.");
          });
        }
        window.setTimeout(() => {
          if (form.dataset.submitting !== "true") {
            return;
          }
          form.querySelectorAll("button[type='submit'], input[type='submit']").forEach((button) => {
            if (button instanceof HTMLButtonElement || button instanceof HTMLInputElement) {
              button.disabled = true;
            }
          });
        }, 0);
      }

      function handleClick(event) {
        if (!(event.target instanceof HTMLElement)) {
          return;
        }
        const dialogButton = event.target.closest("[data-open-dialog]");
        if (dialogButton) {
          event.preventDefault();
          const dialog = document.getElementById(dialogButton.getAttribute("data-open-dialog") || "");
          if (dialog instanceof HTMLDialogElement) {
            dialog.showModal();
          }
          return;
        }
        const closeButton = event.target.closest("[data-close-dialog]");
        if (closeButton) {
          event.preventDefault();
          const dialog = closeButton.closest("dialog");
          if (dialog instanceof HTMLDialogElement) {
            dialog.close();
          }
          return;
        }
        const multiSelectButton = event.target.closest("button[data-multiselect-target]");
        if (multiSelectButton) {
          event.preventDefault();
          const targetId = multiSelectButton.getAttribute("data-multiselect-target") || "";
          const target = targetId ? document.getElementById(targetId) : null;
          if (target instanceof HTMLSelectElement) {
            const mode = multiSelectButton.getAttribute("data-multiselect-mode") || "all";
            const wantSelected = mode !== "none";
            Array.from(target.options || []).forEach((option) => {
              option.selected = wantSelected;
            });
            target.dispatchEvent(new Event("change", { bubbles: true }));
          }
          return;
        }
        const selectButton = event.target.closest("button[data-select-group]");
        if (selectButton) {
          event.preventDefault();
          const group = selectButton.getAttribute("data-select-group");
          const mode = selectButton.getAttribute("data-select-mode") || "all";
          const scope = selectButton.closest("form") || document;
          const selectionSelectId = selectButton.getAttribute("data-ready-select") || "";
          const selectionSelect = selectionSelectId ? document.getElementById(selectionSelectId) : null;
          const activeSelection = selectionSelect instanceof HTMLSelectElement ? selectedValuesForSelect(selectionSelect) : [];
          scope.querySelectorAll(`input[type="checkbox"][data-check-group="${group}"]`).forEach((box) => {
            if (!(box instanceof HTMLInputElement)) {
              return;
            }
            if (mode === "none") {
              box.checked = false;
            } else if (mode === "ready") {
              box.checked = readyValueForSelection(box, activeSelection) === "yes";
            } else {
              box.checked = true;
            }
          });
          updateSelectionCounter(group, scope);
          return;
        }
        const fillButton = event.target.closest("button[data-fill-project-name]");
        if (fillButton) {
          event.preventDefault();
          const projectName = fillButton.getAttribute("data-fill-project-name") || "";
          const projectBackend = fillButton.getAttribute("data-fill-project-backend") || "";
          const nameTarget = document.getElementById(fillButton.getAttribute("data-fill-name-target") || "");
          const backendTarget = document.getElementById(fillButton.getAttribute("data-fill-backend-target") || "");
          if (nameTarget) {
            nameTarget.value = projectName;
          }
          if (backendTarget && projectBackend) {
            backendTarget.value = projectBackend;
            syncFineTuneBaseModelInput(backendTarget);
          }
          const focusTarget = document.getElementById(fillButton.getAttribute("data-fill-focus-target") || "");
          if (focusTarget) {
            focusTarget.scrollIntoView({ behavior: "smooth", block: "center" });
            focusTarget.focus();
          }
        }
      }

      // When the tab regains focus after being hidden, the polling loop has
      // been parked on its idle cadence. Kick a fresh poll right away so the
      // dashboard catches up on whatever happened while the user was gone
      // (a Slurm job that finished, a YouTube download that landed, etc.).
      function handleVisibilityChange() {
        if (!document.hidden && !trackingStopped) {
          scheduleTrackingPoll(0);
        }
      }
      document.addEventListener("change", handleChange);
      document.addEventListener("submit", handleSubmit);
      document.addEventListener("click", handleClick);
      document.addEventListener("visibilitychange", handleVisibilityChange);
      return () => {
        document.removeEventListener("change", handleChange);
        document.removeEventListener("submit", handleSubmit);
        document.removeEventListener("click", handleClick);
        document.removeEventListener("visibilitychange", handleVisibilityChange);
        trackingStopped = true;
        clearTrackingTimer();
      };
    }, []);
    return h(AppLayout, null, h(Page));
  }

  applyState(initialState, { immediatePoll: true });
})();
