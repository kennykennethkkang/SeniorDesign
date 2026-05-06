/* global React, ReactDOM */
(function () {
  "use strict";

  const h = React.createElement;
  const initialState = JSON.parse(document.getElementById("dashboard-state").textContent);
  const APP_TITLE = "ML Speech Diarization";
  const LIVE_TRACKING_INTERVAL_MS = 1000;
  const IDLE_TRACKING_INTERVAL_MS = 5000;
  const ACTIVE_RUN_STATUSES = ["running", "submitted", "pending", "waiting", "configuring"];
  const THEME_STORAGE_KEY = "ml-speech-diarization-theme";
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

  function TextInput({ id, name, defaultValue, placeholder, type = "text" }) {
    return h("input", {
      id,
      name: name || id,
      type,
      defaultValue: defaultValue === undefined ? "" : defaultValue,
      placeholder,
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
    return h(
      "dialog",
      {
        id,
        className: "dashboard-dialog",
        "aria-modal": "true",
        "aria-labelledby": titleId,
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
        children
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
    const folderLabel = (row) => text(row.folder || "Unsorted Root", "Unsorted Root");
    const folderChoices = Array.from(
      rows.reduce((choices, row) => {
        const key = folderLabel(row);
        const current = choices.get(key) || { key, label: key, count: 0 };
        current.count += 1;
        choices.set(key, current);
        return choices;
      }, new Map()).values()
    ).sort((left, right) => {
      if (left.key === "Unsorted Root") return -1;
      if (right.key === "Unsorted Root") return 1;
      return left.label.toLowerCase().localeCompare(right.label.toLowerCase());
    });
    const folderKeys = folderChoices.map((folder) => folder.key);
    const visibleRows = selectedFolderKeys === null
      ? rows
      : rows.filter((row) => selectedFolderKeys.has(folderLabel(row)));
    const selectedVisibleCount = visibleRows.filter((row) => selected.has(row.name)).length;
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

    const allChecked = visibleRows.length > 0 && visibleRows.every((row) => selected.has(row.name));
    const someChecked = !allChecked && visibleRows.some((row) => selected.has(row.name));
    const headerRef = React.useRef(null);
    React.useEffect(() => {
      if (headerRef.current) {
        headerRef.current.indeterminate = someChecked;
      }
    }, [someChecked]);

    const toggleShownRows = (checked) => {
      setSelected((prev) => {
        const next = new Set(prev);
        visibleRows.forEach((row) => {
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
              ? `${visibleRows.length} shown of ${rows.length}; ${selectedVisibleCount} shown selected${selectedCount > selectedVisibleCount ? ` (${selectedCount} total)` : ""}`
              : "No audio files yet"
          )
        ),
        h(
          "div",
          { className: "audio-inventory-bulk-actions" },
          h(
            "button",
            { type: "button", className: "secondary btn-sm", disabled: !visibleRows.length, onClick: () => toggleShownRows(true) },
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
            disabled: !visibleRows.length,
            onChange: (event) => toggleShownRows(event.target.checked),
          }),
          "#",
          "Folder",
          "Filename",
          "Type",
          "Actions",
        ],
        rows: visibleRows,
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
            h("td", null, h("strong", { className: "file-name" }, row.fileName || row.name), row.path ? h("p", { className: "row-note" }, row.path) : null),
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
    return h(DataTable, {
      className: "diarized-files-table",
      headers: ["Audio", "Status", "Model / Run", "Files", "Inspect"],
      rows: rows || [],
      emptyText: h(
        "div",
        { className: "empty-state" },
        h("strong", null, "No diarized files yet."),
        "Pick one or more audio files on the Diarization tab and submit a run — transcripts, SRT timing, and review pages will appear here when each file finishes."
      ),
      renderRow: (row, index) => {
        const dialogId = dialogIdFor("media-artifact", row.audioFile, row.runName, index);
        const created = createdArtifactLabels(row.links);
        return h(
          "tr",
          { key: `${row.audioFile}-${row.runName}-${index}` },
          h(
            "td",
            null,
            h("strong", { className: "file-name" }, row.audioFile),
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
    });
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
      ["Run Core Processing", "Select audio, run diarization, then review generated files.", [["/diarization", "Diarization"], ["/uploads", "Diarized Files"]]],
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
      h("article", { className: "panel" }, h("div", { className: "panel-head" }, h("div", null, h("h2", null, "More Details"), h("p", null, "Open logs, history, and queue exceptions only when needed."))), h("div", { className: "button-row" }, h("button", { className: "secondary", type: "button", "data-open-dialog": "youtube-run-dialog" }, "Latest run"), h("button", { className: "secondary", type: "button", "data-open-dialog": "youtube-history-dialog" }, "History"), h("button", { className: "secondary", type: "button", "data-open-dialog": "youtube-issues-dialog" }, "Queue issues")), h(Dialog, { id: "youtube-run-dialog", title: "Latest Audio Conversion Run", detail: "Newest result counts, artifact files, and log previews." }, h(YoutubeRunPanel, { latestRun })), h(Dialog, { id: "youtube-history-dialog", title: "Recent Conversion History", detail: "Confirm whether a link already produced audio or needs another attempt." }, h(DataTable, { headers: ["Status", "Title / Video", "Audio File", "Note", "Last Attempt"], rows: youtube.history || [], emptyText: "No conversion history yet.", renderRow: (row, index) => h("tr", { key: `${row.title}-${index}` }, h("td", null, row.status), h("td", null, row.title), h("td", null, row.audioHref ? h("a", { href: row.audioHref }, row.audioFile || "Open audio") : row.audioFile), h("td", null, row.note), h("td", null, row.lastAttempt)) })), h(Dialog, { id: "youtube-issues-dialog", title: "Queue Issues", detail: "Retryable downloader errors stay available. No-data links are separated." }, h(DataTable, { headers: ["URL", "Summary", "Queue Result", "Last Attempt"], rows: youtube.retryRows || [], emptyText: "No retry-needed URLs are recorded right now.", renderRow: (row, index) => h("tr", { key: `${row.url}-${index}` }, h("td", null, row.url), h("td", null, row.summary), h("td", null, row.queue_result || "kept"), h("td", null, row.when)) }), h("h3", null, "URLs With No Public Audio Data"), h(DataTable, { headers: ["URL", "Summary", "Queue Result", "Last Attempt"], rows: youtube.noDataRows || [], emptyText: "No no-data URLs are recorded right now.", renderRow: (row, index) => h("tr", { key: `${row.url}-${index}` }, h("td", null, row.url), h("td", null, row.summary), h("td", null, row.queue_result || "removed"), h("td", null, row.when)) }))),
      h(ClusterQueuePanel, { title: "Cluster Queue" })
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
          h("div", { className: "selection-toolbar" }, h("button", { className: "secondary", type: "button", "data-select-group": "diarization-audio", "data-select-mode": "ready", "data-ready-select": "diarization_model_keys" }, "Select ready + retry"), h("button", { className: "secondary", type: "button", "data-select-group": "diarization-audio", "data-select-mode": "all" }, "Select all"), h("button", { className: "ghost", type: "button", "data-select-group": "diarization-audio", "data-select-mode": "none" }, "Clear"), h("label", { className: "checkbox-row" }, h("input", { type: "checkbox", name: "diarization_include_completed" }), " Allow re-running already diarized files")),
          h(DataTable, { className: "compact-table", headers: ["Select", "#", "Selected Model Status", "Audio File", "Folder", "Model Coverage", "Last Result"], rows: files, emptyText: h("div", { className: "empty-state" }, h("strong", null, "No audio files to diarize yet."), "Upload audio in Media Library or convert a YouTube URL first; this tracker fills in once files land in audio_in/."), renderRow: (row, index) => h("tr", { key: row.name }, h("td", null, h("input", { type: "checkbox", name: "selected_audio", value: row.name, "data-check-group": "diarization-audio", "data-ready": row.selection_ready || "yes", "data-ready-by-model": row.selectionByModelJson || JSON.stringify(row.selectionByModel || {}) })), h("td", null, row.index || index + 1), h("td", null, h("span", { className: `queue-state ${row.state_class || "ready"}` }, row.queue_state || "ready"), h("p", { className: "row-note" }, row.targetModelLabel || row.targetBackendLabel || row.targetBackend || "selected model")), h("td", null, h("strong", { className: "file-name" }, row.fileName || row.name), h("p", { className: "row-note" }, row.name)), h("td", null, row.folder || "Unsorted Root"), h("td", null, h(ModelCoverage, { statuses: row.modelStatuses || [] })), h("td", null, h("span", { className: "detail-text" }, row.detail || "No previous run."), row.lastRun ? h("p", { className: "row-note" }, row.lastRun) : null)) }),
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
      return parsedRows.length ? parsedRows : [makeLabelEditorRow()];
    });
    const [transcriptText, setTranscriptText] = React.useState(row.transcriptText || "");
    // Default off unless the saved label explicitly kept dialogue.
    const [includeTranscript, setIncludeTranscript] = React.useState(() => {
      return row.includeTranscript === true;
    });
    const [issueQuestions, setIssueQuestions] = React.useState(row.issueQuestions || "");
    const [selectedModelKey, setSelectedModelKey] = React.useState("");
    const [autoSaveStatus, setAutoSaveStatus] = React.useState({ state: "idle", at: "" });
    const audioRef = React.useRef(null);
    const stopHandlerRef = React.useRef(null);
    const dragRowIdRef = React.useRef("");
    const formRef = React.useRef(null);
    const autoSaveAbortRef = React.useRef(null);
    const initialAutoSaveSkipRef = React.useRef(true);
    const activeTimeFieldRef = React.useRef(null);
    const serializedSegments = serializeLabelSegmentRows(segmentRows);
    const validSegmentCount = segmentRows.filter(labelEditorRowIsValid).length;
    const incompleteSegmentCount = segmentRows.filter((segment) => labelEditorRowHasAnyValue(segment) && !labelEditorRowIsValid(segment)).length;
    const completeDisabled = validSegmentCount === 0 || incompleteSegmentCount > 0;

    React.useEffect(() => {
      return () => {
        if (audioRef.current && stopHandlerRef.current) {
          audioRef.current.removeEventListener("timeupdate", stopHandlerRef.current);
        }
        // Don't leave a half-finished auto-save in flight when the dialog unmounts —
        // it would race against a real save submitted right after.
        if (autoSaveAbortRef.current) {
          autoSaveAbortRef.current.abort();
        }
      };
    }, []);

    // Debounced auto-save: any time labels, transcript, or questions change
    // we re-save as a draft. Skip the first render (we just opened the dialog
    // with existing data — no need to save it back). Don't auto-save when
    // there's an in-flight request — the abort controller handles cancellation.
    React.useEffect(() => {
      if (initialAutoSaveSkipRef.current) {
        initialAutoSaveSkipRef.current = false;
        return;
      }
      const handle = window.setTimeout(() => {
        runAutoSaveDraft();
      }, 1500);
      return () => window.clearTimeout(handle);
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [serializedSegments, transcriptText, issueQuestions, includeTranscript]);

    function runAutoSaveDraft() {
      if (!routes.saveTrainingLabel || !formRef.current) return;
      // Cancel any prior auto-save still in flight so we never save stale state.
      if (autoSaveAbortRef.current) {
        autoSaveAbortRef.current.abort();
      }
      const controller = new AbortController();
      autoSaveAbortRef.current = controller;
      const data = new FormData(formRef.current);
      data.set("label_action", "draft");
      // The "auto" flag lets the server skip the redirect chain and treat
      // this as a background draft instead of a user-driven submission.
      data.set("label_auto_save", "1");
      setAutoSaveStatus({ state: "saving", at: "" });
      fetch(routes.saveTrainingLabel, {
        method: "POST",
        body: data,
        credentials: "same-origin",
        signal: controller.signal,
        redirect: "manual",
      })
        .then((response) => {
          // redirect: "manual" turns 30x into an opaqueredirect with status 0.
          if (response.type === "opaqueredirect" || (response.status >= 200 && response.status < 400) || response.status === 0) {
            const stamp = new Date().toLocaleTimeString();
            setAutoSaveStatus({ state: "saved", at: stamp });
          } else {
            setAutoSaveStatus({ state: "error", at: "" });
          }
        })
        .catch((error) => {
          if (error?.name === "AbortError") return;
          setAutoSaveStatus({ state: "error", at: "" });
        });
    }

    function replaceSegmentRows(rawSegments) {
      const parsedRows = parseLabelSegmentRows(rawSegments);
      setSegmentRows(parsedRows.length ? parsedRows : [makeLabelEditorRow()]);
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
      if (stopHandlerRef.current) {
        audio.removeEventListener("timeupdate", stopHandlerRef.current);
        stopHandlerRef.current = null;
      }
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
      h("input", { type: "hidden", id: segmentsId, name: "label_segments", value: serializedSegments, readOnly: true, "data-label-segments-input": "true" }),
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
            h("audio", { ref: audioRef, controls: true, preload: "metadata", src: row.audioHref, "aria-label": `Audio preview for ${row.fileName || row.name}` })
          )
        : h(
            "div",
            { className: "label-audio empty-state" },
            h("p", { className: "label-audio-caption" }, "Audio sample"),
            h("h3", { className: "label-audio-title" }, "Audio file is not on disk."),
            h("p", { className: "row-note" }, "The WAV may have been moved or deleted. Re-upload it from the Media Library to label.")
          ),
      h(LabelSourcePicker, {
        row,
        selectedModelKey,
        onSelectedModelKeyChange: setSelectedModelKey,
        onUseSegments: replaceSegmentRows,
        onUseTranscript: (value) => setTranscriptText((current) => (current ? `${current}\n\n${value}` : value)),
        pickerId: `label-dialog-${suffix}`,
      }),
      h(
        "div",
        { className: "inline" },
        h(Field, { id: backendId, label: "Training backend" }, h(SelectInput, { id: backendId, name: "label_backend", defaultValue: row.backend || "both" }, h(Option, { value: "both" }, "NeMo + pyannote"), h(Option, { value: "nemo" }, "NeMo only"), h(Option, { value: "pyannote" }, "pyannote only"))),
        h(Field, { id: projectId, label: "Fine-tuning project" }, h("input", { id: projectId, name: "label_project_name", list: "training_label_project_names", defaultValue: row.projectName || ctx.trainingLabels?.defaultProjectName || "uploaded-site-training", placeholder: "uploaded-site-training", required: true }))
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
            h("button", { className: "secondary", type: "button", onClick: fillActiveTimeField, disabled: !row.audioHref }, "Use Current Seconds")
          )
        ),
        h(
          "div",
          { className: "table-scroll label-editor-table-wrap" },
          h(
            "table",
            { className: "label-editor-table" },
            h("thead", null, h("tr", null, h("th", null, "#"), h("th", null, "Start"), h("th", null, "End"), h("th", null, "Speaker"), h("th", null, "Actions"))),
            h(
              "tbody",
              null,
              segmentRows.map((segment, index) =>
                h(
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
                )
              )
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
          ? h(Field, { id: transcriptId, label: "Transcript export", note: "Saved in the project's text/ folder. Marked as not aligned with speech-time labels." }, h("textarea", { id: transcriptId, name: "label_transcript_text", value: transcriptText, onChange: (event) => setTranscriptText(event.target.value), placeholder: "Transcript text or annotation notes" }))
          : h("p", { className: "row-note" }, "Dialogue is excluded from this sample. Toggle on to save it for your records (training is unaffected either way)."),
        h(Field, { id: questionsId, label: "Questions or issues", note: "Anything here keeps the item out of completed training until it is answered." }, h("textarea", { id: questionsId, name: "label_issue_questions", value: issueQuestions, onChange: (event) => setIssueQuestions(event.target.value), placeholder: "What needs to be clarified before this can be used for training?" }))
      ),
      h(
        "div",
        { className: "button-row" },
        h("button", { className: "secondary", type: "submit", name: "label_action", value: "draft" }, "Save For Later"),
        h("button", { type: "submit", name: "label_action", value: "complete", disabled: completeDisabled }, "Complete Label"),
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

  function TrainingLabelsTable({ rows }) {
    return h(DataTable, {
      className: "training-label-table",
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
            null,
            h("strong", { className: "file-name" }, row.fileName || row.name),
            h("p", { className: "row-note" }, row.folder || "Unsorted Root"),
            h("p", { className: "row-note" }, row.name)
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
              : h("span", { className: "row-note" }, "Run diarization to create a review page")
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
    React.useEffect(() => {
      uiState.showCompletedTrainingLabels = showCompleted;
    }, [showCompleted]);
    const visibleRows = showCompleted ? rows : rows.filter((row) => row.status !== "completed");
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
    return h(
      "div",
      { className: "workspace-file-picker" },
      rows.length
        ? rows.map((file) =>
            h(
              "label",
              { key: file.path, className: "checkbox-row file-choice" },
              h("input", { type: "checkbox", name, value: file.path, "data-check-group": group }),
              h("span", null, h("strong", null, file.name || file.path), h("small", null, file.path || ""))
            )
          )
        : h("p", { className: "field-status" }, emptyText)
    );
  }

  function FineTuneUploadCard({ preferences, projectNames, trainingSources }) {
    const sources = trainingSources || {};
    const audioFiles = sources.audioFiles || [];
    const rttmFiles = sources.rttmFiles || [];
    const transcriptFiles = sources.transcriptFiles || [];
    const labelHref = navItems.find((item) => item.path === "/training-labels")?.href || "/training-labels";
    return h(
      "section",
      { className: "subpanel", id: "fine-tune-upload" },
      h("div", { className: "panel-head" }, h("div", null, h("h2", null, "1. Add Training Samples"), h("p", null, "Select SSH workspace audio and matching RTTM labels. Multiple audio files can be added in one submit."))),
      h(
        "form",
        { method: "post", action: routes.fineTuneUpload, "data-loading-message": "Adding training sample batch..." },
        h(Field, { id: "fine_tuning_backend", label: "Backend" }, h(SelectInput, { id: "fine_tuning_backend", defaultValue: preferences.fine_tuning_backend || "pyannote" }, h(Option, { value: "nemo" }, "NeMo"), h(Option, { value: "pyannote" }, "pyannote"))),
        h(Field, { id: "project_name", label: "Project name" }, h("input", { id: "project_name", name: "project_name", list: "fine_tuning_project_names", placeholder: "callhome-msdd" })),
        projectNames.length ? h("datalist", { id: "fine_tuning_project_names" }, projectNames.map((name) => h("option", { key: name, value: name }))) : null,
        h("p", { className: "footer-note" }, "Use an existing project name to append samples."),
        h("div", { className: "selection-toolbar" }, h("button", { className: "secondary", type: "button", "data-select-group": "fine-tune-audio", "data-select-mode": "all" }, "Select all"), h("button", { className: "ghost", type: "button", "data-select-group": "fine-tune-audio", "data-select-mode": "none" }, "Clear")),
        h(Field, { id: "server_audio_paths", label: "SSH audio in audio_in/" }, h(WorkspaceFileChecklist, { group: "fine-tune-audio", name: "server_audio_paths", files: audioFiles, emptyText: "No audio files are present in audio_in/ yet." })),
        h("p", { className: "field-status" }, h("span", { "data-selection-count": "fine-tune-audio" }, "0"), " audio file(s) selected."),
        h("div", { className: "selection-toolbar" }, h("button", { className: "secondary", type: "button", "data-select-group": "fine-tune-rttm", "data-select-mode": "all" }, "Select all"), h("button", { className: "ghost", type: "button", "data-select-group": "fine-tune-rttm", "data-select-mode": "none" }, "Clear")),
        h(Field, { id: "server_rttm_paths", label: "SSH RTTM label files" }, h(WorkspaceFileChecklist, { group: "fine-tune-rttm", name: "server_rttm_paths", files: rttmFiles, emptyText: h(React.Fragment, null, "No RTTM files found yet. Use ", h("a", { href: labelHref }, "Training Labels"), " to create labels from any audio_in/ file.") })),
        h("p", { className: "field-status" }, h("span", { "data-selection-count": "fine-tune-rttm" }, "0"), " RTTM file(s) selected."),
        h("details", { className: "details-box compact-details" }, h("summary", null, "Optional Transcripts"), h("div", null, h("div", { className: "selection-toolbar" }, h("button", { className: "secondary", type: "button", "data-select-group": "fine-tune-transcript", "data-select-mode": "all" }, "Select all"), h("button", { className: "ghost", type: "button", "data-select-group": "fine-tune-transcript", "data-select-mode": "none" }, "Clear")), h(Field, { id: "server_transcript_paths", label: "SSH transcript files" }, h(WorkspaceFileChecklist, { group: "fine-tune-transcript", name: "server_transcript_paths", files: transcriptFiles, emptyText: "No optional transcript files found." })), h("p", { className: "field-status" }, h("span", { "data-selection-count": "fine-tune-transcript" }, "0"), " transcript file(s) selected."))),
        h(Field, { id: "training_transcript_text", label: "Optional shared transcript text" }, h("textarea", { id: "training_transcript_text", name: "training_transcript_text", placeholder: "Transcript text or annotation notes" })),
        h("p", { className: "footer-note" }, "Batch pairing uses filename stems: ", h("code", null, "001_clip.wav"), " pairs with ", h("code", null, "001_clip.rttm"), " and optional ", h("code", null, "001_clip.txt"), "."),
        h("p", null, h("button", { type: "submit" }, "Add samples"))
      )
    );
  }

  function FineTunePrepareCard({ preferences, sampleProjects }) {
    const nemo = preferences.nemo_fine_tuning || {};
    const pyannote = preferences.pyannote_fine_tuning || {};
    const disabled = sampleProjects.length === 0;
    return h(
      "section",
      { className: "subpanel" },
      h("div", { className: "panel-head" }, h("div", null, h("h2", null, "2. Prepare Artifacts"), h("p", null, "Generate backend-specific training files."))),
      h(
        "form",
        { method: "post", action: routes.fineTunePrepare },
        h(Field, { id: "prepare_project_choice", label: "Project with uploaded samples" }, h("select", { id: "prepare_project_choice", "data-project-name-target": "prepare_project_name", "data-project-backend-target": "prepare_backend" }, h(FineTuneProjectOptions, { projects: sampleProjects }))),
        h(Field, { id: "prepare_backend", label: "Backend" }, h(SelectInput, { id: "prepare_backend", defaultValue: preferences.fine_tuning_backend || "pyannote" }, h(Option, { value: "nemo" }, "NeMo"), h(Option, { value: "pyannote" }, "pyannote"))),
        h(Field, { id: "prepare_project_name", label: "Project name" }, h(TextInput, { id: "prepare_project_name", placeholder: "callhome-msdd" })),
        h(
          "details",
          { className: "details-box compact-details" },
          h("summary", null, "Advanced Settings"),
          h("div", null,
            h("div", { className: "inline" }, h(Field, { id: "train_ratio", label: "Train ratio" }, h(TextInput, { id: "train_ratio", defaultValue: nemo.train_ratio || defaults.trainRatio })), h(Field, { id: "devices", label: "Devices" }, h(TextInput, { id: "devices", defaultValue: nemo.devices || "" }))),
            h("div", { className: "inline-3" }, h(Field, { id: "base_window", label: "NeMo base window" }, h(TextInput, { id: "base_window", defaultValue: nemo.base_window || defaults.baseWindow })), h(Field, { id: "base_shift", label: "NeMo base shift" }, h(TextInput, { id: "base_shift", defaultValue: nemo.base_shift || defaults.baseShift })), h(Field, { id: "step_count", label: "NeMo step count" }, h(TextInput, { id: "step_count", defaultValue: nemo.step_count || defaults.stepCount }))),
            h(Field, { id: "config_name", label: "NeMo config name" }, h(TextInput, { id: "config_name", defaultValue: nemo.config_name || defaults.configName })),
            h(Field, { id: "speaker_model", label: "NeMo speaker model" }, h(TextInput, { id: "speaker_model", defaultValue: nemo.speaker_model || defaults.speakerModel })),
            h(Field, { id: "pyannote_pretrained_model", label: "pyannote pretrained model" }, h(TextInput, { id: "pyannote_pretrained_model", defaultValue: pyannote.pretrained_model || "" })),
            h("div", { className: "inline-3" }, h(Field, { id: "pyannote_duration", label: "pyannote chunk duration" }, h(TextInput, { id: "pyannote_duration", defaultValue: pyannote.duration || "" })), h(Field, { id: "pyannote_max_speakers_per_chunk", label: "Max speakers per chunk" }, h(TextInput, { id: "pyannote_max_speakers_per_chunk", defaultValue: pyannote.max_speakers_per_chunk || "" })), h(Field, { id: "pyannote_max_speakers_per_frame", label: "Max speakers per frame" }, h(TextInput, { id: "pyannote_max_speakers_per_frame", defaultValue: pyannote.max_speakers_per_frame || "" }))),
            h("div", { className: "inline" }, h(Field, { id: "max_epochs", label: "Max epochs" }, h(TextInput, { id: "max_epochs", defaultValue: nemo.max_epochs || defaults.maxEpochs })), h(Field, { id: "nemo_root", label: "Optional NeMo root" }, h(TextInput, { id: "nemo_root", placeholder: "path/to/NeMo" }))),
            h("div", { className: "inline-3" }, h(Field, { id: "slurm_partition", label: "Slurm partition" }, h(TextInput, { id: "slurm_partition", defaultValue: nemo.slurm_partition || defaults.slurmPartition })), h(Field, { id: "slurm_time", label: "Slurm time" }, h(TextInput, { id: "slurm_time", defaultValue: nemo.slurm_time || defaults.slurmTime })), h(Field, { id: "slurm_memory", label: "Slurm memory" }, h(TextInput, { id: "slurm_memory", defaultValue: nemo.slurm_memory || defaults.slurmMemory }))),
            h("div", { className: "inline" }, h(Field, { id: "slurm_cpus", label: "Slurm CPUs" }, h(TextInput, { id: "slurm_cpus", defaultValue: nemo.slurm_cpus || defaults.slurmCpus })), h(Field, { id: "slurm_gpus", label: "Slurm GPUs" }, h(TextInput, { id: "slurm_gpus", defaultValue: nemo.slurm_gpus || defaults.slurmGpus })))
          )
        ),
        h("p", null, h("button", { className: "secondary", type: "submit", disabled }, "Prepare Training Artifacts"))
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
      h("p", { className: "progress-meta" }, `${project.completedSteps} of 3 stages complete: upload, prepare, launch.`),
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
                project.recentRuns.map((run) =>
                  h(
                    "li",
                    { key: run.runName },
                    h("strong", null, run.displayName || run.versionName),
                    `: ${run.status}`,
                    run.displayName && run.versionName && run.displayName !== run.versionName
                      ? h("span", { className: "row-note" }, ` (v: ${run.versionName})`)
                      : null,
                    run.runDir
                      ? h(
                          "button",
                          {
                            className: "ghost",
                            type: "button",
                            style: { marginLeft: "8px" },
                            onClick: () => promptRenameRun(run),
                            title: "Give this training run a friendlier name. Useful when you've got several versions and need to remember which one was the best.",
                          },
                          "Rename"
                        )
                      : null
                  )
                )
              )
            : h("p", { className: "row-note" }, "No training runs yet."),
          project.warnings?.length ? h("ul", null, project.warnings.map((warning, index) => h("li", { key: index }, warning))) : h("p", { className: "row-note" }, "No preparation warnings.")
        )
      ),
      h(
        "div",
        { className: "project-actions" },
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
      ),
      h(ClusterQueuePanel, { title: "Cluster Queue" })
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
        h("div", { className: "three-grid" }, h(FineTuneUploadCard, { preferences, projectNames, trainingSources: ctx.trainingSources || {} }), h(FineTunePrepareCard, { preferences, sampleProjects }), h(FineTuneLaunchCard, { preferences, preparedProjects }))
      ),
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
      ),
      h(ClusterQueuePanel, { title: "Cluster Queue" })
    );
  }

  function Page() {
    switch (state.currentPath) {
      case "/uploads":
        return h(UploadsPage);
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

  function formIdentityForControl(node) {
    const form = node.form || node.closest("form");
    return text(form?.getAttribute("data-form-key") || form?.id || form?.getAttribute("action") || "", "").trim();
  }

  function formControlKey(node) {
    const formKey = formIdentityForControl(node);
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

  function snapshotFormState() {
    const snapshot = {};
    document.querySelectorAll("input, select, textarea").forEach((node) => {
      if (!(node instanceof HTMLInputElement || node instanceof HTMLSelectElement || node instanceof HTMLTextAreaElement)) {
        return;
      }
      if (node instanceof HTMLInputElement && node.type === "file") {
        return;
      }
      const key = formControlKey(node);
      if (!key) {
        return;
      }
      if (node instanceof HTMLInputElement && (node.type === "checkbox" || node.type === "radio")) {
        snapshot[key] = { checked: node.checked };
        return;
      }
      if (node instanceof HTMLSelectElement && node.multiple) {
        snapshot[key] = {
          values: Array.from(node.selectedOptions || []).map((option) => option.value),
        };
        return;
      }
      snapshot[key] = { value: node.value };
    });
    return snapshot;
  }

  function restoreFormState(snapshot) {
    if (!snapshot) {
      return;
    }
    document.querySelectorAll("input, select, textarea").forEach((node) => {
      if (!(node instanceof HTMLInputElement || node instanceof HTMLSelectElement || node instanceof HTMLTextAreaElement)) {
        return;
      }
      if (node instanceof HTMLInputElement && node.type === "file") {
        return;
      }
      const saved = snapshot[formControlKey(node)];
      if (!saved) {
        return;
      }
      if (node instanceof HTMLInputElement && (node.type === "checkbox" || node.type === "radio")) {
        node.checked = Boolean(saved.checked);
        return;
      }
      if (node instanceof HTMLSelectElement && node.multiple) {
        const selectedValues = new Set(saved.values || []);
        Array.from(node.options || []).forEach((option) => {
          option.selected = selectedValues.has(option.value);
        });
        return;
      }
      if (Object.prototype.hasOwnProperty.call(saved, "value")) {
        if (node instanceof HTMLSelectElement) {
          const hasOption = Array.from(node.options || []).some((option) => option.value === saved.value);
          if (!hasOption) {
            return;
          }
        }
        node.value = saved.value;
      }
    });
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
    }
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

  function postRenderSync() {
    document.querySelectorAll("input[type='file'][data-file-summary]").forEach((input) => {
      if (input instanceof HTMLInputElement) {
        updateFileSummary(input);
      }
    });
    document.querySelectorAll("[data-selection-count]").forEach((node) => {
      const group = node.getAttribute("data-selection-count");
      if (group) {
        updateSelectionCounter(group, node.closest("form") || document);
      }
    });
    document.querySelectorAll("select[data-project-name-target]").forEach((select) => {
      if (select instanceof HTMLSelectElement) {
        syncProjectSelector(select);
      }
    });
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
          if (target.type === "file") {
            updateFileSummary(target);
          }
        }
        if (target instanceof HTMLSelectElement && target.hasAttribute("data-project-name-target")) {
          syncProjectSelector(target);
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
