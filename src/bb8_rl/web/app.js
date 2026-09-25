/* BB8-RL control room. No dependencies; the server remains command authority. */
(function () {
  "use strict";

  const finitePoint = (point) => Array.isArray(point) && point.length === 2 && point.every(Number.isFinite);
  const coordinates = (point) => finitePoint(point) ? `${point[0].toFixed(2)}, ${point[1].toFixed(2)} m` : "—";

  function localizationView(state = {}) {
    const explicit = Object.prototype.hasOwnProperty.call(state, "localization_status")
      || Object.prototype.hasOwnProperty.call(state, "localization_valid");
    const status = explicit ? String(state.localization_status || "uninitialized") : "legacy";
    const validRadius = Number.isFinite(state.position_radius) && state.position_radius >= 0;
    const usable = explicit ? state.localization_valid === true && ["measured", "predicted"].includes(status)
      && finitePoint(state.pose) && validRadius : finitePoint(state.pose);
    const currentPose = usable ? [...state.pose] : null;
    const currentRadius = usable && validRadius ? state.position_radius : null;
    const ghostPose = explicit && !usable && finitePoint(state.last_seen_pose) ? [...state.last_seen_pose] : null;
    const age = Number.isFinite(state.last_seen_age_s) && state.last_seen_age_s >= 0 ? state.last_seen_age_s : null;
    const ageText = age === null ? "time unavailable" : `${age.toFixed(1)} s ago (sim)`;
    const requiresNewGoal = explicit && state.requires_new_goal === true;
    const label = status === "lost" ? "Localization lost" : status === "reacquiring" ? "Reacquiring position"
      : usable ? status === "predicted" ? "Predicted position" : "Position available" : "Waiting for position";
    const message = status === "lost" ? "The current position is unavailable. The last-seen marker is historical. Wait for visual recovery, or reset the episode."
      : status === "reacquiring" ? "Checking returning camera observations. Navigation stays disabled until position recovery is complete."
        : requiresNewGoal && usable ? "Position recovered. BB-8 is stopped; choose a new destination to move again."
          : explicit && !usable ? "Waiting for a usable camera position. Reset or Demo can start a new episode." : null;
    return {
      explicit, status, usable, currentPose, currentRadius, ghostPose, requiresNewGoal, label, message,
      positionText: currentPose ? coordinates(currentPose) : explicit ? "Unavailable" : "Waiting for vision",
      radiusText: currentRadius === null ? explicit ? "Unavailable" : "—" : `${(currentRadius * 100).toFixed(1)} cm`,
      lastSeenText: ghostPose ? `${coordinates(ghostPose)} · ${ageText}` : null,
      ghostLabel: ghostPose ? `Last seen · ${ageText}` : null,
      showRoute: !explicit || usable && !requiresNewGoal,
    };
  }

  function controlAvailability(state = {}, { connected = false, token = null, mapReady = false, pendingCommands = 0 } = {}) {
    const sessionReady = connected && !!token;
    const recoveryReady = sessionReady && state.asset_ready === true && pendingCommands === 0;
    const phaseReady = !/error|failed|loading|starting|initializ|resetting|stopping|closed/.test(String(state.phase || "").toLowerCase());
    const localization = localizationView(state);
    return {
      navigate: recoveryReady && mapReady && phaseReady && (!localization.explicit || localization.usable),
      stop: sessionReady,
      reset: recoveryReady,
      mode: recoveryReady,
      // The supervisor resets the episode before starting this prepared route.
      demo: recoveryReady && phaseReady,
    };
  }

  function agencyView(state = {}, session = {}) {
    const agency = state.agency && typeof state.agency === "object" && !Array.isArray(state.agency) ? state.agency : null;
    const cleanText = (value, fallback = "", limit = 260) => typeof value === "string" && value.trim() ? value.trim().slice(0, limit) : fallback;
    const available = agency?.available === true;
    const purpose = agency?.selection_policy === "learned_station_outcomes";
    const resource = purpose && Number.isFinite(agency.resource) && agency.resource >= 0 && agency.resource <= 1 ? agency.resource : null;
    const enabled = agency?.enabled === true;
    const connected = session.connected === true;
    const localization = localizationView(state);
    const hasPosition = !localization.explicit || localization.usable;
    const canToggle = connected && !!session.token && (session.pendingCommands || 0) === 0
      && (enabled || available && controlAvailability(state, session).navigate);
    const episodes = Number.isSafeInteger(agency?.episodes) && agency.episodes >= 0 ? agency.episodes : 0;
    const recentExperience = cleanText(agency?.recent_experience?.summary) || null;
    const intention = connected && enabled && hasPosition && agency?.intention && finitePoint(agency.intention.goal) ? {
      label: cleanText(agency.intention.label, "Visit a selected place", 100),
      explanation: [cleanText(agency.intention.question), cleanText(agency.intention.explanation)].filter(Boolean).join(" "),
    } : null;
    const preferences = (Array.isArray(agency?.preferences) ? agency.preferences : [])
      .filter((item) => item && Number.isFinite(item.value) && Number.isSafeInteger(item.visits) && item.visits > 0)
      .slice().sort((left, right) => right.value - left.value).slice(0, 5)
      .map((item) => ({ label: cleanText(item.label, "Remembered place", 100), visits: item.visits,
        ...(purpose && Number.isFinite(item.response_probability) && item.response_probability >= 0 && item.response_probability <= 1 ? {response: `${Math.round(item.response_probability * 100)}% predicted response`} : {}),
      }));
    const status = !connected ? "Waiting for connection" : !agency ? "Unavailable in this session"
      : !hasPosition ? localization.label : agency.status === "exhausted" ? "No new reachable target"
        : purpose ? (enabled ? ({choosing: "Comparing possible outcomes", travelling: "Approaching a station", interacting: "Testing a station response", remembering: "Remembering the outcome", satisfied: "Need satisfied · waiting", idle: "No useful action · waiting", waiting: "Waiting for an observation"}[agency.status] || "Learning") : available ? "Learning paused" : "Learning unavailable")
          : enabled ? (agency.status === "checking" ? "Checking routes" : "Exploring") : available ? "Exploration paused" : "Exploration unavailable";
    const message = !connected ? "Exploration controls will return when the local app reconnects."
      : !agency ? "This session does not provide autonomous exploration. You can still choose a destination."
        : !hasPosition ? localization.message
          : cleanText(agency.message, enabled ? "Selecting map targets without a completed exploration visit." : available ? "Start exploring to let BB-8 choose its next destination." : "Exploration is not ready in this session.");
    return { available, enabled, canToggle, status, message, intention, preferences, recentExperience, purpose, resource,
      buttonText: purpose ? enabled ? "Pause learning" : "Start learning" : enabled ? "Pause exploration" : "Start exploring",
      experienceText: `${episodes} ${episodes === 1 ? "experience" : "experiences"} remembered`,
    };
  }

  function plotRect(bounds, width, height, padding = 0) {
    if (!Array.isArray(bounds) || bounds.length !== 4 || !bounds.every(Number.isFinite)
      || bounds[2] <= bounds[0] || bounds[3] <= bounds[1]
      || ![width, height, padding].every(Number.isFinite) || padding < 0
      || width <= 2 * padding || height <= 2 * padding) return null;
    const scale = Math.min((width - 2 * padding) / (bounds[2] - bounds[0]), (height - 2 * padding) / (bounds[3] - bounds[1]));
    const w = (bounds[2] - bounds[0]) * scale;
    const h = (bounds[3] - bounds[1]) * scale;
    return { x: (width - w) / 2, y: (height - h) / 2, width: w, height: h, scale };
  }

  function worldToCanvas(point, bounds, width, height, padding = 0) {
    const rect = plotRect(bounds, width, height, padding);
    if (!rect || !finitePoint(point)) return null;
    return [rect.x + (point[0] - bounds[0]) * rect.scale, rect.y + (bounds[3] - point[1]) * rect.scale];
  }

  function canvasToWorld(point, bounds, width, height, padding = 0) {
    const rect = plotRect(bounds, width, height, padding);
    if (!rect || !finitePoint(point) || point[0] < rect.x || point[0] > rect.x + rect.width
      || point[1] < rect.y || point[1] > rect.y + rect.height) return null;
    return [bounds[0] + (point[0] - rect.x) / rect.scale, bounds[3] - (point[1] - rect.y) / rect.scale];
  }

  function worldToCell(point, map) {
    if (!finitePoint(point) || !map || !Array.isArray(map.bounds)) return null;
    const [xmin, ymin, xmax, ymax] = map.bounds;
    if (point[0] < xmin || point[0] >= xmax || point[1] < ymin || point[1] >= ymax) return null;
    const x = Math.floor((point[0] - xmin) / (xmax - xmin) * map.width);
    const y = Math.floor((point[1] - ymin) / (ymax - ymin) * map.height);
    return { x, y };
  }

  function cellAtWorld(point, map) {
    const cell = worldToCell(point, map);
    return cell && Array.isArray(map.cells) && Array.isArray(map.cells[cell.y]) ? map.cells[cell.y][cell.x] : null;
  }

  function validMap(map) {
    return !!map && !!plotRect(map.bounds, 100, 100) && Number.isInteger(map.width) && Number.isInteger(map.height)
      && map.width > 0 && map.height > 0 && map.width <= 2048 && map.height <= 2048
      && Number.isFinite(map.resolution) && map.resolution > 0
      && Array.isArray(map.cells) && map.cells.length === map.height
      && map.cells.every((row) => Array.isArray(row) && row.length === map.width && row.every((value) => value === 0 || value === 1 || value === 2));
  }

  const helpers = Object.freeze({ plotRect, worldToCanvas, canvasToWorld, worldToCell, cellAtWorld, validMap, localizationView, controlAvailability, agencyView });
  if (typeof module !== "undefined" && module.exports) module.exports = helpers;
  if (typeof window === "undefined") return;
  window.BB8Map = helpers;
  if (typeof document === "undefined") return;

  const $ = (id) => document.getElementById(id);
  const text = (id, value) => { if ($(id).textContent !== String(value)) $(id).textContent = String(value); };
  const humanize = (value) => String(value || "Waiting").replace(/[_-]+/g, " ").replace(/^\w/, (letter) => letter.toUpperCase()).slice(0, 180);
  const ids = ["A", "B", "C"];
  const canvas = $("map-canvas");
  const context = canvas.getContext("2d");
  const padding = 34;
  const modeButtons = Array.from(document.querySelectorAll("[data-mode]")).filter((element) => element.tagName === "BUTTON");
  let state = null;
  let map = null;
  let mapImage = null;
  let connected = false;
  let token = null;
  let polling = false;
  let mapLoading = false;
  let nextMapAttempt = 0;
  let selectedGoal = null;
  let hoverPoint = null;
  let keyboardPoint = null;
  let canvasFocused = false;
  let pendingCommands = 0;
  let commandGeneration = 0;
  let frameEpoch = 0;
  let toastUntil = 0;
  let drawRequested = false;
  let heartbeatBusy = false;
  let preferenceSignature = null;
  const cameras = new Map(ids.map((id) => [id, {
    card: document.querySelector(`[data-camera="${id}"]`),
    image: document.querySelector(`[data-camera-image="${id}"]`),
    placeholder: document.querySelector(`[data-camera-placeholder="${id}"]`),
    status: document.querySelector(`[data-camera-status="${id}"]`),
    source: document.querySelector(`[data-camera-source="${id}"]`),
    frame: document.querySelector(`[data-camera-frame="${id}"]`),
    loading: false, lastKey: null, next: null, hasFrame: false,
  }]));

  function activity(title, message, tone = "neutral", temporary = false) {
    text("activity-title", title);
    text("activity-message", message);
    $("activity-title").closest(".activity-bar").dataset.tone = tone;
    text("activity-icon", tone === "error" || tone === "warning" ? "!" : "i");
    if (temporary) toastUntil = Date.now() + 5000;
  }

  function canNavigate() {
    return availableControls().navigate;
  }

  function availableControls() {
    return controlAvailability(state || {}, { connected, token, mapReady: !!map, pendingCommands });
  }

  function renderAgency() {
    const agency = agencyView(state || {}, { connected, token, mapReady: !!map, pendingCommands });
    const button = $("autonomy-button");
    button.disabled = !agency.canToggle;
    button.setAttribute("aria-pressed", String(agency.enabled));
    text("autonomy-button", agency.buttonText);
    $("agency-panel").dataset.enabled = String(connected && agency.enabled);
    text("agency-status", agency.status);
    text("agency-message", agency.message);
    text("agency-caption", agency.purpose ? "Experimental · learned interaction outcomes" : "Diagnostic · map target coverage");
    text("agency-title", agency.purpose ? "A reason to move" : "Autonomous exploration");
    text("agency-description", agency.purpose ? "Learn which station restores a depleted resource, then wait when the need is satisfied." : "Visit reachable targets not previously completed, then pause.");
    text("agency-preference-title", agency.purpose ? "Learned station responses" : "Places with experience");
    text("agency-help", agency.purpose ? "Stop or a manual destination pauses learning. Reset starts a new resource episode and keeps learned outcomes. This is an engineered motivation experiment, not a personality claim." : "Stop or a manual destination pauses exploration. Completed targets stay remembered after Reset.");
    $("purpose-resource").hidden = !agency.purpose;
    text("resource-value", agency.resource === null ? "Awaiting telemetry" : `${Math.round(agency.resource * 100)}%`);
    $("resource-meter").value = agency.resource ?? 0;
    $("agency-intention").hidden = !agency.intention;
    text("agency-intention-label", agency.intention?.label || "—");
    text("agency-intention-explanation", agency.intention?.explanation || "");
    $("agency-recent").hidden = !agency.recentExperience;
    text("agency-recent-summary", agency.recentExperience || "");
    text("agency-experiences", agency.experienceText);
    $("agency-preferences").hidden = agency.preferences.length === 0;
    const signature = JSON.stringify(agency.preferences);
    if (signature !== preferenceSignature) {
      preferenceSignature = signature;
      $("agency-preference-list").replaceChildren(...agency.preferences.map((preference) => {
        const item = document.createElement("li");
        const label = document.createElement("span");
        const visits = document.createElement("span");
        label.textContent = preference.label;
        visits.textContent = `${preference.visits} ${preference.visits === 1 ? "outcome" : "outcomes"}${preference.response ? ` · ${preference.response}` : ""}`;
        item.append(label, visits);
        return item;
      }));
    }
  }

  function updateControls() {
    const controls = availableControls();
    ["goal-x", "goal-y", "goal-button"].forEach((id) => { $(id).disabled = !controls.navigate; });
    $("demo-button").disabled = !controls.demo;
    $("stop-button").disabled = !controls.stop;
    $("stop-button").title = connected ? "Cancel motion and the current route (Escape)" : "Stop is unavailable until the local app reconnects";
    $("reset-button").disabled = !controls.reset;
    modeButtons.forEach((button) => { button.disabled = !controls.mode; });
    $("map-stage").dataset.enabled = String(controls.navigate);
    canvas.setAttribute("aria-disabled", String(!controls.navigate));
    renderAgency();
  }

  function setConnection(value) {
    connected = value;
    document.body.dataset.connection = value ? "connected" : "disconnected";
    $("connection-status").dataset.state = value ? "connected" : "disconnected";
    text("connection-text", value ? "Local app connected" : "Disconnected");
    $("connection-notice").hidden = value;
    if (!value) {
      text("phase-text", "Disconnected");
      $("phase-chip").dataset.tone = "error";
      text("controller-value", "Connection lost");
    }
    for (const camera of cameras.values()) {
      camera.card.querySelector(".feed-label").textContent = camera.hasFrame ? (value ? "RGB · LIVE" : "RGB · LAST FRAME") : "RGB · WAITING";
    }
    updateControls();
  }

  async function requestJSON(url, options = {}, timeout = 2500) {
    const abort = new AbortController();
    const timer = window.setTimeout(() => abort.abort(), timeout);
    try {
      const response = await fetch(url, { cache: "no-store", credentials: "same-origin", mode: "same-origin", ...options, signal: abort.signal });
      let payload;
      try { payload = await response.json(); } catch (_) { throw new Error(`The local app returned an unreadable response (${response.status}).`); }
      if (!response.ok || payload?.ok === false || payload?.error) throw new Error(String(payload?.error || payload?.message || `Request failed (${response.status}).`));
      return payload;
    } catch (error) {
      if (error.name === "AbortError") throw new Error("The local app did not respond in time.");
      throw error;
    } finally { window.clearTimeout(timer); }
  }

  function buildMapImage() {
    mapImage = document.createElement("canvas");
    mapImage.width = map.width;
    mapImage.height = map.height;
    const imageContext = mapImage.getContext("2d");
    const image = imageContext.createImageData(map.width, map.height);
    for (let y = 0; y < map.height; y += 1) {
      for (let x = 0; x < map.width; x += 1) {
        const value = map.cells[y][x];
        const offset = ((map.height - 1 - y) * map.width + x) * 4;
        const color = value === 1 ? [247, 249, 242] : [58, 78, 88];
        image.data.set([...color, value === 0 ? 0 : 255], offset);
      }
    }
    imageContext.putImageData(image, 0, 0);
  }

  async function loadMap() {
    if (mapLoading || map || Date.now() < nextMapAttempt) return;
    mapLoading = true;
    try {
      const candidate = await requestJSON("/api/map");
      if (!validMap(candidate)) throw new Error("The saved map has an unsupported or incomplete format.");
      map = candidate;
      buildMapImage();
      $("map-placeholder").hidden = true;
      text("map-subtitle", `${(map.bounds[2] - map.bounds[0]).toFixed(1)} × ${(map.bounds[3] - map.bounds[1]).toFixed(1)} m · ${(map.resolution * 100).toFixed(0)} cm cells · saved RGB scan`);
      $("goal-x").min = String(map.bounds[0]);
      $("goal-x").max = String(map.bounds[2]);
      $("goal-y").min = String(map.bounds[1]);
      $("goal-y").max = String(map.bounds[3]);
      updateControls();
      scheduleDraw();
    } catch (error) {
      nextMapAttempt = Date.now() + 3000;
      text("map-placeholder-title", "Room memory unavailable");
      text("map-placeholder-description", state?.asset_ready === false ? "Load the demo assets to view the saved scan." : error.message);
    } finally { mapLoading = false; }
  }

  function updateFrame(id, sequence) {
    const camera = cameras.get(id);
    if (sequence === null || sequence === undefined) return;
    const key = `${frameEpoch}:${sequence}`;
    if (key === camera.lastKey || key === camera.next?.key) return;
    const next = { key, sequence, url: `/api/frame/${encodeURIComponent(id)}?seq=${encodeURIComponent(sequence)}&epoch=${frameEpoch}` };
    if (camera.loading) { camera.next = next; return; }
    startFrame(camera, next);
  }

  function startFrame(camera, next) {
    camera.loading = true;
    camera.lastKey = next.key;
    camera.next = null;
    const finish = (success) => {
      camera.loading = false;
      if (success) {
        camera.hasFrame = true;
        camera.image.hidden = false;
        camera.placeholder.hidden = true;
        camera.frame.textContent = `FRAME ${next.sequence}`;
      } else {
        camera.hasFrame = false;
        camera.lastKey = null;
        camera.image.hidden = true;
        camera.placeholder.hidden = false;
        camera.placeholder.querySelector("span").textContent = "Camera frame unavailable";
        camera.frame.textContent = "Waiting for next frame";
      }
      camera.card.querySelector(".feed-label").textContent = camera.hasFrame ? (connected ? "RGB · LIVE" : "RGB · LAST FRAME") : "RGB · WAITING";
      if (camera.next && !camera.card.hidden && connected) startFrame(camera, camera.next);
    };
    camera.image.onload = () => finish(true);
    camera.image.onerror = () => finish(false);
    camera.image.src = next.url;
  }

  function renderState(previous) {
    const localization = localizationView(state);
    const localizationWarning = localization.explicit && !localization.usable;
    const recovered = localization.usable && localization.requiresNewGoal;
    if (localization.explicit && (!localization.usable || recovered)) {
      selectedGoal = null;
      text("map-selection", recovered ? "Choose a new destination" : "Navigation unavailable");
    } else if (!selectedGoal) {
      text("map-selection", finitePoint(state.goal) ? `Requested: ${coordinates(state.goal)}` : "No target selected");
    }
    const mode = [1, 2, 3].includes(state.mode) ? state.mode : 1;
    if (previous && (previous.mode !== state.mode || (Number.isFinite(state.frame_seq) && state.frame_seq < previous.frame_seq))) frameEpoch += 1;
    modeButtons.forEach((button) => button.setAttribute("aria-pressed", String(Number(button.dataset.mode) === mode)));
    $("camera-feeds").dataset.mode = String(mode);
    text("camera-count", `${mode} ${mode === 1 ? "VIEW" : "VIEWS"}`);
    const sources = Array.isArray(state.source_ids) ? state.source_ids.map(String) : [];
    ids.forEach((id, index) => {
      const camera = cameras.get(id);
      camera.card.hidden = index >= mode;
      const view = state.views?.[id];
      const status = typeof view === "string" ? view : view?.status;
      camera.status.textContent = humanize(status);
      camera.status.dataset.tone = /visible|tracking|active|ready|ok/i.test(status || "") ? "active" : /occlud|stale|lost|missing/i.test(status || "") ? "warning" : "neutral";
      camera.source.textContent = sources.includes(id) ? localizationWarning ? "Observation awaiting recovery" : "Contributing to estimate" : status ? "Not in current estimate" : "Awaiting observation";
      if (index < mode && state.asset_ready !== false) updateFrame(id, typeof state.frame_seq === "object" ? state.frame_seq?.[id] : state.frame_seq);
    });
    text("sim-time-value", Number.isFinite(state.sim_time) ? `${state.sim_time.toFixed(1)} s` : "—");
    text("processing-value", Number.isFinite(state.processing_ms) ? `${Math.round(state.processing_ms)} ms` : "—");
    text("pose-label", localization.status === "predicted" && localization.usable ? "Position (predicted)" : "Position");
    text("pose-value", localization.positionText);
    text("radius-value", localization.radiusText);
    $("last-seen-row").hidden = !localization.ghostPose;
    text("last-seen-value", localization.lastSeenText || "—");
    $("last-seen-legend").hidden = !localization.ghostPose;
    $("estimate-title").closest(".estimate-card").dataset.localization = localization.status;
    text("goal-value", localization.showRoute && finitePoint(state.goal) ? coordinates(state.goal) : "Not set");
    text("controller-value", localizationWarning ? localization.label : recovered ? "Stopped · choose a goal" : humanize(state.controller_status || state.phase));
    text("map-instructions-text", localizationWarning ? localization.message : recovered ? "Position recovered. Choose a new destination." : "Click free space to request a destination.");
    const phase = String(state.phase || "waiting");
    const tone = /error|fail/i.test(phase) ? "error" : localizationWarning ? "warning" : recovered ? "neutral" : /running|navigat|active/i.test(phase) ? "active" : /stop|reset|loading|initializ|missing/i.test(phase) ? "warning" : "neutral";
    text("phase-text", localizationWarning ? localization.label : recovered ? "Localized · choose a goal" : humanize(phase));
    $("phase-chip").dataset.tone = tone;
    $("asset-notice").hidden = state.asset_ready !== false;
    if (state.asset_ready === false) text("asset-message", state.message || "The local app has not loaded its map and policy bundle. Follow the asset setup instructions, then restart it.");
    if (localizationWarning || recovered || Date.now() >= toastUntil) {
      const defaultMessage = state.asset_ready === false ? "Navigation becomes available after the demo assets are loaded."
        : finitePoint(state.goal) ? "Following the requested destination using camera estimates and remembered free space."
          : "Choose a destination on the map, or start with the prepared demo route.";
      activity(localizationWarning ? localization.label : recovered ? "Position recovered" : humanize(state.controller_status || state.phase || "Ready when you are"), localization.message || state.message || defaultMessage, tone);
    }
    updateControls();
    scheduleDraw();
  }

  async function pollState() {
    if (polling) return;
    polling = true;
    try {
      const latest = await requestJSON("/api/state");
      if (!latest || typeof latest !== "object" || Array.isArray(latest)) throw new Error("The local app returned an invalid state.");
      const previous = state;
      state = latest;
      token = typeof state.token === "string" && state.token.length <= 1024 ? state.token : null;
      setConnection(true);
      renderState(previous);
      void loadMap();
    } catch (error) {
      setConnection(false);
      activity("Local app unavailable", `${error.message} Navigation controls are disabled until reconnection.`, "error");
      if (!map) {
        text("map-placeholder-title", "Waiting for the local app");
        text("map-placeholder-description", "Start the BB8-RL application to load its saved map and live observations.");
      }
    } finally { polling = false; }
  }

  async function command(payload, title, message) {
    if (!connected || !token) { activity("Not connected", "Reconnect to the local app before issuing a command.", "error", true); return false; }
    const generation = ++commandGeneration;
    pendingCommands += 1;
    updateControls();
    activity(title, message, payload.action === "stop" ? "warning" : "neutral", true);
    try {
      const response = await requestJSON("/api/command", {
        method: "POST", headers: { "Content-Type": "application/json", "X-BB8-Token": token }, body: JSON.stringify(payload),
      });
      if (generation === commandGeneration && response?.message) activity(title, response.message, "neutral", true);
      void pollState();
      return true;
    } catch (error) {
      if (generation === commandGeneration) activity("Request not accepted", error.message, "error", true);
      return false;
    } finally { pendingCommands -= 1; updateControls(); }
  }

  async function submitGoal(point) {
    if (!canNavigate()) {
      const localization = localizationView(state || {});
      activity(localization.message ? localization.label : "Navigation is not ready", localization.message || "Wait for the camera runtime and map to become available.", "warning", true);
      return;
    }
    if (!finitePoint(point)) { activity("Enter a destination", "Both X and Y must be finite coordinates in metres.", "warning", true); return; }
    const cell = cellAtWorld(point, map);
    selectedGoal = { point: [...point], status: cell === 1 ? "pending" : "rejected" };
    $("goal-x").value = point[0].toFixed(2);
    $("goal-y").value = point[1].toFixed(2);
    text("map-selection", `${cell === 1 ? "Selected" : "Rejected"}: ${coordinates(point)}`);
    scheduleDraw();
    if (cell !== 1) {
      const reason = cell === 2 ? "That cell contains occupied evidence." : cell === 0 ? "That area is unknown; it cannot be used as free space." : "The destination is outside the saved map.";
      activity("Choose observed free space", `${reason} Select another point.`, "warning", true);
      return;
    }
    const requested = selectedGoal;
    const accepted = await command({ action: "goal", x: point[0], y: point[1] }, "Destination requested", "The controller is checking the complete route and clearance envelope.");
    if (selectedGoal === requested) {
      selectedGoal.status = accepted ? "submitted" : "rejected";
      text("map-selection", `${accepted ? "Requested" : "Rejected"}: ${coordinates(point)}`);
      scheduleDraw();
    }
  }

  function stop() {
    selectedGoal = null;
    text("map-selection", "Stop requested");
    scheduleDraw();
    void command({ action: "stop" }, "Stop requested", "Waiting for the controller to cancel motion and clear the route.");
  }

  function scheduleDraw() {
    if (drawRequested) return;
    drawRequested = true;
    window.requestAnimationFrame(() => { drawRequested = false; drawMap(); });
  }

  function drawMap() {
    const localization = localizationView(state || {});
    const size = canvas.getBoundingClientRect();
    if (size.width < 1 || size.height < 1 || !context) return;
    const ratio = window.devicePixelRatio || 1;
    if (canvas.width !== Math.round(size.width * ratio) || canvas.height !== Math.round(size.height * ratio)) {
      canvas.width = Math.round(size.width * ratio); canvas.height = Math.round(size.height * ratio);
    }
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, size.width, size.height);
    context.fillStyle = "#eef1e8";
    context.fillRect(0, 0, size.width, size.height);
    if (!map || !mapImage) return;
    const rect = plotRect(map.bounds, size.width, size.height, padding);
    if (!rect) return;
    const project = (point) => worldToCanvas(point, map.bounds, size.width, size.height, padding);
    context.save();
    context.beginPath(); context.rect(rect.x, rect.y, rect.width, rect.height); context.clip();
    context.fillStyle = "#d9dfd8"; context.fillRect(rect.x, rect.y, rect.width, rect.height);
    context.strokeStyle = "#b8c6c0"; context.lineWidth = 0.6;
    for (let line = -rect.height; line < rect.width; line += 7) {
      context.beginPath(); context.moveTo(rect.x + line, rect.y + rect.height); context.lineTo(rect.x + line + rect.height, rect.y); context.stroke();
    }
    context.imageSmoothingEnabled = false;
    context.drawImage(mapImage, rect.x, rect.y, rect.width, rect.height);
    const span = Math.max(map.bounds[2] - map.bounds[0], map.bounds[3] - map.bounds[1]);
    const gridStep = span <= 2 ? 0.5 : span <= 6 ? 1 : span <= 12 ? 2 : 5;
    context.strokeStyle = "#263f4b18"; context.lineWidth = 0.65;
    for (let x = Math.ceil(map.bounds[0] / gridStep) * gridStep; x <= map.bounds[2]; x += gridStep) {
      const a = project([x, map.bounds[1]]), b = project([x, map.bounds[3]]);
      context.beginPath(); context.moveTo(...a); context.lineTo(...b); context.stroke();
    }
    for (let y = Math.ceil(map.bounds[1] / gridStep) * gridStep; y <= map.bounds[3]; y += gridStep) {
      const a = project([map.bounds[0], y]), b = project([map.bounds[2], y]);
      context.beginPath(); context.moveTo(...a); context.lineTo(...b); context.stroke();
    }
    // Declared virtual task zones, not camera-recognized objects or map occupancy.
    const stations = Array.isArray(state?.agency?.stations) ? state.agency.stations.slice(0, 8) : [];
    for (const station of stations) {
      if (!finitePoint(station.goal)) continue;
      const [sx, sy] = project(station.goal);
      context.beginPath(); context.arc(sx, sy, 0.14 * rect.scale, 0, Math.PI * 2);
      context.fillStyle = "#8566bb20"; context.fill(); context.strokeStyle = "#76589f"; context.lineWidth = 1.3;
      context.setLineDash([3, 3]); context.stroke(); context.setLineDash([]);
      context.font = "600 10px system-ui, sans-serif"; context.fillStyle = "#604782";
      context.fillText(typeof station.label === "string" ? station.label.slice(0, 40) : "Station", sx - 24, sy - 0.14 * rect.scale - 5);
    }
    const route = localization.showRoute && Array.isArray(state?.route) ? state.route.filter(finitePoint) : [];
    if (route.length > 1) {
      context.beginPath(); route.forEach((point, index) => { const projected = project(point); if (index === 0) context.moveTo(...projected); else context.lineTo(...projected); });
      context.lineWidth = 2.5; context.strokeStyle = "#078e98"; context.lineJoin = "round"; context.lineCap = "round"; context.stroke();
      route.slice(1, -1).forEach((point) => { context.beginPath(); context.arc(...project(point), 2.5, 0, Math.PI * 2); context.fillStyle = "#078e98"; context.fill(); });
    }
    const cursor = canvasFocused && keyboardPoint ? keyboardPoint : hoverPoint;
    if (cursor) {
      const [x, y] = project(cursor);
      context.strokeStyle = "#486f7d90"; context.lineWidth = 1;
      context.beginPath(); context.moveTo(x - 7, y); context.lineTo(x + 7, y); context.moveTo(x, y - 7); context.lineTo(x, y + 7); context.stroke();
    }
    if (localization.currentPose) {
      const point = project(localization.currentPose);
      if (localization.currentRadius !== null) {
        context.beginPath(); context.arc(...point, Math.max(0, localization.currentRadius * rect.scale), 0, Math.PI * 2);
        context.fillStyle = "#e99d3930"; context.fill(); context.strokeStyle = "#bd792f"; context.lineWidth = 1; context.setLineDash([4, 3]); context.stroke(); context.setLineDash([]);
      }
      context.beginPath(); context.arc(...point, 7, 0, Math.PI * 2); context.fillStyle = "#f4b350"; context.fill(); context.lineWidth = 2; context.strokeStyle = "#fdf8eb"; context.stroke();
      context.beginPath(); context.arc(...point, 2.5, 0, Math.PI * 2); context.strokeStyle = "#72562c"; context.lineWidth = 1.2; context.stroke();
      const label = localization.status === "predicted" ? "BB-8 · predicted" : "BB-8";
      context.font = "600 9px system-ui, sans-serif"; context.lineWidth = 3; context.strokeStyle = "#f7f9f2"; context.strokeText(label, point[0] + 11, point[1] - 9); context.fillStyle = "#344d50"; context.fillText(label, point[0] + 11, point[1] - 9);
    } else if (localization.ghostPose) {
      // A historical point only: no uncertainty circle or braking region can
      // imply a usable current position while localization is expired.
      const point = project(localization.ghostPose);
      context.strokeStyle = "#66777c"; context.lineWidth = 1.5; context.setLineDash([3, 3]);
      context.beginPath(); context.arc(...point, 8, 0, Math.PI * 2); context.stroke(); context.setLineDash([]);
      context.beginPath(); context.moveTo(point[0] - 3, point[1]); context.lineTo(point[0] + 3, point[1]); context.moveTo(point[0], point[1] - 3); context.lineTo(point[0], point[1] + 3); context.stroke();
      context.font = "600 9px system-ui, sans-serif";
      const labelWidth = context.measureText(localization.ghostLabel).width;
      const labelX = Math.max(rect.x + 3, Math.min(point[0] + 12, rect.x + rect.width - labelWidth - 3));
      const labelY = Math.max(rect.y + 12, point[1] - 12);
      context.lineWidth = 3; context.strokeStyle = "#f7f9f2"; context.strokeText(localization.ghostLabel, labelX, labelY);
      context.fillStyle = "#56696e"; context.fillText(localization.ghostLabel, labelX, labelY);
    }
    function targetMarker(point, color, dashed, rejected) {
      const [x, y] = project(point);
      context.strokeStyle = color; context.lineWidth = 1.7; context.setLineDash(dashed ? [3, 3] : []);
      context.beginPath(); context.arc(x, y, 9, 0, Math.PI * 2); context.stroke(); context.setLineDash([]);
      context.beginPath();
      if (rejected) { context.moveTo(x - 4, y - 4); context.lineTo(x + 4, y + 4); context.moveTo(x + 4, y - 4); context.lineTo(x - 4, y + 4); }
      else { context.moveTo(x - 4, y); context.lineTo(x + 4, y); context.moveTo(x, y - 4); context.lineTo(x, y + 4); }
      context.stroke();
    }
    if (localization.showRoute && finitePoint(state?.goal)) targetMarker(state.goal, "#047c83", false, false);
    if (localization.showRoute && selectedGoal && (!finitePoint(state?.goal) || Math.hypot(state.goal[0] - selectedGoal.point[0], state.goal[1] - selectedGoal.point[1]) > 0.005)) {
      targetMarker(selectedGoal.point, selectedGoal.status === "rejected" ? "#b6534d" : "#aa742f", true, selectedGoal.status === "rejected");
    }
    context.restore();
    context.strokeStyle = "#9badab"; context.lineWidth = 1; context.strokeRect(rect.x, rect.y, rect.width, rect.height);
    context.fillStyle = "#64817f"; context.font = "9px system-ui, sans-serif"; context.textAlign = "center";
    for (let x = Math.ceil(map.bounds[0] / gridStep) * gridStep; x <= map.bounds[2]; x += gridStep) { const p = project([x, map.bounds[1]]); context.fillText(String(x), p[0], rect.y + rect.height + 15); }
    context.textAlign = "right";
    for (let y = Math.ceil(map.bounds[1] / gridStep) * gridStep; y <= map.bounds[3]; y += gridStep) { const p = project([map.bounds[0], y]); context.fillText(String(y), rect.x - 9, p[1] + 3); }
    context.textAlign = "left"; context.fillText("+Y", rect.x + 2, rect.y - 12);
    context.textAlign = "right"; context.fillText("+X", rect.x + rect.width, rect.y + rect.height + 29);
    context.textAlign = "left";
  }

  function showCoordinate(point) {
    if (!point) { text("coordinate-value", "Move over the map"); text("coordinate-state", ""); return; }
    text("coordinate-value", coordinates(point));
    const cell = cellAtWorld(point, map);
    text("coordinate-state", cell === 1 ? "Observed free" : cell === 2 ? "Occupied" : cell === 0 ? "Unknown" : "Outside map");
    $("coordinate-state").dataset.cell = String(cell);
  }

  function eventPoint(event) {
    if (!map) return null;
    const bounds = canvas.getBoundingClientRect();
    return canvasToWorld([event.clientX - bounds.left, event.clientY - bounds.top], map.bounds, bounds.width, bounds.height, padding);
  }

  canvas.addEventListener("pointermove", (event) => { hoverPoint = eventPoint(event); showCoordinate(hoverPoint); scheduleDraw(); });
  canvas.addEventListener("pointerleave", () => { hoverPoint = null; if (!canvasFocused) showCoordinate(null); scheduleDraw(); });
  canvas.addEventListener("click", (event) => { const point = eventPoint(event); if (point) void submitGoal(point); });
  canvas.addEventListener("focus", () => {
    canvasFocused = true;
    if (map) keyboardPoint = selectedGoal?.point || localizationView(state || {}).currentPose || [(map.bounds[0] + map.bounds[2]) / 2, (map.bounds[1] + map.bounds[3]) / 2];
    showCoordinate(keyboardPoint); scheduleDraw();
  });
  canvas.addEventListener("blur", () => { canvasFocused = false; showCoordinate(hoverPoint); scheduleDraw(); });
  canvas.addEventListener("keydown", (event) => {
    if (!map || !keyboardPoint) return;
    const movement = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, 1], ArrowDown: [0, -1] }[event.key];
    if (movement) {
      event.preventDefault();
      const step = event.shiftKey ? 0.2 : Math.max(map.resolution, 0.05);
      keyboardPoint = [Math.max(map.bounds[0], Math.min(map.bounds[2] - map.resolution / 2, keyboardPoint[0] + movement[0] * step)), Math.max(map.bounds[1], Math.min(map.bounds[3] - map.resolution / 2, keyboardPoint[1] + movement[1] * step))];
      showCoordinate(keyboardPoint); scheduleDraw();
    } else if (event.key === "Enter" || event.key === " ") { event.preventDefault(); void submitGoal(keyboardPoint); }
  });
  $("goal-form").addEventListener("submit", (event) => { event.preventDefault(); void submitGoal([$("goal-x").valueAsNumber, $("goal-y").valueAsNumber]); });
  $("stop-button").addEventListener("click", stop);
  $("autonomy-button").addEventListener("click", () => {
    const agency = agencyView(state || {}, { connected, token, mapReady: !!map, pendingCommands });
    if (!agency.canToggle) return;
    const enabled = !agency.enabled;
    if (enabled) { selectedGoal = null; text("map-selection", agency.purpose ? "Purposeful interaction requested" : "Autonomous exploration requested"); }
    void command({ action: "autonomy", enabled }, enabled ? agency.purpose ? "Learning requested" : "Exploration requested" : "Pause requested",
      enabled ? agency.purpose ? "BB-8 will compare station outcomes with its current simulated resource need." : "BB-8 will choose map targets without a completed exploration visit." : "Waiting for the controller to pause autonomy and stop motion.");
  });
  window.addEventListener("keydown", (event) => { if (event.key === "Escape" && connected && token) { event.preventDefault(); stop(); } });
  $("reset-button").addEventListener("click", () => { selectedGoal = null; text("map-selection", "No target selected"); frameEpoch += 1; void command({ action: "reset" }, "Reset requested", "The episode, position estimate and route will restart."); });
  $("demo-button").addEventListener("click", () => {
    if (!availableControls().demo) return;
    selectedGoal = finitePoint(map?.demo_goal) ? { point: [...map.demo_goal], status: "submitted" } : null;
    text("map-selection", selectedGoal ? `Demo target: ${coordinates(selectedGoal.point)}` : "Demo route requested");
    void command({ action: "demo" }, "Demo reset requested", "Resetting the episode before the saved demonstration route.");
  });
  modeButtons.forEach((button) => button.addEventListener("click", async () => {
    const mode = Number(button.dataset.mode);
    if (mode === state?.mode) return;
    button.dataset.pending = "true"; selectedGoal = null; text("map-selection", "No target selected");
    await command({ action: "mode", mode }, `Switching to ${mode} ${mode === 1 ? "camera" : "cameras"}`, "The episode resets when its camera configuration changes.");
    delete button.dataset.pending;
  }));
  window.addEventListener("pagehide", () => {
    if (token) void fetch("/api/command", { method: "POST", headers: { "Content-Type": "application/json", "X-BB8-Token": token }, body: JSON.stringify({ action: "stop" }), keepalive: true, mode: "same-origin", credentials: "same-origin" }).catch(() => {});
  });
  window.setInterval(async () => {
    if (!connected || !token || heartbeatBusy) return;
    heartbeatBusy = true;
    try { await requestJSON("/api/command", { method: "POST", headers: { "Content-Type": "application/json", "X-BB8-Token": token }, body: JSON.stringify({ action: "heartbeat" }) }, 1800); }
    catch (error) { setConnection(false); activity("Heartbeat interrupted", `${error.message} Waiting for reconnection.`, "error"); }
    finally { heartbeatBusy = false; }
  }, 1000);
  if (typeof ResizeObserver !== "undefined") new ResizeObserver(scheduleDraw).observe($("map-stage"));
  else window.addEventListener("resize", scheduleDraw);
  updateControls();
  scheduleDraw();
  void pollState();
  window.setInterval(pollState, 250);
})();
