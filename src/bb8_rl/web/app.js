/* BB8-RL control room. No dependencies; the server remains command authority. */
(function () {
  "use strict";

  const finitePoint = (point) => Array.isArray(point) && point.length === 2 && point.every(Number.isFinite);

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

  const helpers = Object.freeze({ plotRect, worldToCanvas, canvasToWorld, worldToCell, cellAtWorld, validMap });
  if (typeof module !== "undefined" && module.exports) module.exports = helpers;
  if (typeof window === "undefined") return;
  window.BB8Map = helpers;
  if (typeof document === "undefined") return;

  const $ = (id) => document.getElementById(id);
  const text = (id, value) => { if ($(id).textContent !== String(value)) $(id).textContent = String(value); };
  const humanize = (value) => String(value || "Waiting").replace(/[_-]+/g, " ").replace(/^\w/, (letter) => letter.toUpperCase()).slice(0, 180);
  const coordinates = (point) => finitePoint(point) ? `${point[0].toFixed(2)}, ${point[1].toFixed(2)} m` : "—";
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
    const phase = String(state?.phase || "").toLowerCase();
    return connected && !!token && state?.asset_ready === true && !!map
      && !/error|failed|loading|starting|initializ|resetting|stopping|closed/.test(phase) && pendingCommands === 0;
  }

  function updateControls() {
    const ready = canNavigate();
    ["goal-x", "goal-y", "goal-button", "demo-button"].forEach((id) => { $(id).disabled = !ready; });
    $("stop-button").disabled = !connected || !token;
    $("stop-button").title = connected ? "Cancel motion and the current route (Escape)" : "Stop is unavailable until the local app reconnects";
    $("reset-button").disabled = !connected || !token || state?.asset_ready !== true || pendingCommands > 0;
    modeButtons.forEach((button) => { button.disabled = !connected || !token || state?.asset_ready !== true || pendingCommands > 0; });
    $("map-stage").dataset.enabled = String(ready);
    canvas.setAttribute("aria-disabled", String(!ready));
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
      camera.source.textContent = sources.includes(id) ? "Contributing to estimate" : status ? "Not in current estimate" : "Awaiting observation";
      if (index < mode && state.asset_ready !== false) updateFrame(id, typeof state.frame_seq === "object" ? state.frame_seq?.[id] : state.frame_seq);
    });
    text("sim-time-value", Number.isFinite(state.sim_time) ? `${state.sim_time.toFixed(1)} s` : "—");
    text("processing-value", Number.isFinite(state.processing_ms) ? `${Math.round(state.processing_ms)} ms` : "—");
    text("pose-value", finitePoint(state.pose) ? coordinates(state.pose) : "Waiting for vision");
    text("radius-value", Number.isFinite(state.position_radius) && state.position_radius >= 0 ? `${(state.position_radius * 100).toFixed(1)} cm` : "—");
    text("goal-value", finitePoint(state.goal) ? coordinates(state.goal) : "Not set");
    text("controller-value", humanize(state.controller_status || state.phase));
    const phase = String(state.phase || "waiting");
    const tone = /error|fail/i.test(phase) ? "error" : /running|navigat|active/i.test(phase) ? "active" : /stop|reset|loading|initializ|missing/i.test(phase) ? "warning" : "neutral";
    text("phase-text", humanize(phase));
    $("phase-chip").dataset.tone = tone;
    $("asset-notice").hidden = state.asset_ready !== false;
    if (state.asset_ready === false) text("asset-message", state.message || "The local app has not loaded its map and policy bundle. Follow the asset setup instructions, then restart it.");
    if (Date.now() >= toastUntil) {
      const defaultMessage = state.asset_ready === false ? "Navigation becomes available after the demo assets are loaded."
        : finitePoint(state.goal) ? "Following the requested destination using camera estimates and remembered free space."
          : "Choose a destination on the map, or start with the prepared demo route.";
      activity(humanize(state.controller_status || state.phase || "Ready when you are"), state.message || defaultMessage, tone);
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
    if (!canNavigate()) { activity("Navigation is not ready", "Wait for the camera runtime and map to become available.", "warning", true); return; }
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
    const route = Array.isArray(state?.route) ? state.route.filter(finitePoint) : [];
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
    if (finitePoint(state?.pose)) {
      const point = project(state.pose);
      if (Number.isFinite(state.position_radius) && state.position_radius >= 0) {
        context.beginPath(); context.arc(...point, Math.max(0, state.position_radius * rect.scale), 0, Math.PI * 2);
        context.fillStyle = "#e99d3930"; context.fill(); context.strokeStyle = "#bd792f"; context.lineWidth = 1; context.setLineDash([4, 3]); context.stroke(); context.setLineDash([]);
      }
      context.beginPath(); context.arc(...point, 7, 0, Math.PI * 2); context.fillStyle = "#f4b350"; context.fill(); context.lineWidth = 2; context.strokeStyle = "#fdf8eb"; context.stroke();
      context.beginPath(); context.arc(...point, 2.5, 0, Math.PI * 2); context.strokeStyle = "#72562c"; context.lineWidth = 1.2; context.stroke();
      context.font = "600 9px system-ui, sans-serif"; context.lineWidth = 3; context.strokeStyle = "#f7f9f2"; context.strokeText("BB-8", point[0] + 11, point[1] - 9); context.fillStyle = "#344d50"; context.fillText("BB-8", point[0] + 11, point[1] - 9);
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
    if (finitePoint(state?.goal)) targetMarker(state.goal, "#047c83", false, false);
    if (selectedGoal && (!finitePoint(state?.goal) || Math.hypot(state.goal[0] - selectedGoal.point[0], state.goal[1] - selectedGoal.point[1]) > 0.005)) {
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
    if (map) keyboardPoint = selectedGoal?.point || (finitePoint(state?.pose) ? [...state.pose] : [(map.bounds[0] + map.bounds[2]) / 2, (map.bounds[1] + map.bounds[3]) / 2]);
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
  window.addEventListener("keydown", (event) => { if (event.key === "Escape" && connected && token) { event.preventDefault(); stop(); } });
  $("reset-button").addEventListener("click", () => { selectedGoal = null; text("map-selection", "No target selected"); frameEpoch += 1; void command({ action: "reset" }, "Reset requested", "The episode, position estimate and route will restart."); });
  $("demo-button").addEventListener("click", () => {
    if (!canNavigate()) return;
    selectedGoal = finitePoint(map?.demo_goal) ? { point: [...map.demo_goal], status: "submitted" } : null;
    text("map-selection", selectedGoal ? `Demo target: ${coordinates(selectedGoal.point)}` : "Demo route requested");
    void command({ action: "demo" }, "Demo route requested", "The local app is preparing the saved demonstration route.");
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
