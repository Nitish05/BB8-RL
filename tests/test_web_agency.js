"use strict";
const assert = require("node:assert/strict");
const { agencyView } = require("../src/bb8_rl/web/app.js");

const session = { connected: true, token: "local-session", mapReady: true, pendingCommands: 0 };
const state = {
  asset_ready: true, phase: "idle", localization_status: "measured", localization_valid: true,
  pose: [0, 0], position_radius: 0.04,
  agency: { enabled: false, available: true, status: "paused", episodes: 2, message: "Remembered visits are available.", preferences: [] },
};
const paused = agencyView(state, session);
assert.equal(paused.canToggle, true);
assert.equal(paused.enabled, false);
assert.equal(paused.buttonText, "Start exploring");
assert.equal(paused.experienceText, "2 experiences remembered");

// Old servers and incomplete startup states must never offer an enable action.
for (const agency of [undefined, null, [], "enabled", { enabled: "true", available: "true" }]) {
  assert.equal(agencyView({ ...state, agency }, session).canToggle, false);
  assert.equal(agencyView({ ...state, agency }, session).enabled, false);
}
assert.equal(agencyView({ ...state, agency: undefined }, session).status, "Unavailable in this session");
for (const override of [{ connected: false }, { token: null }, { mapReady: false }, { pendingCommands: 1 }]) {
  assert.equal(agencyView(state, { ...session, ...override }).canToggle, false);
}
for (const override of [
  { asset_ready: false }, { phase: "starting" }, { phase: "resetting" },
  { localization_status: "lost", localization_valid: false },
  { localization_status: "reacquiring", localization_valid: false },
]) {
  assert.equal(agencyView({ ...state, ...override }, session).canToggle, false);
}

const exploring = {
  ...state,
  agency: { ...state.agency, enabled: true, intention: { goal: [1, 0], label: "East clearing", explanation: "Previous visits completed successfully." } },
};
assert.equal(agencyView(exploring, session).buttonText, "Pause exploration");
assert.equal(agencyView(exploring, session).intention.label, "East clearing");
// Cancellation remains possible when a stale server state says exploration is
// active, even if its navigation readiness has just disappeared.
assert.equal(agencyView({ ...exploring, asset_ready: false }, session).canToggle, true);
assert.equal(agencyView({ ...exploring, localization_status: "lost", localization_valid: false }, session).canToggle, true);
assert.equal(agencyView({ ...exploring, localization_status: "lost", localization_valid: false }, session).intention, null);
assert.equal(agencyView(exploring, { ...session, connected: false }).intention, null);
assert.equal(agencyView(exploring, { ...session, connected: false }).canToggle, false);
assert.equal(agencyView(exploring, { ...session, pendingCommands: 1 }).canToggle, false);

// Reset/Stop state must hide stale intentions while retaining historical data.
const remembered = { ...state, agency: { ...exploring.agency, enabled: false } };
assert.equal(agencyView(remembered, session).intention, null);
assert.equal(agencyView(remembered, session).experienceText, "2 experiences remembered");
const recent = { ...state, agency: { ...state.agency, recent_experience: { label: "East clearing", outcome: "rejected", summary: "East clearing: route rejected" } } };
assert.equal(agencyView(recent, session).recentExperience, "East clearing: route rejected");
assert.equal(agencyView(state, session).recentExperience, null);
assert.equal(agencyView({ ...state, agency: { ...state.agency, recent_experience: { summary: { unsafe: "object" } } } }, session).recentExperience, null);
const rawPreferences = [
  { label: "West clearing", value: -0.1, visits: 2 },
  { label: "East clearing", value: 0.8, visits: 3 },
  { label: "Unvisited", value: 100, visits: 0 },
  { label: "Invalid", value: Infinity, visits: 7 },
  { label: "Invalid", value: 0, visits: -1 },
  null,
];
const preferences = agencyView({ ...state, agency: { ...state.agency, preferences: rawPreferences } }, session).preferences;
assert.deepEqual(preferences, [{ label: "East clearing", visits: 3 }, { label: "West clearing", visits: 2 }]);
assert.equal(rawPreferences[0].label, "West clearing");
for (const episodes of [-1, 1.5, Infinity, "2"]) {
  assert.equal(agencyView({ ...state, agency: { ...state.agency, episodes } }, session).experienceText, "0 experiences remembered");
}
assert.equal(agencyView({ ...state, agency: { ...state.agency, episodes: 1 } }, session).experienceText, "1 experience remembered");
console.log("Agency UI tests passed: explicit start, session readiness, cancellation, missing-server fallback, historical experience and stale-intention suppression.");

const purposeState = {...state, agency: {...state.agency, selection_policy: "learned_station_outcomes", resource: .76, target: .8, status: "paused", preferences: [{label: "Station B", value: .3, visits: 2, response_probability: .75}]}};
assert.equal(agencyView(purposeState, session).buttonText, "Start learning");
assert.equal(agencyView(purposeState, session).resource, .76);
assert.equal(agencyView(purposeState, session).preferences[0].response, "75% predicted response");
assert.equal(agencyView({...purposeState, agency: {...purposeState.agency, enabled: true, status: "satisfied"}}, session).status, "Need satisfied · waiting");
assert.equal(agencyView({...purposeState, agency: {...purposeState.agency, enabled: true, status: "interacting"}}, session).status, "Testing a station response");
for (const resource of [NaN, Infinity, -1, 2, ".8"]) {
  assert.equal(agencyView({...purposeState, agency: {...purposeState.agency, resource}}, session).resource, null);
}
assert.equal(agencyView({...purposeState, localization_status: "lost", localization_valid: false, agency: {...purposeState.agency, enabled: true}}, session).canToggle, true);
console.log("Purpose UI tests passed: explicit start, simulated resource, learned predictions, satisfied idle and loss-safe pause.");

// A completed scene warmup must not keep its old pause reason visible once the
// user can explicitly start again. Rendering must not resume learning or mutate
// retained memory, and current safety messages still take precedence.
const warmupReason = "Checking a stable camera reference; navigation is paused.";
const recoveredScene = {
  ...purposeState,
  scene_validity: { ready: true, navigation_allowed: true, invalidated: false },
  agency: { ...purposeState.agency, enabled: false, status: "paused", message: warmupReason },
};
const retainedScene = JSON.stringify(recoveredScene);
const recoveredView = agencyView(recoveredScene, session);
assert.equal(recoveredView.message, "Camera reference is ready. Start learning when you are ready; remembered outcomes are retained.");
assert.equal(recoveredView.buttonText, "Start learning");
assert.equal(recoveredView.canToggle, true);
assert.equal(recoveredView.enabled, false);
assert.equal(recoveredView.intention, null);
assert.equal(recoveredView.experienceText, "2 experiences remembered");
assert.equal(JSON.stringify(recoveredScene), retainedScene);
for (const override of [
  { scene_validity: { ready: false, navigation_allowed: false, invalidated: false } },
  { scene_validity: { ready: true, navigation_allowed: false, invalidated: true } },
  { localization_status: "lost", localization_valid: false },
]) {
  const blockedView = agencyView({ ...recoveredScene, ...override }, session);
  assert.equal(blockedView.canToggle, false);
  assert.notEqual(blockedView.message, recoveredView.message);
  assert.match(blockedView.message, /paused|locked|unavailable/);
}
assert.notEqual(agencyView(recoveredScene, { ...session, connected: false }).message, recoveredView.message);
for (const agency of [
  { ...recoveredScene.agency, enabled: true },
  { ...recoveredScene.agency, available: false },
  { ...recoveredScene.agency, message: "Manual destination selected." },
]) {
  assert.equal(agencyView({ ...recoveredScene, agency }, session).message, agency.message);
}
assert.equal(agencyView({ ...recoveredScene, agency: { ...recoveredScene.agency, selection_policy: "coverage" } }, session).message,
  "Camera reference is ready. Start exploring when you are ready; remembered visits are retained.");
console.log("Scene warmup UI regression passed: current readiness replaces only the expired pause message without changing authority or memory.");

// Visual mode is selected by the explicit native RGB source, while retaining
// purpose learning and all existing motion/readiness gates.
const visualState = {
  ...purposeState,
  agency: {...purposeState.agency, resource_source: "native_scene_rgb", resource: null, message: ""},
};
const visualBefore = JSON.stringify(visualState);
const visualView = agencyView(visualState, session);
assert.equal(visualView.purpose, true);
assert.equal(visualView.visual, true);
assert.equal(visualView.enabled, false);
assert.equal(visualView.buttonText, "Start learning");
assert.equal(visualView.resourceText, "Awaiting fixture pixels");
assert.match(visualView.caption, /live RGB fixture learning/);
assert.match(visualView.description, /native camera images/);
assert.match(visualView.help, /fixture-specific decoder/);
for (const field of ["caption", "description", "help", "resourceText", "resourceLabel", "resourceHelp", "enableMessage", "message"]) {
  assert.doesNotMatch(visualView[field], /telemetry|current simulated resource/i);
}
assert.equal(JSON.stringify(visualState), visualBefore);
assert.equal(agencyView({...visualState, agency: {...visualState.agency, resource: .5}}, session).resourceText, "Observed gauge · 50%");
assert.equal(agencyView({...visualState, agency: {...visualState.agency, enabled: true, status: "waiting"}}, session).status, "Awaiting fixture pixels");
assert.equal(agencyView({...visualState, agency: {...visualState.agency, enabled: true, status: "interacting"}}, session).status, "Observing a fixture response");
assert.equal(agencyView({...visualState, agency: {...visualState.agency, available: false}}, session).canToggle, false);
const visualLost = {...visualState, localization_status: "lost", localization_valid: false};
assert.equal(agencyView(visualLost, session).canToggle, false);
assert.equal(agencyView({...visualLost, agency: {...visualLost.agency, enabled: true}}, session).canToggle, true);
assert.equal(agencyView({...visualState, scene_validity: {invalidated: true, navigation_allowed: false}}, session).canToggle, false);
assert.equal(agencyView(visualState, {...session, connected: false}).canToggle, false);
for (const resource_source of [undefined, null, "simulated_station_telemetry", "native_scene_rgb ", true]) {
  const view = agencyView({...visualState, agency: {...visualState.agency, resource_source}}, session);
  assert.equal(view.visual, false);
  assert.equal(view.resourceText, "Awaiting telemetry");
}
assert.equal(agencyView({...visualState, agency: {...visualState.agency, selection_policy: "coverage"}}, session).visual, false);
assert.equal(visualView.resourceLabel, "Observed fixture gauge");
assert.match(visualView.resourceHelp, /before\/after native RGB/);
assert.match(visualView.resourceHelp, /fixed synthetic markers and gauges/);
for (const mode of [1, 2, 3]) {
  const visual = agencyView({...visualState, mode}, session);
  assert.equal(visual.cameraCount, `${mode} NAV + 1 RGB`);
  assert.match(visual.cameraSetup, /navigation cameras/);
  assert.match(visual.cameraDisclosure, /additional camera at the registered B view/);
  assert.match(visual.cameraDisclosure, /feed is not displayed/);
  const purpose = agencyView({...purposeState, mode}, session);
  assert.equal(purpose.cameraCount, `${mode} ${mode === 1 ? "VIEW" : "VIEWS"}`);
  assert.equal(purpose.cameraDisclosure, "");
  assert.equal(purpose.resourceLabel, "Simulated resource");
  assert.match(purpose.resourceHelp, /Effects come from simulation telemetry/);
}
for (const mode of [0, 4, "3", NaN, null]) {
  assert.equal(agencyView({...visualState, mode}, session).cameraCount, "1 NAV + 1 RGB");
}
console.log("Visual fixture UI tests passed: explicit RGB provenance, pixel waiting, bounded claims, unchanged cancellation and memory-safe rendering.");
