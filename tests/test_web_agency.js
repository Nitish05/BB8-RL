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
