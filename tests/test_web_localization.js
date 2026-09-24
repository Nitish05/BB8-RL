"use strict";
const assert = require("node:assert/strict");
const { localizationView, controlAvailability } = require("../src/bb8_rl/web/app.js");

const ready = { connected: true, token: "local-session", mapReady: true, pendingCommands: 0 };
const measured = {
  asset_ready: true, phase: "idle", localization_status: "measured", localization_valid: true,
  pose: [-0.2, 0.4], position_radius: 0.035, requires_new_goal: false,
};
assert.equal(localizationView(measured).positionText, "-0.20, 0.40 m");
assert.equal(localizationView(measured).radiusText, "3.5 cm");
assert.equal(controlAvailability(measured, ready).navigate, true);

// Brief accepted predictions remain useful, but have a different presentation.
const predicted = localizationView({ ...measured, localization_status: "predicted" });
assert.equal(predicted.label, "Predicted position");
assert.deepEqual(predicted.currentPose, measured.pose);
assert.equal(predicted.currentRadius, measured.position_radius);
assert.equal(predicted.ghostPose, null);

// Even contradictory stale pose fields and a huge raw enclosure cannot become
// a current robot marker, uncertainty circle, or usable navigation position.
const lost = {
  ...measured, phase: "localization_lost", localization_status: "lost", localization_valid: false,
  pose: [1.7, -1.7], position_radius: 7.192, raw_position_radius: 7.192,
  last_seen_pose: [-0.2, 0.4], last_seen_age_s: 241.25, last_seen_radius_m: 0.04,
  requires_new_goal: true, goal: [1, 1], route: [[-0.2, 0.4], [1, 1]],
};
const expired = localizationView(lost);
assert.equal(expired.label, "Localization lost");
assert.equal(expired.currentPose, null);
assert.equal(expired.currentRadius, null);
assert.equal(expired.positionText, "Unavailable");
assert.equal(expired.radiusText, "Unavailable");
assert.deepEqual(expired.ghostPose, [-0.2, 0.4]);
assert.equal(expired.ghostLabel, "Last seen · 241.3 s ago (sim)");
assert.equal(expired.lastSeenText, "-0.20, 0.40 m · 241.3 s ago (sim)");
assert.equal(expired.showRoute, false);
assert(!JSON.stringify(expired).includes("719.2"));
assert.deepEqual(controlAvailability(lost, ready), { navigate: false, stop: true, reset: true, mode: true, demo: true });

const reacquiring = { ...lost, localization_status: "reacquiring", phase: "reacquiring" };
const recovery = localizationView(reacquiring);
assert.equal(recovery.label, "Reacquiring position");
assert.equal(recovery.currentPose, null);
assert.equal(recovery.currentRadius, null);
assert.deepEqual(recovery.ghostPose, measured.pose);
assert.equal(controlAvailability(reacquiring, ready).navigate, false);
assert.equal(controlAvailability(reacquiring, ready).demo, true);

// A recovered localization permits an explicit new goal, while its cancelled
// destination/route stays hidden. This helper never creates a motion command.
const recovered = { ...measured, requires_new_goal: true };
assert.equal(controlAvailability(recovered, ready).navigate, true);
assert.equal(localizationView(recovered).showRoute, false);
assert.match(localizationView(recovered).message, /stopped; choose a new destination/);
assert.equal(localizationView({ ...recovered, requires_new_goal: false }).showRoute, true);

// The contract is authoritative whenever either field is present; only an old
// server with both fields absent gets the earlier readiness behavior.
const legacy = { asset_ready: true, phase: "idle", pose: [0, 0], position_radius: 0.05 };
assert.equal(localizationView(legacy).explicit, false);
assert.equal(controlAvailability(legacy, ready).navigate, true);
assert.equal(controlAvailability({ asset_ready: true, phase: "idle" }, ready).navigate, true);
for (const state of [
  { ...legacy, localization_status: "measured" },
  { ...legacy, localization_valid: true },
  { ...measured, localization_status: "unexpected" },
  { ...measured, localization_valid: false },
  { ...measured, localization_valid: "true" },
  { ...measured, pose: [NaN, 0] },
  { ...measured, position_radius: null },
  { ...measured, position_radius: -1 },
  { ...measured, position_radius: Infinity },
]) {
  assert.equal(controlAvailability(state, ready).navigate, false);
  assert.equal(localizationView(state).currentPose, null);
  assert.equal(localizationView(state).currentRadius, null);
}
assert.equal(controlAvailability({ ...legacy, phase: "starting" }, ready).navigate, false);
assert.equal(controlAvailability({ ...measured, localization_status: "uninitialized" }, ready).navigate, false);
assert.equal(localizationView({ ...lost, last_seen_pose: [0, Infinity] }).ghostPose, null);
assert.equal(localizationView({ ...lost, last_seen_age_s: -1 }).ghostLabel, "Last seen · time unavailable");

// Connection and in-flight command guards still apply independently. Reset,
// mode changes and demo resets never depend on valid localization or a UI map.
assert.deepEqual(controlAvailability(lost, { ...ready, connected: false }), { navigate: false, stop: false, reset: false, mode: false, demo: false });
assert.deepEqual(controlAvailability(lost, { ...ready, token: null }), { navigate: false, stop: false, reset: false, mode: false, demo: false });
assert.deepEqual(controlAvailability(lost, { ...ready, pendingCommands: 1 }), { navigate: false, stop: true, reset: false, mode: false, demo: false });
assert.equal(controlAvailability(lost, { ...ready, mapReady: false }).demo, true);
assert.equal(controlAvailability(measured, { ...ready, mapReady: false }).navigate, false);
console.log("Localization UI tests passed: explicit validity, loss/recovery, historical marker age, expired-radius suppression and recovery controls.");
