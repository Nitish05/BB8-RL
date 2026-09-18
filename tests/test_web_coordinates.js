"use strict";
const assert = require("node:assert/strict");
const map = require("../src/bb8_rl/web/app.js");
const bounds = [-2, -2, 2, 2];
for (const [width, height] of [[900, 600], [390, 500], [800, 800]]) {
  for (const point of [[0, 0], [-1.75, 1.25], [0.24746, 1.30868], [1.99, -1.99]]) {
    const screen = map.worldToCanvas(point, bounds, width, height, 34);
    const restored = map.canvasToWorld(screen, bounds, width, height, 34);
    assert(restored.every((value, axis) => Math.abs(value - point[axis]) < 1e-12));
  }
  assert(map.worldToCanvas([0, 1], bounds, width, height, 34)[1] < map.worldToCanvas([0, -1], bounds, width, height, 34)[1]);
  assert.equal(map.canvasToWorld([0, 0], bounds, width, height, 34), null);
}
const grid = {bounds: [-1, -1, 1, 1], width: 2, height: 2, resolution: 1, cells: [[0, 1], [2, 1]]};
assert(map.validMap(grid));
assert.equal(map.cellAtWorld([-0.5, -0.5], grid), 0);
assert.equal(map.cellAtWorld([-0.5, 0.5], grid), 2);
assert.equal(map.cellAtWorld([0.5, -0.5], grid), 1);
assert.equal(map.worldToCell([1, 0], grid), null);
assert.equal(map.worldToCell([0, 1], grid), null);
assert.equal(map.validMap({...grid, cells: [[1, 1], [1, 3]]}), false);
assert.equal(map.worldToCanvas([NaN, 0], bounds, 400, 400), null);
console.log("Map coordinate tests passed: responsive round trips, y-axis, bounds and unknown/occupied classes.");
