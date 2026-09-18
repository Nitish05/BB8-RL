# Synthetic BB-8 fixture

M1 uses native sphere primitives stored in `projects/bb8/empty-floor.genesis.json`.
No imported visual asset or manufacturer model is needed. The dark shell and pale head
are schematic original geometry. The body radius, total effective mass, inertia,
surface friction and command response are provisional synthetic values.

The navigation runtime positions the non-colliding head independently from the rolling
body. Opening the project in the regular Studio authoring viewer shows the initial
geometry; head following and planar commands run through `bb8-rl simulate`.
Measured collision and original visual assets can be placed here in a later milestone.
