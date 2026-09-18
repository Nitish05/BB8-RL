# BB8-RL project instructions

- BB8-RL is a separate project using Genesis Studio. Keep BB-8 worlds, assets,
  control, RL environments/trainers, policies, tests and documentation here.
- Treat Genesis Studio as an external dependency. Use its existing project model,
  scene builder and native editor. Do not move this project's code into Studio or
  patch Studio as an implicit part of BB-8 work.
- Genesis World owns physics and rendering. Keep this project's actuator calls in
  `src/bb8_rl/control/genesis_backend.py`. Use body pose/velocity setters only for
  initialization/reset; the independent non-colliding head is visual state.
- Importing modules and running simulation must not connect to hardware. Future
  hardware adapters start read-only and require explicit command authority.
- Label synthetic/estimated/measured parameters honestly. Do not claim physical
  compatibility or trained-policy performance from simulation-only tests.
- Use `scripts/python.sh` for the selected runtime. Run non-GUI tests for code
  changes; run real Genesis tests for dynamics/scene/adapter changes and native
  tests for viewer integration. Keep generated evidence in ignored `work/`.
- Update README and `docs/M0_M1.md` when behavior or setup changes. Do not commit
  environments, datasets, generated training runs, credentials or third-party dependencies.
