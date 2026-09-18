# M2: arbitrary room and fixed simulated camera

The user chose synthetic room measurements, random obstacles and a camera with a
view of the room and BB8. This replaces the measured-room prerequisite for current
simulation development. Physical room/camera measurements are deferred until real
transfer is wanted; they do not block this project.

## Current scene

`projects/bb8/synthetic-room/` contains an ordinary Genesis Studio world,
`room.yaml` for physics, `task.yaml` for navigation, and `layout.json` with the seed,
obstacle dimensions, start/goal and a collision-free geometric route.

| Parameter | Synthetic value |
| --- | --- |
| Floor | 4 × 4 m square, X/Y in metres, Z up |
| Random layout | Seed 42, seven fixed boxes |
| Obstacle widths/depths | Sampled from 0.25–0.60 m |
| Obstacle heights | Sampled from 0.15–0.45 m |
| Initial BB8 position | (-1.3, -1.3) m |
| Example route goal | (1.3, 1.3) m |
| Camera position | (2.8, -3.4, 5.2) m, fixed above one corner |
| Camera looks at | Floor centre (0, 0, 0) |
| Camera image | 1280 × 960, 58° vertical field of view |
| Lens | Ideal pinhole, zero distortion |
| Body radius / effective mass | Existing synthetic 0.037 m / 0.2 kg |

The 15 cm perimeter walls present a cutaway floor boundary, so full-height walls
do not hide the room. There is no ceiling geometry. BB8 is an approximate shell
and independent head, not a detailed perception-training asset.

## Run and vary it

From BB8-RL:

```bash
# Inspect the actual fixed-camera RGB view and export exact simulated calibration.
./scripts/run-bb8.sh room preview --output work/my-room-camera

# Drive the controller in this room, using its authored obstacle dimensions.
./scripts/run-bb8.sh evaluate --task projects/bb8/synthetic-room/task.yaml --suite train --episodes 1 --controllers baseline --viewer --output work/my-room-demo

# Create a different random room, with its own camera and navigation task.
./scripts/run-bb8.sh room generate --side 4 --obstacles 7 --seed 123 --output work/room-123
./scripts/run-bb8.sh room preview --config work/room-123/room.yaml --output work/room-123-camera
./scripts/run-bb8.sh evaluate --task work/room-123/task.yaml --suite train --episodes 4 --controllers baseline --output work/room-123-navigation
```

Outputs use fresh directories to preserve evidence. The generator accepts square
sides from 3 to 6 m in 10 cm increments and zero to eight boxes. Obstacles stay
fixed within a room; a new seed produces a new room. Rejection sampling prevents
overlap and checks a footprint-inflated route between the initial endpoints.
The camera keeps the same relative perch as room size changes. Preview each new
room to check actual initial robot visibility; obstacles can occlude the camera.

Open `projects/bb8/synthetic-room/room.genesis.json` using Studio's Open Project
flow. The editor's camera may be moved independently; BB8's observation camera is
the fixed sensor named `navigation_camera`. Studio itself is unchanged.

## Camera checks and calibration

`room preview` uses real Genesis RGB and segmentation rendering. It checks that
all four floor corners project into the image, at least eight visible pixels
belong to BB8, and a floor-plane projection/inverse round trip is numerically
consistent. `camera-report.json` records those checks and source/world hashes.

`camera.png` is the actual view. `bb8-detail.png` is an enlarged crop from it.
`segmentation.npy` is simulator ground truth used only for verification.
`calibration.json` stores the engine's intrinsics, world-to-camera transform,
floor-to-pixel homography and inverse. Camera coordinates use X right, Y down,
Z forward; pixels use u right and v down. Calibration is exact synthetic geometry,
not a calibration fitted to physical measurements.

Room framing is not a claim that every floor pixel is visible. Obstacles hide
parts of the floor, and BB8 may disappear behind them when it moves. The floor
homography applies to floor pixels; applying it to a box top or the robot's head
would introduce parallax error. A later perception loop needs masks/occlusion
handling. A small robot being visible is not proof it can be reliably detected.

## Navigation integration

The new `layout_mode: authored` loads the exact obstacle positions and dimensions
from the room project. Resets preserve that geometry and reproducibly sample valid
starts/goals. The room task allows 60 simulated seconds per episode; the existing
3 m procedural M3 benchmark keeps its original configuration.

For an authored room, the train/validation/test episode seeds vary endpoints;
**they do not create held-out room geometry**. Generalization evaluation needs
separately generated room seeds, frozen before training. Four room smoke episodes
cannot replace the original final performance suite. Observations remain oracle
state/local maps; the rendered camera has not yet been connected to a vision model.

This synthetic setup is sufficient to continue with SAC + HER development.
Actual BB8 response fitting, physical camera calibration and hardware integration
remain future transfer work, rather than prerequisites for synthetic M2.
