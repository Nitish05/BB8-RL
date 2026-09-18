"""Small learned RGB floor/head observer; simulator labels are never inputs."""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .camera import VisualMeasurement


def candidates(rgb):
    """Native-resolution color proposals; this stage still assumes a gray head."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = ((hsv[..., 1] < 60) & (hsv[..., 2] > 65)).astype(np.uint8)
    count, _, stats, centers = cv2.connectedComponentsWithStats(mask)
    return [centers[i] for i in range(1, count) if 3 <= stats[i, 4] <= 120]


def crop(rgb, center, size=64):
    origin = np.rint(center).astype(int) - size // 2
    padded = cv2.copyMakeBorder(rgb, size, size, size, size, cv2.BORDER_REFLECT_101)
    x, y = origin + size
    return padded[y : y + size, x : x + size].copy(), origin


def guided_floor_probability(rgb, probability, *, radius=4, epsilon=1e-4):
    """Align a native-resolution probability field to grayscale RGB edges.

    This standard local-linear guided filter uses no segmentation or geometry
    labels. Radius 4 and epsilon 1e-4 were selected on eight validation scenes;
    the classifier threshold and trained weights remain unchanged.
    """
    probability = np.asarray(probability, dtype=np.float32)
    if (
        rgb.dtype != np.uint8
        or rgb.ndim != 3
        or rgb.shape[-1] != 3
        or min(rgb.shape[:2]) < 1
        or probability.shape != rgb.shape[:2]
        or not np.isfinite(probability).all()
        or np.any((probability < 0) | (probability > 1))
        or not isinstance(radius, int)
        or isinstance(radius, bool)
        or radius < 1
        or not np.isfinite(epsilon)
        or epsilon <= 0
    ):
        raise ValueError("Invalid RGB, probability field or guided-filter parameters")
    guidance = cv2.cvtColor(rgb.astype(np.float32) / 255, cv2.COLOR_RGB2GRAY)
    size = (radius * 2 + 1,) * 2

    def box(value):
        return cv2.boxFilter(value, -1, size, borderType=cv2.BORDER_REFLECT_101)

    mean_i, mean_p = box(guidance), box(probability)
    covariance = box(guidance * probability) - mean_i * mean_p
    variance = box(guidance * guidance) - mean_i * mean_i
    a = covariance / (variance + epsilon)
    b = mean_p - a * mean_i
    refined = np.clip(box(a) * guidance + box(b), 0, 1)
    if not np.isfinite(refined).all():
        raise ValueError("Guided filter produced nonfinite probabilities")
    return refined


def block(a, b):
    return nn.Sequential(
        nn.Conv2d(a, b, 3, padding=1),
        nn.GroupNorm(4, b),
        nn.SiLU(),
        nn.Conv2d(b, b, 3, padding=1),
        nn.GroupNorm(4, b),
        nn.SiLU(),
    )


class Segmenter(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.enc1, self.enc2 = block(3, width), block(width, width * 2)
        self.enc3 = block(width * 2, width * 4)
        self.dec2, self.dec1 = block(width * 6, width * 2), block(width * 3, width)
        self.out = nn.Conv2d(width, 2, 1)

    def forward(self, x):
        a = self.enc1(x)
        b = self.enc2(F.avg_pool2d(a, 2))
        c = self.enc3(F.avg_pool2d(b, 2))
        b = self.dec2(
            torch.cat(
                (
                    b,
                    F.interpolate(
                        c, size=b.shape[-2:], mode="bilinear", align_corners=False
                    ),
                ),
                1,
            )
        )
        a = self.dec1(
            torch.cat(
                (
                    a,
                    F.interpolate(
                        b, size=a.shape[-2:], mode="bilinear", align_corners=False
                    ),
                ),
                1,
            )
        )
        return self.out(a)


class VisionStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.floor = Segmenter(8)
        self.head = Segmenter(8)


class LearnedObserver:
    """Learned masks with native-resolution RGB proposals and causal Kalman downstream.

    No SAM/TAP model is embedded here. These weights are supervised on synthetic
    render labels. The head proposal rule and geometric projection remain explicit.
    """

    def __init__(
        self,
        calibration,
        checkpoint,
        *,
        device="cpu",
        floor_threshold=None,
        floor_refinement="none",
    ):
        if floor_refinement not in ("none", "guided"):
            raise ValueError("Unknown floor refinement; use 'none' or 'guided'")
        self.floor_refinement = floor_refinement
        self.calibration, self.device = calibration, device
        state = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
        if state["format"] != "bb8-vision-v1":
            raise ValueError("Unsupported vision checkpoint")
        required_refinement = state.get("calibration_refinement")
        if required_refinement is not None and (
            required_refinement != "guided-radius4-epsilon1e-4"
            or floor_refinement != "guided"
        ):
            raise ValueError("Checkpoint requires its calibrated guided refinement")
        self.model = VisionStudent().to(device).eval()
        self.model.load_state_dict(state["model"])
        self.floor_target = state.get("floor_target", "floor")
        self.floor_threshold = (
            state["floor_threshold"] if floor_threshold is None else floor_threshold
        )
        if not 0 < self.floor_threshold < 1:
            raise ValueError("Floor threshold must be between zero and one")
        self.head_threshold = state["head_threshold"]

    def probabilities(self, images, head=False):
        tensor = (
            torch.as_tensor(
                np.stack(images).transpose(0, 3, 1, 2).copy(), device=self.device
            ).float()
            / 255
        )
        with torch.inference_mode():
            network = self.model.head if head else self.model.floor
            probability = network(tensor).softmax(1)[:, 1].cpu().numpy()
        if not np.isfinite(probability).all():
            raise ValueError("Vision network produced nonfinite probabilities")
        return probability

    def _validate_frame(self, rgb, timestamp):
        width, height = self.calibration.resolution
        if (
            rgb.shape != (height, width, 3)
            or rgb.dtype != np.uint8
            or not np.isfinite(timestamp)
            or timestamp < 0
        ):
            raise ValueError("Invalid RGB frame or capture timestamp")

    def observe(self, rgb, timestamp):
        self._validate_frame(rgb, timestamp)
        width, height = self.calibration.resolution
        small = cv2.resize(rgb, (320, 240), interpolation=cv2.INTER_AREA)
        probability = self.probabilities([small])[0]
        probability = cv2.resize(
            probability, (width, height), interpolation=cv2.INTER_LINEAR
        )
        if self.floor_refinement == "guided":
            probability = guided_floor_probability(rgb, probability)
        # The corrected checkpoint predicts visible ground plus visible self.
        # It never gets an oracle robot mask at inference time.
        floor = probability >= self.floor_threshold
        xy, pixel, status = self._head_position(rgb)
        robot = np.zeros((height, width), bool)
        if status != "visible":
            return VisualMeasurement(timestamp, None, None, floor, robot, status)
        # Self pixels only: the known body/head image neighborhood uses the same
        # neutral-color rule as the baseline. Never clear arbitrary hidden floor.
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        yy, xx = np.indices((height, width))
        robot = (
            ((xx - pixel[0]) ** 2 + (yy - pixel[1]) ** 2 < 14**2)
            & (hsv[..., 1] < 65)
            & (hsv[..., 2] > 15)
        )
        return VisualMeasurement(
            timestamp,
            xy,
            self.calibration.measurement_covariance(pixel),
            floor,
            robot,
            status,
        )

    def observe_position(self, rgb, timestamp):
        """Head-only localization with no fresh floor or robot-mask evidence.

        Uses exactly the same proposal, head network and projection as observe.
        Both native-shaped masks are immutable all-false views: callers cannot
        mistake an old floor mask for newly inferred traversable space. This
        skips floor inference/refinement and the full-frame robot neighborhood.
        """
        self._validate_frame(rgb, timestamp)
        xy, pixel, status = self._head_position(rgb)
        empty = np.broadcast_to(np.array(False), rgb.shape[:2])
        return VisualMeasurement(
            timestamp,
            xy,
            self.calibration.measurement_covariance(pixel)
            if pixel is not None
            else None,
            empty,
            empty,
            status,
        )

    def _head_position(self, rgb):
        proposals = candidates(rgb)
        accepted = []
        if len(proposals) > 64:
            return None, None, "ambiguous"
        if proposals:
            patches, origins = zip(*(crop(rgb, center) for center in proposals))
            heads = self.probabilities(patches, head=True)
            for probability, origin in zip(heads, origins):
                count, components, stats, _ = cv2.connectedComponentsWithStats(
                    (probability >= self.head_threshold).astype(np.uint8)
                )
                for index in range(1, count):
                    if not 3 <= stats[index, 4] <= 120:
                        continue
                    ys, xs = np.where(components == index)
                    weights = probability[ys, xs]
                    pixel = origin + [
                        np.average(xs, weights=weights),
                        np.average(ys, weights=weights),
                    ]
                    xy = self.calibration.to_plane(pixel, self.calibration.head_height)
                    if np.max(abs(xy)) < self.calibration.extent - 0.06 and not any(
                        np.linalg.norm(pixel - previous[1]) < 4 for previous in accepted
                    ):
                        accepted.append((xy, pixel))
        if len(accepted) != 1:
            return None, None, "missing" if not accepted else "ambiguous"
        xy, pixel = accepted[0]
        return xy, pixel, "visible"


class PositionOnlyObserver:
    """CameraView adapter that produces positions and no free-space evidence."""

    def __init__(self, observer):
        if not isinstance(observer, LearnedObserver):
            raise TypeError("Position-only adapter requires a LearnedObserver")
        self.observer = observer

    @property
    def calibration(self):
        return self.observer.calibration

    def observe(self, rgb, timestamp):
        return self.observer.observe_position(rgb, timestamp)
