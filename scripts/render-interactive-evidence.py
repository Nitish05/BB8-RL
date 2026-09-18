"""Create a labeled two-camera GIF from audited native RGB; no generated imagery."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    rows = [
        json.loads(line) for line in (args.run / "rows.jsonl").read_text().splitlines()
    ]
    demo = json.loads((args.assets / "demo.json").read_text())
    ids = list(rows[0]["images"])
    if len(ids) != 2:
        raise ValueError("This evidence layout expects exactly two camera views")
    points = np.array([row["controller"]["xy"] for row in rows])
    world = np.c_[points, np.full(len(points), 0.083)]
    crops = {}
    for name in ids:
        camera = demo["cameras"][name]
        transform, intrinsics = (
            np.array(camera["world_to_camera"]),
            np.array(camera["intrinsics"]),
        )
        projected = (world @ transform[:3, :3].T + transform[:3, 3]) @ intrinsics.T
        pixels = projected[:, :2] / projected[:, 2:]
        center = (pixels.min(0) + pixels.max(0)) / 2
        width, height = camera["resolution"]
        left, top = np.clip(
            center - [160, 150], [0, 0], [width - 320, height - 300]
        ).astype(int)
        crops[name] = [int(left), int(top), int(left + 320), int(top + 300)]
    font_path = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
    font = (
        ImageFont.truetype(str(font_path), 19)
        if font_path.exists()
        else ImageFont.load_default(size=19)
    )
    small = (
        ImageFont.truetype(str(font_path), 16)
        if font_path.exists()
        else ImageFont.load_default(size=16)
    )
    selected = list(range(0, len(rows), 2))
    if selected[-1] != len(rows) - 1:
        selected.append(len(rows) - 1)
    frames = []
    for index in selected:
        row = rows[index]
        frame = Image.new("RGB", (1000, 570), "#0c1724")
        draw = ImageDraw.Draw(frame)
        draw.text(
            (24, 18),
            "BB8-RL   /   Two-camera visual control",
            font=font,
            fill="#edf3f4",
        )
        draw.text(
            (24, 48),
            "Actual native RGB crops · selected synthetic development route",
            font=small,
            fill="#9cb4bd",
        )
        for column, name in enumerate(ids):
            artifact = row["images"][name]
            path = args.run / artifact["rgb_path"]
            if digest(path) != artifact["rgb_sha256"]:
                raise ValueError(f"Input image hash mismatch: {path}")
            image = (
                Image.open(path)
                .convert("RGB")
                .crop(crops[name])
                .resize((470, 440), Image.Resampling.LANCZOS)
            )
            x = 20 + column * 490
            frame.paste(image, (x, 82))
            status = "accepted RGB" if name in row["source_ids"] else "no accepted fix"
            draw.rectangle((x, 82, x + 470, 113), fill="#152739")
            draw.text(
                (x + 12, 89),
                f"CAMERA {name}  ·  {status}",
                font=small,
                fill="#6ee7d4" if name in row["source_ids"] else "#f7bd70",
            )
        draw.text(
            (24, 540),
            f"t = {row['time']:.2f} s    |    sources: {' + '.join(row['source_ids']) or 'prediction'}    |    {row['controller_status']}",
            font=small,
            fill="#edf3f4",
        )
        frames.append(frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        args.output, save_all=True, append_images=frames[1:], duration=100, loop=0
    )
    frames[len(frames) // 2].save(args.output.with_suffix(".png"))
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "source_rows_sha256": digest(args.run / "rows.jsonl"),
                "source_demo_sha256": digest(args.assets / "demo.json"),
                "crops": crops,
                "frames": len(frames),
                "gif_sha256": digest(args.output),
                "scope": "Synchronized native RGB crops and recorded observer labels; no scene synthesis or truth supplied to control.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
