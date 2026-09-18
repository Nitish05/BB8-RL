"""Capture fixed supplemental RGB views with one moving synthetic scan camera.

No depth, segmentation or scene geometry is exported as mapping evidence. The
existing original twenty mapping RGB/calibration pairs are copied unchanged.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def capture_plan(stage="initial"):
    positions = [[x, y, 3.2] for y in (-1., 0., 1.) for x in (-1., 0., 1.)]
    positions += [[-1.5, -1.5, 4.], [1.5, -1.5, 4.], [0., 1.5, 4.]]
    plan = {
        'schema': 'bb8.supplemental-scan-plan.v1',
        'scope': 'Uniform room coverage; camera positions fixed before captures and independent geometry scoring.',
        'seed': 78100, 'layout_seed': 11178, 'parked_robot_xy': [-1.75, -1.75],
        'excluded_query_ids': [14, 17, 20, 23],
        'resolution_wh': [1280, 960],
        'views': [{'id': f'scan-{i:03}', 'index': i, 'position': p,
                   'lookat': [p[0], p[1] + .05, 0.]} for i, p in enumerate(positions, 24)],
        'depth_used': False, 'segmentation_used': False,
        'camera_moving_robot_stationary': True,
    }
    if stage == "offset":
        positions = [[x, y, 3.] for y in (-1.5, -.5, .5, 1.5)
                     for x in (-1.5, -.5, .5, 1.5)]
        plan.update(
            schema='bb8.supplemental-scan-plan.v2',
            scope='Global half-offset grid improves overlap for four-view full-prism evidence; positions fixed before capture and scoring.',
            views=[{'id': f'scan-{i:03}', 'index': i, 'position': p,
                    'lookat': [p[0], p[1] + .05, 0.]} for i, p in enumerate(positions, 36)],
            retained_view_ids=[f'scan-{i:03}' for i in range(36)
                               if i not in plan['excluded_query_ids']],
            prior_capture_manifest_sha256='845da8774aa953f97fdcfae9c4265b901e5c751b1da6f3524b7e7bed1651c52f',
        )
    elif stage != "initial":
        raise ValueError('Unknown capture stage')
    return plan


def main(args):
    import cv2
    import torch

    from bb8_rl.camera_rig import calibration_from_live_camera
    from bb8_rl.env import NavigationEnv

    plan = json.loads(args.plan.read_text())
    if plan != capture_plan(args.stage):
        raise ValueError('Capture plan differs from the fixed camera-only scan design')
    previous = None
    if args.stage == "offset":
        previous_path = args.original_scan / 'capture-manifest.json'
        if digest(previous_path) != plan['prior_capture_manifest_sha256']:
            raise ValueError('Prior scan differs from frozen input')
        previous = json.loads(previous_path.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.plan, args.output / 'capture-plan.json')
    (args.output / 'source').mkdir()
    shutil.copy2(__file__, args.output / 'source' / Path(__file__).name)
    records = {}
    retained = plan.get('retained_view_ids', [f'scan-{i:03}' for i in range(24)
                                             if i not in plan['excluded_query_ids']])
    for name in retained:
        for suffix in ('.png', '.json'):
            shutil.copy2(args.original_scan / (name + suffix), args.output / (name + suffix))
        records[name] = {'rgb': digest(args.output / (name + '.png')),
                         'calibration': digest(args.output / (name + '.json'))}
        if previous and records[name] != previous['input_sha256'][name]:
            raise ValueError('An existing RGB/calibration input changed')
    torch.set_num_threads(2)
    manifest = {'schema': 'bb8.supplemental-scan-capture.v1', 'status': 'running',
                'plan_sha256': digest(args.plan), 'source_sha256': digest(__file__),
                'task_sha256': digest(args.task), 'input_sha256': records,
                'additional_view_ids': [v['id'] for v in plan['views']],
                'depth_used': False, 'segmentation_used': False}
    def save():
        (args.output / 'capture-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    save()
    with NavigationEnv(args.task, split='validation', render_mode='rgb_array') as env:
        env.reset(seed=plan['seed'], options={'layout_seed': plan['layout_seed'],
                   'start': plan['parked_robot_xy'], 'goal': [-1.4, -1.75]})
        env.world.camera_period = env.world._next_frame = 1e9
        camera = env.world.camera
        for view in plan['views']:
            camera.set_pose(pos=view['position'], lookat=view['lookat'])
            rgb = camera.render(rgb=True, force_render=True)[0].copy()
            c = calibration_from_live_camera(camera, 2)
            if list(c.resolution) != plan['resolution_wh']:
                raise ValueError('Capture resolution differs from the original scan')
            meta = {**view, 'intrinsics': c.intrinsics.tolist(),
                    'world_to_camera': c.world_to_camera.tolist(),
                    'calibration': 'synthetic_exact', 'depth_used': False,
                    'segmentation_used': False, 'camera_moving_robot_stationary': True}
            name = view['id']
            cv2.imwrite(str(args.output / (name + '.png')), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            (args.output / (name + '.json')).write_text(json.dumps(meta, indent=2)+'\n')
            records[name] = {'rgb': digest(args.output / (name + '.png')),
                             'calibration': digest(args.output / (name + '.json'))}
            save()
            print(json.dumps({'captured': name, **records[name]}), flush=True)
    manifest['status'] = 'complete'
    manifest['mapping_input_view_ids'] = list(records)
    save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--stage', choices=('initial', 'offset'), default='initial')
    parser.add_argument('--original-scan', type=Path, required=True)
    parser.add_argument('--task', type=Path, default=ROOT / 'work/interactive-assets/scene/task.yaml')
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
