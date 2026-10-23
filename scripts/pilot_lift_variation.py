import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import robosuite as suite
from PIL import Image, ImageDraw
from robosuite.controllers import load_composite_controller_config

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from scripts.test_controller_matrix import make_policy
from wam_dataset.variation import apply_camera_variation, apply_task_variation, make_variation

OUT = ROOT / 'reports' / 'lift_variation_pilot'
CAMERAS = ['agentview', 'frontview', 'birdview']
SIZE = 128
MAX_STEPS = 1400
MIN_RANDOM_STATE_DISTANCE = 0.025


def capture(env):
    frames = {}
    for camera in CAMERAS:
        rgb, depth = env.sim.render(camera_name=camera, width=SIZE, height=SIZE, depth=True)
        frames[camera] = {
            'rgb': np.flipud(np.asarray(rgb)).astype(np.uint8),
            'depth': np.flipud(np.asarray(depth)).astype(np.float32),
        }
    return frames


def rgb_delta(start, end):
    values = []
    for camera in CAMERAS:
        delta = np.abs(end[camera]['rgb'].astype(np.int16) - start[camera]['rgb'].astype(np.int16))
        values.append({
            'mean_abs_delta': float(delta.mean()),
            'changed_fraction': float(np.mean(np.max(delta, axis=2) > 2)),
        })
    return values


def start_distance(left, right):
    values = []
    for camera in CAMERAS:
        delta = np.abs(left[camera]['rgb'].astype(np.int16) - right[camera]['rgb'].astype(np.int16))
        values.append(float(delta.mean()))
    return float(np.mean(values))


def save_preview(path, mode, slot, seed, start, end):
    canvas = Image.new('RGB', (SIZE * len(CAMERAS), SIZE * 3 + 38), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 4), f'Lift | {mode} | slot {slot} | seed {seed}', fill='black')
    for index, camera in enumerate(CAMERAS):
        x = index * SIZE
        start_rgb = start[camera]['rgb']
        end_rgb = end[camera]['rgb']
        diff = np.abs(end_rgb.astype(np.int16) - start_rgb.astype(np.int16)).mean(axis=2)
        scale = max(float(np.percentile(diff, 99.0)), 1.0)
        diff = np.sqrt(np.clip(diff / scale, 0.0, 1.0)) * 255.0
        canvas.paste(Image.fromarray(start_rgb), (x, 22))
        canvas.paste(Image.fromarray(end_rgb), (x, 22 + SIZE))
        canvas.paste(Image.fromarray(diff.astype(np.uint8)).convert('RGB'), (x, 22 + 2 * SIZE))
        draw.text((x + 3, SIZE * 3 + 24), camera, fill='black')
    draw.text((SIZE * 3 - 120, 22), 'start', fill='yellow', stroke_width=2, stroke_fill='black')
    draw.text((SIZE * 3 - 120, 22 + SIZE), 'end', fill='yellow', stroke_width=2, stroke_fill='black')
    draw.text((SIZE * 3 - 120, 22 + 2 * SIZE), 'RGB diff', fill='yellow', stroke_width=2, stroke_fill='black')
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def run_episode(seed, randomized, controller_cfg):
    random.seed(seed)
    np.random.seed(seed)
    env = suite.make(
        env_name='Lift',
        robots='Panda',
        controller_configs=copy.deepcopy(controller_cfg),
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        camera_names=CAMERAS,
        camera_widths=SIZE,
        camera_heights=SIZE,
        camera_depths=True,
        seed=seed,
        horizon=MAX_STEPS,
    )
    try:
        seed_fn = getattr(env, 'seed', None)
        if callable(seed_fn):
            seed_fn(seed)
        env.reset()
        variation = make_variation(seed, enabled=randomized)
        apply_task_variation(env, 'Lift', variation)
        apply_camera_variation(env, CAMERAS, variation)
        start = capture(env)
        start_pos = np.asarray(env.sim.data.body_xpos[env.cube_body_id], dtype=np.float32).copy()
        policy = make_policy('Lift', MAX_STEPS)
        policy.reset(env, seed)
        step = -1
        for step in range(MAX_STEPS):
            action = np.asarray(policy.act(env), dtype=np.float32)
            if action.shape != (env.action_dim,) or not np.isfinite(action).all():
                policy.fail('invalid controller action')
                break
            _, _, terminated, _ = env.step(action)
            policy.update(env)
            if terminated or policy.done(env):
                break
        end = capture(env)
        result = policy.result().as_dict()
        metrics = rgb_delta(start, end)
        depth_valid = all(np.isfinite(frame['depth']).all() for frames in (start, end) for frame in frames.values())
        passed = bool(
            result.get('success')
            and result.get('grasp_seen')
            and result.get('lifted')
            and depth_valid
            and max(item['changed_fraction'] for item in metrics) > 0.002
        )
        return {
            'passed': passed,
            'seed': seed,
            'mode': 'random' if randomized else 'normal',
            'steps': step + 1,
            'start_pos': start_pos.tolist(),
            'variation': variation.as_dict(),
            'result': result,
            'depth_valid': depth_valid,
            'image_metrics': metrics,
            'start': start,
            'end': end,
        }
    except Exception as exc:
        return {
            'passed': False,
            'seed': seed,
            'mode': 'random' if randomized else 'normal',
            'error': f'{type(exc).__name__}: {exc}',
        }
    finally:
        env.close()


def collect_mode(mode, base_seed, randomized, controller_cfg):
    accepted = []
    rejected = []
    for slot in range(5):
        for attempt in range(10):
            seed = base_seed + slot + attempt * 1000
            item = run_episode(seed, randomized, controller_cfg)
            if not item['passed']:
                rejected.append({'slot': slot + 1, 'attempt': attempt + 1, **{key: value for key, value in item.items() if key not in ('start', 'end')}})
                print(f'REJECT {mode:6s} slot={slot + 1} seed={seed} reason={item.get("result", {}).get("failure_reason", item.get("error", "quality"))}', flush=True)
                continue
            if randomized and any(
                np.linalg.norm(np.asarray(item['start_pos']) - np.asarray(previous['start_pos'])) < MIN_RANDOM_STATE_DISTANCE
                for previous in accepted
            ):
                rejected.append({'slot': slot + 1, 'attempt': attempt + 1, 'seed': seed, 'reason': 'near-duplicate object start'})
                print(f'REJECT {mode:6s} slot={slot + 1} seed={seed} reason=near-duplicate object start', flush=True)
                continue
            item['slot'] = slot + 1
            item['attempt'] = attempt + 1
            item['preview_path'] = str(OUT / mode / f'Lift_{mode}_slot{slot + 1}_seed{seed}.png')
            save_preview(Path(item['preview_path']), mode, slot + 1, seed, item['start'], item['end'])
            accepted.append(item)
            print(f'PASS    {mode:6s} slot={slot + 1} seed={seed} start_pos={item["start_pos"]}', flush=True)
            break
        else:
            raise RuntimeError(f'could not collect accepted {mode} slot {slot + 1}')
    return accepted, rejected


def strip_frames(items):
    output = []
    for item in items:
        output.append({key: value for key, value in item.items() if key not in ('start', 'end')})
    return output


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    controller_cfg = load_composite_controller_config(robot='Panda')
    normal, normal_rejected = collect_mode('normal', 24000, False, controller_cfg)
    randomized, random_rejected = collect_mode('random', 25000, True, controller_cfg)
    report = {
        'task': 'Lift',
        'normal_target': 5,
        'random_target': 5,
        'normal': strip_frames(normal),
        'random': strip_frames(randomized),
        'rejected': normal_rejected + random_rejected,
        'accepted': len(normal) + len(randomized),
    }
    (OUT / 'pilot_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({
        'accepted': report['accepted'],
        'normal': len(normal),
        'random': len(randomized),
        'rejected': len(report['rejected']),
        'report': str(OUT / 'pilot_report.json'),
    }, indent=2))


if __name__ == '__main__':
    main()
