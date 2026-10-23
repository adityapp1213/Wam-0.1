import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import robosuite as suite
from PIL import Image, ImageDraw
from robosuite.controllers import load_composite_controller_config

ROOT = Path('/home/aditya/robotics/wam-0.1')
OUT = ROOT / 'reports' / 'final_10_task_gate'
TASKS = [item['name'] for item in json.loads((ROOT / 'configs' / 'tasks.json').read_text())]
CAMERAS = ['agentview', 'frontview', 'birdview']
SIZE = 128
EXAMPLES_PER_MODE = 3
MAX_ATTEMPTS = 20

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from scripts.test_controller_matrix import make_policy
from wam_dataset.variation import apply_camera_variation, apply_task_variation, make_variation


def max_steps_for(task):
    return 500 if task == 'Wipe' else 1400


def capture(env):
    frames = {}
    for camera in CAMERAS:
        rgb, depth = env.sim.render(camera_name=camera, width=SIZE, height=SIZE, depth=True)
        frames[camera] = {
            'rgb': np.flipud(np.asarray(rgb)).astype(np.uint8),
            'depth': np.flipud(np.asarray(depth)).astype(np.float32),
        }
    return frames


def image_metrics(start, end):
    values = []
    for camera in CAMERAS:
        start_rgb = start[camera]['rgb'].astype(np.int16)
        end_rgb = end[camera]['rgb'].astype(np.int16)
        delta = np.abs(end_rgb - start_rgb)
        values.append({
            'camera': camera,
            'mean_abs_delta': float(delta.mean()),
            'changed_fraction': float(np.mean(np.max(delta, axis=2) > 2)),
            'depth_mean_abs_delta': float(np.abs(end[camera]['depth'] - start[camera]['depth']).mean()),
        })
    return values


def save_preview(path, task, mode, example, seed, start, end):
    canvas = Image.new('RGB', (SIZE * len(CAMERAS), SIZE * 3 + 38), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 4), f'{task} | {mode} | example {example} | seed {seed}', fill='black')
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


def run_episode(task, seed, randomized, controller_cfg):
    max_steps = max_steps_for(task)
    random.seed(seed)
    np.random.seed(seed)
    env = suite.make(
        env_name=task,
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
        horizon=max(max_steps, 1000),
    )
    try:
        seed_fn = getattr(env, 'seed', None)
        if callable(seed_fn):
            seed_fn(seed)
        env.reset()
        variation = make_variation(seed, enabled=randomized)
        apply_task_variation(env, task, variation)
        apply_camera_variation(env, CAMERAS, variation)
        start = capture(env)
        policy = make_policy(task, max_steps)
        policy.reset(env, seed)
        step = -1
        for step in range(max_steps):
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
        metrics = image_metrics(start, end)
        depth_valid = all(np.isfinite(frame['depth']).all() for frames in (start, end) for frame in frames.values())
        image_changed = max(item['changed_fraction'] for item in metrics) > 0.002
        passed = bool(result.get('success') and depth_valid and image_changed)
        return {
            'passed': passed,
            'task': task,
            'mode': 'random' if randomized else 'normal',
            'seed': seed,
            'steps': step + 1,
            'result': result,
            'variation': variation.as_dict(),
            'depth_valid': depth_valid,
            'image_metrics': metrics,
            'start': start,
            'end': end,
        }
    except Exception as exc:
        return {
            'passed': False,
            'task': task,
            'mode': 'random' if randomized else 'normal',
            'seed': seed,
            'error': f'{type(exc).__name__}: {exc}',
        }
    finally:
        env.close()


def make_sheet(paths, out):
    thumb_w = 384
    thumb_h = 422
    cols = 5
    rows = (len(paths) + cols - 1) // cols
    sheet = Image.new('RGB', (cols * thumb_w, rows * thumb_h), 'white')
    for index, path in enumerate(paths):
        image = Image.open(path).convert('RGB')
        image.thumbnail((thumb_w - 8, thumb_h - 8))
        x = (index % cols) * thumb_w + (thumb_w - image.width) // 2
        y = (index // cols) * thumb_h + (thumb_h - image.height) // 2
        sheet.paste(image, (x, y))
    sheet.save(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    controller_cfg = load_composite_controller_config(robot='Panda')
    rows = []
    preview_paths = {'normal': [], 'random': []}
    for mode, randomized, base_seed in (('normal', False, 60000), ('random', True, 61000)):
        for task_index, task in enumerate(TASKS):
            accepted = 0
            for attempt in range(MAX_ATTEMPTS):
                seed = base_seed + task_index * 100 + attempt
                item = run_episode(task, seed, randomized, controller_cfg)
                if not item['passed']:
                    rows.append({key: value for key, value in item.items() if key not in ('start', 'end')})
                    print(f'REJECT {mode:6s} {task:20s} seed={seed} reason={item.get("result", {}).get("failure_reason", item.get("error", "quality"))}', flush=True)
                    continue
                accepted += 1
                path = OUT / mode / f'{task}_example{accepted}_seed{seed}.png'
                save_preview(path, task, mode, accepted, seed, item['start'], item['end'])
                row = {key: value for key, value in item.items() if key not in ('start', 'end')}
                row['example'] = accepted
                row['preview_path'] = str(path)
                rows.append(row)
                preview_paths[mode].append(path)
                print(f'PASS    {mode:6s} {task:20s} example={accepted} seed={seed} steps={item["steps"]}', flush=True)
                if accepted == EXAMPLES_PER_MODE:
                    break
            if accepted != EXAMPLES_PER_MODE:
                raise RuntimeError(f'{task} did not produce {EXAMPLES_PER_MODE} valid {mode} examples')

    make_sheet(preview_paths['normal'], OUT / 'normal_contact_sheet.png')
    make_sheet(preview_paths['random'], OUT / 'random_contact_sheet.png')
    report = {
        'tasks': TASKS,
        'examples_per_task_per_mode': EXAMPLES_PER_MODE,
        'accepted': sum(bool(row.get('passed')) for row in rows),
        'rejected_attempts': sum(not bool(row.get('passed')) for row in rows),
        'rows': rows,
        'normal_contact_sheet': str(OUT / 'normal_contact_sheet.png'),
        'random_contact_sheet': str(OUT / 'random_contact_sheet.png'),
    }
    (OUT / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({
        'tasks': len(TASKS),
        'accepted': report['accepted'],
        'rejected_attempts': report['rejected_attempts'],
        'normal_contact_sheet': str(OUT / 'normal_contact_sheet.png'),
        'random_contact_sheet': str(OUT / 'random_contact_sheet.png'),
    }, indent=2))


if __name__ == '__main__':
    main()
