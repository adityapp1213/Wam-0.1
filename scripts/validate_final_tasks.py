import argparse
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
sys.path.insert(0, str(ROOT / "src"))

from wam_dataset.variation import apply_camera_variation, make_variation
from scripts.test_controller_matrix import make_policy


TASKS = [
    "Lift",
    "Stack",
    "Door",
    "Wipe",
    "PickPlaceBread",
    "PickPlaceMilk",
    "NutAssemblySquare",
    "PickPlaceSingle",
    "PickPlaceCan",
    "PickPlaceCereal",
]
MOTION_TASKS = {
    "Lift",
    "Stack",
    "PickPlaceBread",
    "PickPlaceMilk",
    "NutAssemblySquare",
    "PickPlaceSingle",
    "PickPlaceCan",
    "PickPlaceCereal",
}
CAMERAS = ["agentview", "frontview", "birdview"]
SIZE = 128


def capture(env):
    frames = {}
    for camera in CAMERAS:
        rgb, depth = env.sim.render(camera_name=camera, width=SIZE, height=SIZE, depth=True)
        frames[camera] = {
            "rgb": np.flipud(np.asarray(rgb)).astype(np.uint8),
            "depth": np.flipud(np.asarray(depth)).astype(np.float32),
        }
    return frames


def image_metrics(start, end):
    metrics = {}
    for camera in CAMERAS:
        rgb0 = start[camera]["rgb"].astype(np.float32)
        rgb1 = end[camera]["rgb"].astype(np.float32)
        depth0 = start[camera]["depth"]
        depth1 = end[camera]["depth"]
        rgb_delta = np.abs(rgb1 - rgb0)
        depth_delta = np.abs(depth1 - depth0)
        metrics[camera] = {
            "rgb_mean_abs_delta": float(rgb_delta.mean()),
            "rgb_changed_fraction": float(np.mean(np.max(rgb_delta, axis=2) > 2.0)),
            "depth_mean_abs_delta": float(depth_delta.mean()),
            "rgb_valid": bool(np.isfinite(rgb0).all() and np.isfinite(rgb1).all()),
            "depth_valid": bool(np.isfinite(depth0).all() and np.isfinite(depth1).all()),
        }
    return metrics


def save_preview(path, task, mode, seed, start, end):
    canvas = Image.new("RGB", (SIZE * len(CAMERAS), SIZE * 2 + 38), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 4), f"{task} | {mode} | seed {seed}", fill="black")
    for col, camera in enumerate(CAMERAS):
        x = col * SIZE
        canvas.paste(Image.fromarray(start[camera]["rgb"]), (x, 22))
        canvas.paste(Image.fromarray(end[camera]["rgb"]), (x, 22 + SIZE))
        draw.text((x + 3, SIZE * 2 + 24), camera, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def motion_quality_ok(task, result):
    if not result.get("success", False):
        return False
    if task in MOTION_TASKS:
        return bool(result.get("grasp_seen") and result.get("lifted"))
    return True


def run_episode(task, seed, randomized, controller_cfg, preview_path=None):
    random.seed(seed)
    np.random.seed(seed)
    max_steps = 1400
    env = suite.make(
        env_name=task,
        robots="Panda",
        controller_configs=copy.deepcopy(controller_cfg),
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        camera_names=CAMERAS,
        camera_widths=SIZE,
        camera_heights=SIZE,
        camera_depths=True,
        seed=seed,
        horizon=max_steps,
    )
    try:
        seed_fn = getattr(env, "seed", None)
        if callable(seed_fn):
            seed_fn(seed)
        env.reset()
        variation = make_variation(seed, enabled=randomized)
        apply_camera_variation(env, CAMERAS, variation)
        start = capture(env)
        policy = make_policy(task, max_steps)
        policy.reset(env, seed)
        step = -1
        for step in range(max_steps):
            action = np.asarray(policy.act(env), dtype=np.float32)
            if action.shape != (env.action_dim,) or not np.isfinite(action).all():
                policy.fail("invalid controller action")
                break
            _, _, terminated, _ = env.step(action)
            policy.update(env)
            if terminated or policy.done(env):
                break
        end = capture(env)
        metrics = image_metrics(start, end)
        if preview_path is not None:
            save_preview(preview_path, task, "random" if randomized else "normal", seed, start, end)
        result = policy.result().as_dict()
        image_valid = all(
            value["rgb_valid"] and value["depth_valid"] for value in metrics.values()
        )
        image_changed = max(value["rgb_changed_fraction"] for value in metrics.values()) > 0.002
        motion_valid = motion_quality_ok(task, result)
        return {
            "task": task,
            "seed": seed,
            "mode": "random" if randomized else "normal",
            "steps": step + 1,
            "success": bool(result["success"]),
            "image_valid": image_valid,
            "image_changed": image_changed,
            "motion_valid": motion_valid,
            "passed": bool(motion_valid and image_valid and image_changed),
            "variation": variation.as_dict(),
            "image_metrics": metrics,
            "result": result,
        }
    except Exception as exc:
        return {
            "task": task,
            "seed": seed,
            "mode": "random" if randomized else "normal",
            "success": False,
            "image_valid": False,
            "image_changed": False,
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds-per-mode", type=int, default=10)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--normal-seed", type=int, default=18000)
    parser.add_argument("--random-seed", type=int, default=19000)
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "reports" / "final_tasks_gate.json",
    )
    args = parser.parse_args()

    controller_cfg = load_composite_controller_config(robot="Panda")
    preview_root = ROOT / "reports" / "recovered_task_previews"
    rows = []
    rejected_attempts = []
    total = len(TASKS) * args.seeds_per_mode * 2
    for randomized, base_seed in ((False, args.normal_seed), (True, args.random_seed)):
        mode = "random" if randomized else "normal"
        for task in TASKS:
            for offset in range(args.seeds_per_mode):
                requested_seed = base_seed + offset
                for attempt in range(args.max_attempts):
                    seed = requested_seed + attempt * 1000
                    preview = preview_root / mode / f"{task}_seed{requested_seed}.png" if offset == 0 else None
                    row = run_episode(task, seed, randomized, controller_cfg, preview)
                    row["requested_seed"] = requested_seed
                    row["attempt"] = attempt
                    if row["passed"] or attempt + 1 == args.max_attempts:
                        rows.append(row)
                        status = "PASS" if row["passed"] else "FAIL"
                        print(
                            f"[{len(rows):03d}/{total:03d}] {status:4s} {mode:6s} {task:20s} "
                            f"requested={requested_seed} seed={seed} attempt={attempt + 1} "
                            f"task={row['success']} motion={row.get('motion_valid', False)} "
                            f"rgbd={row['image_valid']} changed={row['image_changed']}",
                            flush=True,
                        )
                        break
                    rejected_attempts.append(row)
                    print(
                        f"REJECT {mode:6s} {task:20s} requested={requested_seed} "
                        f"seed={seed} attempt={attempt + 1} reason={row.get('result', {}).get('failure_reason')}",
                        flush=True,
                    )
                args.report.parent.mkdir(parents=True, exist_ok=True)
                summary = {
                    "tasks": TASKS,
                    "episodes": len(rows),
                    "target_episodes": total,
                    "attempted_episodes": len(rows) + len(rejected_attempts),
                    "rejected_attempts": len(rejected_attempts),
                    "passed": sum(item["passed"] for item in rows),
                    "failed": sum(not item["passed"] for item in rows),
                    "rows": rows,
                    "rejected": rejected_attempts,
                }
                args.report.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in ("episodes", "attempted_episodes", "rejected_attempts", "passed", "failed")}, indent=2))
    return 0 if summary["failed"] == 0 and summary["episodes"] == total else 2


if __name__ == "__main__":
    raise SystemExit(main())
