import copy
import json
from pathlib import Path

import numpy as np
import robosuite as suite
from PIL import Image
from robosuite.controllers import load_composite_controller_config


ROOT = Path(__file__).resolve().parents[1]
PROJECT = json.loads((ROOT / "configs/project.json").read_text())
TASKS = json.loads((ROOT / "configs/tasks.json").read_text())
SEEDS = json.loads((ROOT / "configs/controller_test.json").read_text())["seeds"]
PREVIEW_DIR = ROOT / "reports/controller_preview"


def save_preview(task_name, seed, camera, rgb, depth):
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    rgb = np.flipud(np.asarray(rgb))
    depth = np.flipud(np.asarray(depth)).astype(np.float32)

    stem = f"{task_name.lower()}_seed{seed}_{camera}"
    Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(PREVIEW_DIR / f"{stem}_rgb.png")
    np.save(PREVIEW_DIR / f"{stem}_depth.npy", depth)

    valid = depth[np.isfinite(depth)]
    low, high = np.percentile(valid, [1, 99]) if valid.size else (0.0, 1.0)
    if high <= low:
        high = low + 1.0
    depth_view = np.clip((depth - low) / (high - low), 0.0, 1.0)
    Image.fromarray((depth_view * 255).astype(np.uint8), mode="L").save(
        PREVIEW_DIR / f"{stem}_depth.png"
    )


def test_task(task_spec, seed):
    task_name = task_spec["name"]
    native_action_dim = task_spec.get("robosuite_action_dim", PROJECT["controller"]["action_dim"])
    dataset_action_dim = task_spec.get("dataset_action_dim", PROJECT["controller"]["action_dim"])
    controller = load_composite_controller_config(robot=PROJECT["robot"])
    env = suite.make(
        env_name=task_name,
        robots=PROJECT["robot"],
        controller_configs=copy.deepcopy(controller),
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        camera_names=PROJECT["capture"]["cameras"],
        camera_widths=PROJECT["capture"]["image_width"],
        camera_heights=PROJECT["capture"]["image_height"],
        camera_depths=True,
    )
    try:
        seed_fn = getattr(env, "seed", None)
        if callable(seed_fn):
            seed_fn(seed)
        env.reset()
        action = np.zeros(env.action_dim, dtype=np.float32)
        if env.action_dim != native_action_dim:
            raise ValueError(f"action_dim={env.action_dim}, expected {native_action_dim}")
        rendered = {}
        for camera in PROJECT["capture"]["cameras"]:
            rgb, depth = env.sim.render(
                camera_name=camera,
                width=PROJECT["capture"]["image_width"],
                height=PROJECT["capture"]["image_height"],
                depth=True,
            )
            if seed == SEEDS[0]:
                save_preview(task_name, seed, camera, rgb, depth)
            rendered[camera] = {
                "rgb": list(rgb.shape),
                "depth": list(depth.shape),
            }
        env.step(action)
        return {
            "status": "ok",
            "robosuite_action_dim": env.action_dim,
            "dataset_action_dim": dataset_action_dim,
            "rendered": rendered,
        }
    finally:
        env.close()


def main():
    report = []
    for task in TASKS:
        for seed in SEEDS:
            record = {"task": task["name"], "seed": seed}
            try:
                record.update(test_task(task, seed))
            except Exception as exc:
                record.update({"status": "failed", "error": repr(exc)})
            report.append(record)
            print(record)

    output = ROOT / "reports/controller_smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    failed = [item for item in report if item["status"] != "ok"]
    print(f"Report: {output}")
    print(f"Previews: {PREVIEW_DIR}")
    print(f"Passed: {len(report) - len(failed)} / {len(report)}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
