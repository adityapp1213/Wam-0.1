import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import robosuite as suite
from PIL import Image
from robosuite.controllers import load_composite_controller_config


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wam_dataset.controllers.pick_place import (
    PickPlaceReferenceController,
    PickPlaceSequenceReferenceController,
)
from wam_dataset.controllers.simple_tasks import (
    LiftReferenceController,
    NutAssemblySequenceReferenceController,
    NutAssemblyReferenceController,
    StackReferenceController,
)
from wam_dataset.controllers.task_specific import (
    DoorReferenceController,
    ToolHangReferenceController,
    WipeReferenceController,
)
from wam_dataset.variation import apply_camera_variation, apply_task_variation, make_variation
from wam_dataset.state import ROBOT_STATE_DIM, ROBOT_STATE_LAYOUT, read_panda_state


def save_view(path, rgb, depth):
    path.mkdir(parents=True, exist_ok=True)
    rgb = np.flipud(np.asarray(rgb)).astype(np.uint8)
    depth = np.flipud(np.asarray(depth)).astype(np.float32)
    Image.fromarray(rgb, mode="RGB").save(path / "rgb.png")
    np.save(path / "depth.npy", depth)


def capture(env, cameras, root, step, prefix):
    views = {}
    for camera in cameras:
        rgb, depth = env.sim.render(camera_name=camera, width=128, height=128, depth=True)
        view_dir = root / prefix / camera / f"{step:06d}"
        save_view(view_dir, rgb, depth)
        views[camera] = {"rgb_shape": [128, 128, 3], "depth_shape": [128, 128]}
    return views


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="PickPlaceBread")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=1400)
    parser.add_argument("--no-camera-variation", action="store_true")
    args = parser.parse_args()

    cameras = ["agentview", "frontview", "birdview"]
    controller_cfg = load_composite_controller_config(robot="Panda")
    env_kwargs = dict(
        env_name=args.task,
        robots="Panda",
        controller_configs=copy.deepcopy(controller_cfg),
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        camera_names=cameras,
        camera_widths=128,
        camera_heights=128,
        camera_depths=True,
        seed=args.seed,
        horizon=max(args.max_steps, 1000),
    )
    env = suite.make(**env_kwargs)
    variation = make_variation(args.seed, enabled=not args.no_camera_variation)

    temp_root = ROOT / "data/controller_test/reference_pending" / args.task / f"trajectory_seed{args.seed:04d}"
    temp_root.mkdir(parents=True, exist_ok=True)
    actions = []
    input_states = []
    observation_states = []
    frames = []

    try:
        env.reset()
        apply_task_variation(env, args.task, variation)
        apply_camera_variation(env, cameras, variation)
        if args.task == "PickPlace":
            policy = PickPlaceSequenceReferenceController(args.task, max_steps=args.max_steps)
        elif args.task.startswith("PickPlace"):
            policy = PickPlaceReferenceController(args.task, max_steps=args.max_steps)
        elif args.task == "Lift":
            policy = LiftReferenceController(args.task, max_steps=args.max_steps)
        elif args.task == "Stack":
            policy = StackReferenceController(args.task, max_steps=args.max_steps)
        elif args.task == "NutAssembly":
            policy = NutAssemblySequenceReferenceController(args.task, max_steps=args.max_steps)
        elif args.task.startswith("NutAssembly"):
            policy = NutAssemblyReferenceController(args.task, max_steps=args.max_steps)
        elif args.task == "Door":
            policy = DoorReferenceController(args.task, max_steps=args.max_steps)
        elif args.task == "ToolHang":
            policy = ToolHangReferenceController(args.task, max_steps=args.max_steps)
        elif args.task == "Wipe":
            policy = WipeReferenceController(args.task, max_steps=args.max_steps)
        else:
            raise ValueError(f"no task-specific reference controller registered for {args.task}")
        policy.reset(env, args.seed)

        for step in range(args.max_steps):
            input_states.append(read_panda_state(env).copy())
            frames.append(capture(env, cameras, temp_root, step, "input"))
            action = np.asarray(policy.act(env), dtype=np.float32)
            if action.shape != (env.action_dim,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1.0):
                raise ValueError(f"reference controller emitted an invalid {env.action_dim}D action")
            dataset_action = action if env.action_dim == 7 else np.pad(action, (0, 7 - env.action_dim))
            actions.append(dataset_action.copy())
            step_result = env.step(action)
            policy.update(env)
            observation_states.append(read_panda_state(env).copy())
            frames.append(capture(env, cameras, temp_root, step, "observation"))
            if bool(step_result[2]) and not policy.done(env):
                policy.fail("RoboSuite terminated before the official success condition")
            if policy.done(env):
                break

        result = policy.result()
        metadata = {
            "schema": "wam-0.1-level-0-reference-v1",
            "task": args.task,
            "seed": args.seed,
            "controller": type(policy).__name__,
            "action_dim": 7,
            "robosuite_action_dim": int(env.action_dim),
            "action_layout": ["dx", "dy", "dz", "d_rx_axis_angle", "d_ry_axis_angle", "d_rz_axis_angle", "gripper"],
            "cameras": cameras,
            "transition_count": len(actions),
            "robot_state_dim": ROBOT_STATE_DIM,
            "robot_state_layout": list(ROBOT_STATE_LAYOUT),
            "robot_state_schema": "wam-0.1-panda-state-v1",
            "valid": bool(result.success),
            "validation": result.as_dict(),
            "variation": variation.as_dict(),
        }
        np.save(temp_root / "action.npy", np.asarray(actions, dtype=np.float32))
        np.save(temp_root / "input_state.npy", np.asarray(input_states))
        np.save(temp_root / "observation_state.npy", np.asarray(observation_states))
        (temp_root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

        if result.success:
            final_root = ROOT / "data/controller_test/reference" / args.task / f"trajectory_seed{args.seed:04d}"
            final_root.parent.mkdir(parents=True, exist_ok=True)
            if final_root.exists():
                raise FileExistsError(f"refusing to overwrite {final_root}")
            temp_root.rename(final_root)
            print(json.dumps({"status": "valid", "path": str(final_root), **result.as_dict()}, indent=2))
            return 0

        rejected_root = ROOT / "data/controller_test/rejected" / args.task / f"trajectory_seed{args.seed:04d}"
        rejected_root.parent.mkdir(parents=True, exist_ok=True)
        if rejected_root.exists():
            raise FileExistsError(f"refusing to overwrite {rejected_root}")
        temp_root.rename(rejected_root)
        print(json.dumps({"status": "rejected", "path": str(rejected_root), **result.as_dict()}, indent=2))
        return 2
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
