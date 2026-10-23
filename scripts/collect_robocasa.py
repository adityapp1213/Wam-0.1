"""Collect a small RoboCasa pilot in the WAM-0.1 two-level format."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image

import robocasa  # noqa: F401 - registers RoboCasa Gym tasks

from wam_dataset.state import ROBOT_STATE_DIM, ROBOT_STATE_LAYOUT, read_panda_state


CAMERAS = (
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
)
ACTION_LAYOUT = (
    "dx",
    "dy",
    "dz",
    "d_rx_axis_angle",
    "d_ry_axis_angle",
    "d_rz_axis_angle",
    "gripper",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="PickPlaceCounterToCabinet")
    parser.add_argument("--split", default="pretrain", choices=("pretrain", "target"))
    parser.add_argument("--mode", default="normal", choices=("normal", "random"))
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--chunk-length", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


def make_wrapper_action(action: np.ndarray) -> dict[str, np.ndarray]:
    """Map WAM's 7D arm action into RoboCasa's action dictionary."""
    action = np.asarray(action, dtype=np.float32)
    return {
        "action.gripper_close": np.array([1.0 if action[6] > 0 else 0.0], dtype=np.float32),
        "action.end_effector_position": action[:3],
        "action.end_effector_rotation": action[3:6],
        "action.base_motion": np.zeros(4, dtype=np.float32),
        "action.control_mode": np.array([0.0], dtype=np.float32),
    }


def make_action_schedule(seed: int, steps: int) -> np.ndarray:
    """Create repeatable small arm motions; each action is held for one sim step."""
    rng = np.random.default_rng(seed)
    axes = rng.permutation(3)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=3)
    schedule = []
    for step in range(steps):
        phase = (step // 8) % 4
        action = np.zeros(7, dtype=np.float32)
        if phase < 3:
            action[axes[phase]] = 0.35 * signs[phase]
        else:
            action[5] = 0.25 * signs[0]
        action[6] = 1.0 if (step // 8) % 2 else -1.0
        schedule.append(action)
    return np.asarray(schedule, dtype=np.float32)


def read_state(raw_env) -> np.ndarray:
    return read_panda_state(raw_env)


def capture_frame(raw_env) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    frames = {}
    for camera in CAMERAS:
        rgb, depth = raw_env.sim.render(
            camera_name=camera,
            width=128,
            height=128,
            depth=True,
        )
        # MuJoCo returns OpenGL images bottom-to-top; save them upright.
        frames[camera] = (
            np.flipud(np.asarray(rgb, dtype=np.uint8)).copy(),
            np.flipud(np.asarray(depth, dtype=np.float32)).copy(),
        )
    return frames


def save_canonical_frame(frame_root: Path, frame_id: int, frames) -> None:
    for camera, (rgb, depth) in frames.items():
        camera_root = frame_root / camera / f"{frame_id:06d}"
        camera_root.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(camera_root / "rgb.png")
        np.save(camera_root / "depth.npy", depth)


def link_frame(frame_root: Path, view_root: Path, source_frame_id: int, view_id: int) -> None:
    for camera in CAMERAS:
        source = frame_root / camera / f"{source_frame_id:06d}"
        target = view_root / camera / f"{view_id:06d}"
        target.mkdir(parents=True, exist_ok=True)
        for filename in ("rgb.png", "depth.npy"):
            os.link(source / filename, target / filename)


def save_level_0(trajectory: Path, frames, actions, states) -> None:
    frame_root = trajectory / "frames"
    for frame_id, frame in enumerate(frames):
        save_canonical_frame(frame_root, frame_id, frame)

    level_0 = trajectory / "level_0"
    input_states = np.asarray(states[:-1], dtype=np.float32)
    observation_states = np.asarray(states[1:], dtype=np.float32)
    np.save(level_0 / "action.npy", np.asarray(actions, dtype=np.float32))
    np.save(level_0 / "input_state.npy", input_states)
    np.save(level_0 / "observation_state.npy", observation_states)
    for transition_id in range(len(actions)):
        link_frame(frame_root, level_0 / "input", transition_id, transition_id)
        link_frame(frame_root, level_0 / "observation", transition_id + 1, transition_id)


def save_level_1(trajectory: Path, actions, states, chunk_length: int) -> int:
    frame_root = trajectory / "frames"
    level_1 = trajectory / "level_1"
    chunk_count = 0
    for start in range(0, len(actions) - chunk_length + 1, chunk_length):
        end = start + chunk_length
        segment = level_1 / f"segment_{chunk_count:06d}"
        segment.mkdir(parents=True, exist_ok=True)
        link_frame(frame_root, segment / "input", start, 0)
        link_frame(frame_root, segment / "subgoal", end, 0)
        np.save(segment / "action_block.npy", np.asarray(actions[start:end], dtype=np.float32))
        np.save(segment / "start_state.npy", np.asarray(states[start], dtype=np.float32))
        np.save(segment / "subgoal_state.npy", np.asarray(states[end], dtype=np.float32))
        with (segment / "metadata.json").open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "schema": "wam-0.1-level-1-v1",
                    "start_frame": start,
                    "subgoal_frame": end,
                    "chunk_length": chunk_length,
                    "image_role": "input at start and future subgoal at final transition",
                    "alignment": "frame[start] + actions[start:start+chunk_length] -> frame[end]",
                },
                file,
                indent=2,
            )
        chunk_count += 1
    (level_1 / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "wam-0.1-level-1-manifest-v1",
                "source_level_0": str(trajectory / "level_0"),
                "source_transitions": int(len(actions)),
                "chunk_length": int(chunk_length),
                "chunk_count": int(chunk_count),
                "dropped_tail_transitions": int(len(actions) - chunk_count * chunk_length),
            },
            indent=2,
        )
        + "\n"
    )
    return chunk_count


def collect_episode(args: argparse.Namespace, output_root: Path, episode_id: int, seed: int) -> dict:
    trajectory = output_root / f"trajectory_seed{seed:06d}"
    if trajectory.exists():
        if not args.overwrite:
            raise FileExistsError(f"trajectory exists: {trajectory}; use --overwrite to replace it")
        shutil.rmtree(trajectory)
    trajectory.mkdir(parents=True)

    env = gym.make(
        f"robocasa/{args.task}",
        split=args.split,
        seed=seed,
        camera_widths=128,
        camera_heights=128,
        camera_depths=True,
        enable_render=False,
        randomize_cameras=args.mode == "random",
        use_distractors=args.mode == "random",
    )
    actions = make_action_schedule(seed, args.max_steps)
    frames = []
    states = []
    success = False
    terminated = False
    truncated = False
    try:
        env.reset(seed=seed)
        raw_env = env.unwrapped.env
        frames.append(capture_frame(raw_env))
        states.append(read_state(raw_env))
        for action in actions:
            _, _, terminated, truncated, info = env.step(make_wrapper_action(action))
            frames.append(capture_frame(raw_env))
            states.append(read_state(raw_env))
            success = bool(info.get("success", False))
            if terminated or truncated:
                break
    finally:
        env.close()

    actions = actions[: len(frames) - 1]
    if len(actions) == 0:
        raise RuntimeError("episode produced no transitions")
    if len(states) != len(actions) + 1:
        raise RuntimeError("state/frame/action alignment failed")

    state_delta = float(np.linalg.norm(states[-1] - states[0]))
    pixel_delta = float(
        np.mean(
            np.abs(
                frames[0][CAMERAS[0]][0].astype(np.float32)
                - frames[-1][CAMERAS[0]][0].astype(np.float32)
            )
        )
    )
    if state_delta == 0.0 or pixel_delta == 0.0:
        raise RuntimeError(f"no measurable motion: state_delta={state_delta}, pixel_delta={pixel_delta}")

    level_0 = trajectory / "level_0"
    level_1 = trajectory / "level_1"
    level_0.mkdir()
    level_1.mkdir()
    save_level_0(trajectory, frames, actions, states)
    chunk_count = save_level_1(trajectory, actions, states, args.chunk_length)
    metadata = {
        "task": args.task,
        "split": args.split,
        "mode": args.mode,
        "seed": seed,
        "episode_id": episode_id,
        "robot": "Panda arm on RoboCasa PandaOmron wrapper",
        "base_motion_fixed_to_zero": True,
        "capture": "direct raw.sim.render",
        "cameras": list(CAMERAS),
        "image_size": [128, 128],
        "action_dim": 7,
        "action_layout": list(ACTION_LAYOUT),
        "robot_state_dim": int(states[0].shape[0]),
        "robot_state_layout": list(ROBOT_STATE_LAYOUT),
        "robot_state_schema": "wam-0.1-panda-state-v1",
        "transitions": len(actions),
        "transition_count": len(actions),
        "frames": len(frames),
        "level_1_chunks": chunk_count,
        "task_success_at_end": success,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "state_delta_l2": state_delta,
        "start_end_rgb_mean_abs_diff": pixel_delta,
        "level_0_alignment": "input[t] + action[t] -> observation[t+1]",
        "level_1_alignment": "frame[start] + consecutive actions -> frame[subgoal]",
    }
    with (trajectory / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    return metadata


def main() -> None:
    args = parse_args()
    output_root = args.out_dir or (
        project_root() / "data" / "controller_test" / "robocasa" / safe_name(args.task) / args.mode
    )
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for episode_id in range(args.episodes):
        seed = args.seed_start + episode_id
        result = collect_episode(args, output_root, episode_id, seed)
        results.append(result)
        print(
            f"episode={episode_id} seed={seed} transitions={result['transitions']} "
            f"level_1_chunks={result['level_1_chunks']} success={result['task_success_at_end']}"
        )
    with (output_root / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    print(f"output={output_root}")
    print(f"episodes={len(results)} transitions={sum(item['transitions'] for item in results)}")


if __name__ == "__main__":
    main()
