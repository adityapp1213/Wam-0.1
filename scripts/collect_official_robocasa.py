"""Collect one successful official RoboCasa demo per task into WAM-0.1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from robocasa.scripts.download_datasets import download_and_extract_from_box
from robocasa.utils.dataset_registry import (
    ATOMIC_TASK_DATASETS,
    COMPOSITE_TASK_DATASETS,
)
from robocasa.utils.dataset_registry_utils import get_ds_meta

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_official_robocasa import replay


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "data" / "normal" / "robocasa"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Task names. Omit this to use every official human pretrain task.",
    )
    parser.add_argument("--tasks-file", type=Path, default=None)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--chunk-length", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=1,
        help="Official successful episodes to try per task before recording a failure.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep successful trajectories in the existing manifest and retry the rest.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Fail a missing bundle instead of downloading it.",
    )
    return parser.parse_args()


def all_task_names() -> list[str]:
    names = set(ATOMIC_TASK_DATASETS) | set(COMPOSITE_TASK_DATASETS)
    return sorted(name for name in names if dataset_path(name) is not None)


def requested_tasks(args: argparse.Namespace) -> list[str]:
    if args.tasks_file is not None:
        names = [
            line.strip()
            for line in args.tasks_file.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    elif args.tasks is not None:
        names = args.tasks
    else:
        names = all_task_names()

    unique = list(dict.fromkeys(names))
    if args.max_tasks is not None:
        unique = unique[: args.max_tasks]
    return unique


def dataset_path(task: str) -> Path | None:
    meta = get_ds_meta(task=task, source="human", split="pretrain")
    if meta is None or meta.get("path") is None:
        return None
    return Path(meta["path"])


def bundle_ready(path: Path) -> bool:
    return (
        (path / "extras").is_dir()
        and (path / "data").is_dir()
        and (path / "extras" / "dataset_meta.json").is_file()
    )


def ensure_bundle(task: str, allow_download: bool) -> Path:
    path = dataset_path(task)
    if path is None:
        raise FileNotFoundError(f"No official human pretrain bundle for {task}")
    if bundle_ready(path):
        return path
    if not allow_download:
        raise FileNotFoundError(f"Official bundle is not downloaded: {path}")

    print(f"downloading task={task}")
    download_and_extract_from_box(destination=str(path))
    if not bundle_ready(path):
        raise RuntimeError(f"Bundle download did not produce a usable dataset: {path}")
    return path


def successful_episode_ids(path: Path) -> list[int]:
    files = sorted((path / "data").glob("*/episode_*.parquet"))
    candidates = []
    for file in files:
        episode_name = file.stem
        try:
            episode_id = int(episode_name.split("_")[-1])
        except ValueError:
            continue
        table = pd.read_parquet(file, columns=["action", "next.done"])
        done = table["next.done"]
        actions = np.stack(table["action"].to_list()).astype(np.float32)
        arm_only = float(np.max(np.abs(actions[:, 0:4]))) <= 1e-6
        if len(done) and bool(done.iloc[-1]) and arm_only:
            candidates.append(episode_id)
    return sorted(set(candidates))


def replay_one_task(
    task: str,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict:
    path = ensure_bundle(task, allow_download=not args.no_download)
    candidates = successful_episode_ids(path)[: args.max_candidates]
    if not candidates:
        raise RuntimeError("No official successful arm-only episode was found")

    last_error = None
    for episode in candidates:
        replay_args = argparse.Namespace(
            dataset_dir=path,
            episode=episode,
            chunk_length=args.chunk_length,
            image_size=args.image_size,
            out_dir=out_dir,
            overwrite=args.overwrite,
        )
        try:
            result = replay(replay_args)
            result["official_task_name"] = task
            result["candidate_episodes_checked"] = candidates.index(episode) + 1
            return result
        except Exception as error:  # keep the batch moving and record the cause
            last_error = f"episode {episode}: {type(error).__name__}: {error}"
            print(f"rejected task={task} {last_error}")

    raise RuntimeError(last_error or "all official episodes were rejected")


def main() -> None:
    args = parse_args()
    tasks = requested_tasks(args)
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.out_dir / "manifest.json"
    if args.resume and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest["tasks_requested"] = tasks
    else:
        manifest = {
            "dataset": "wam-0.1",
            "scope": "one successful official RoboCasa human pretrain demo per task",
            "randomness": "disabled",
            "output_root": str(args.out_dir),
            "chunk_length": args.chunk_length,
            "image_size": [args.image_size, args.image_size],
            "tasks_requested": tasks,
            "results": [],
        }
    results_by_task = {
        item["official_task_name"]: item
        for item in manifest.get("results", [])
        if item.get("official_task_name")
    }

    print(f"tasks requested={len(tasks)}")
    for index, task in enumerate(tasks, start=1):
        print(f"[{index}/{len(tasks)}] task={task}")
        if args.resume and results_by_task.get(task, {}).get("status") == "ok":
            print("  skip: already completed")
            continue
        try:
            result = replay_one_task(task, args, args.out_dir)
            result["status"] = "ok"
            print(
                f"  ok episode={result['official_episode']} "
                f"transitions={result['transitions']} "
                f"level_1_chunks={result['level_1_chunks']}"
            )
        except Exception as error:
            result = {
                "official_task_name": task,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
            print(f"  failed: {result['error']}")
        results_by_task[task] = result
        manifest["results"] = [
            results_by_task[name] for name in tasks if name in results_by_task
        ]

        with manifest_path.open("w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2)

    completed = [item for item in manifest["results"] if item["status"] == "ok"]
    failed = [item for item in manifest["results"] if item["status"] == "failed"]
    transitions = sum(item.get("transitions", 0) for item in completed)
    chunks = sum(item.get("level_1_chunks", 0) for item in completed)
    print("batch complete")
    print(f"completed_tasks={len(completed)}")
    print(f"failed_tasks={len(failed)}")
    print(f"level_0_transitions={transitions}")
    print(f"level_1_chunks={chunks}")
    print(f"manifest={args.out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
