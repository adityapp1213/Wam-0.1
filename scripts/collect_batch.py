"""Collect accepted Level 0 trajectories and build their Level 1 chunks."""

import argparse
import concurrent.futures
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STAGING = ROOT / "data" / "controller_test"
PROJECT = json.loads((ROOT / "configs" / "project.json").read_text())
DEFAULT_TASKS = [item["name"] for item in json.loads((ROOT / "configs" / "tasks.json").read_text())]
DEFAULT_MODES = ("normal", "random")
DEFAULT_TARGET = int(PROJECT["scale"]["target_physical_transitions"])

sys.path.insert(0, str(ROOT / "src"))

from wam_dataset.levels import build_level1, validate_level1


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def max_steps_for(task):
    return 500 if task == "Wipe" else 1400


def trajectory_count(root, mode, task):
    total = 0
    task_root = root / "level_0" / mode / task
    for action_path in task_root.glob("trajectory_seed*/action.npy"):
        try:
            actions = np.load(action_path, mmap_mode="r")
            total += int(actions.shape[0])
        except (OSError, ValueError):
            continue
    return total


def existing_seed_max(root):
    values = []
    for path in root.rglob("trajectory_seed*"):
        try:
            values.append(int(path.name.replace("trajectory_seed", "")))
        except ValueError:
            continue
    return max(values, default=0)


def remove_staging_trajectory(task, seed):
    for branch in ("reference", "rejected", "reference_pending"):
        path = STAGING / branch / task / f"trajectory_seed{seed:04d}"
        if path.exists():
            shutil.rmtree(path)


def collect_one(task, mode, seed):
    command = [
        sys.executable,
        str(ROOT / "scripts" / "collect_reference.py"),
        "--task", task,
        "--seed", str(seed),
        "--max-steps", str(max_steps_for(task)),
    ]
    if mode == "normal":
        command.append("--no-camera-variation")
    started = time.monotonic()
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    elapsed = time.monotonic() - started
    source = STAGING / "reference" / task / f"trajectory_seed{seed:04d}"
    if result.returncode == 0 and source.is_dir():
        return {
            "status": "accepted",
            "task": task,
            "mode": mode,
            "seed": int(seed),
            "source": source,
            "seconds": float(elapsed),
        }
    remove_staging_trajectory(task, seed)
    log = (result.stdout + "\n" + result.stderr).strip()
    return {
        "status": "rejected",
        "task": task,
        "mode": mode,
        "seed": int(seed),
        "seconds": float(elapsed),
        "reason": log[-1000:],
    }


def write_manifest(path, manifest):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(path)


def make_quotas(tasks, modes, transitions_per_bucket, total_target):
    keys = [(mode, task) for mode in modes for task in tasks]
    if transitions_per_bucket is not None:
        return {key: int(transitions_per_bucket) for key in keys}
    if total_target is None:
        raise ValueError("set --target-transitions or --transitions-per-bucket")
    base, remainder = divmod(int(total_target), len(keys))
    return {key: base + int(index < remainder) for index, key in enumerate(keys)}


def choose_bucket(quotas, counts, outstanding, start_index):
    keys = list(quotas)
    for offset in range(len(keys)):
        index = (start_index + offset) % len(keys)
        key = keys[index]
        if counts[key] + outstanding[key] < quotas[key]:
            return key, (index + 1) % len(keys)
    return None, start_index


def move_and_build(result, output_root):
    task = result["task"]
    mode = result["mode"]
    source = result["source"]
    trajectory_name = source.name
    level0_root = output_root / "level_0" / mode / task / trajectory_name
    level1_root = output_root / "level_1" / mode / task / trajectory_name
    level0_root.parent.mkdir(parents=True, exist_ok=True)
    if level0_root.exists():
        raise FileExistsError(f"refusing to overwrite {level0_root}")
    shutil.move(str(source), str(level0_root))
    metadata_path = level0_root / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["batch"] = {
        "mode": mode,
        "task": task,
        "seed": result["seed"],
        "accepted_seconds": result["seconds"],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    try:
        build_level1(level0_root, level1_root, chunk_length=8)
        alignment = validate_level1(level0_root, level1_root, chunk_length=8)
    except Exception:
        if level1_root.exists():
            shutil.rmtree(level1_root)
        quarantine = output_root / "quarantine" / mode / task / trajectory_name
        quarantine.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(level0_root), str(quarantine))
        raise
    return int(alignment["level_0_transitions"]), int(alignment["level_1_chunks"])


def main():
    parser = argparse.ArgumentParser(description="collect validated WAM-0.1 Level 0 and Level 1 data")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--modes", nargs="+", choices=DEFAULT_MODES, default=list(DEFAULT_MODES))
    parser.add_argument("--target-transitions", type=int, default=DEFAULT_TARGET)
    parser.add_argument("--transitions-per-bucket", type=int, default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed-start", type=int, default=800000)
    parser.add_argument("--output", default="data/final")
    parser.add_argument("--manifest", default="data/manifests/collection_manifest.json")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.target_transitions is not None and args.target_transitions <= 0:
        raise ValueError("--target-transitions must be positive")
    if args.transitions_per_bucket is not None and args.transitions_per_bucket <= 0:
        raise ValueError("--transitions-per-bucket must be positive")
    unknown = sorted(set(args.tasks) - set(DEFAULT_TASKS))
    if unknown:
        raise ValueError(f"tasks are not in configs/tasks.json: {unknown}")

    output_root = Path(args.output)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    output_root.mkdir(parents=True, exist_ok=True)
    quotas = make_quotas(args.tasks, args.modes, args.transitions_per_bucket, args.target_transitions)
    counts = {key: trajectory_count(output_root, *key) for key in quotas}
    outstanding = {key: 0 for key in quotas}
    next_seed = max(args.seed_start, existing_seed_max(output_root) + 1)
    cursor = 0
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("tasks") != list(args.tasks) or manifest.get("modes") != list(args.modes):
            raise ValueError("existing manifest task/mode scope does not match this run")
        manifest.setdefault("accepted_episodes", [])
        manifest.setdefault("rejected_attempts", [])
        manifest.setdefault("started_at", utc_now())
    else:
        manifest = {
            "schema": "wam-0.1-collection-manifest-v1",
            "started_at": utc_now(),
            "accepted_episodes": [],
            "rejected_attempts": [],
        }
    manifest.update({
        "updated_at": utc_now(),
        "tasks": list(args.tasks),
        "modes": list(args.modes),
        "workers": int(args.workers),
        "quotas": {f"{mode}/{task}": quota for (mode, task), quota in quotas.items()},
        "counts": {f"{mode}/{task}": count for (mode, task), count in counts.items()},
    })
    write_manifest(manifest_path, manifest)

    def complete():
        return all(counts[key] >= quota for key, quota in quotas.items())

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {}
        while not complete() or pending:
            while not complete() and len(pending) < args.workers:
                key, cursor = choose_bucket(quotas, counts, outstanding, cursor)
                if key is None:
                    break
                mode, task = key
                seed = next_seed
                next_seed += 1
                outstanding[key] += 1
                future = pool.submit(collect_one, task, mode, seed)
                pending[future] = (key, seed)
            if not pending:
                break
            done, _ = concurrent.futures.wait(
                pending, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                key, seed = pending.pop(future)
                outstanding[key] -= 1
                result = future.result()
                if result["status"] == "rejected":
                    manifest["rejected_attempts"].append({
                        "task": result["task"],
                        "mode": result["mode"],
                        "seed": result["seed"],
                        "seconds": result["seconds"],
                        "reason": result.get("reason", "rejected"),
                    })
                    print(f"REJECT {result['mode']:6s} {result['task']:20s} seed={seed}", flush=True)
                    continue
                try:
                    transitions, chunks = move_and_build(result, output_root)
                except Exception as exc:
                    manifest["rejected_attempts"].append({
                        "task": result["task"],
                        "mode": result["mode"],
                        "seed": result["seed"],
                        "seconds": result["seconds"],
                        "reason": f"level_validation_failed: {type(exc).__name__}: {exc}",
                    })
                    print(f"QUARANTINE {result['mode']:6s} {result['task']:20s} seed={seed} {exc}", flush=True)
                    continue
                counts[key] += transitions
                manifest["accepted_episodes"].append({
                    "task": result["task"],
                    "mode": result["mode"],
                    "seed": result["seed"],
                    "transitions": transitions,
                    "level_1_chunks": chunks,
                    "seconds": result["seconds"],
                })
                print(
                    f"ACCEPT {result['mode']:6s} {result['task']:20s} seed={seed} "
                    f"transitions={transitions} chunks={chunks}",
                    flush=True,
                )
                manifest["updated_at"] = utc_now()
                manifest["counts"] = {f"{m}/{t}": count for (m, t), count in counts.items()}
                write_manifest(manifest_path, manifest)

    manifest["updated_at"] = utc_now()
    manifest["complete"] = complete()
    manifest["counts"] = {f"{m}/{t}": count for (m, t), count in counts.items()}
    manifest["total_transitions"] = int(sum(counts.values()))
    manifest["total_level_1_chunks"] = int(
        sum(item["level_1_chunks"] for item in manifest["accepted_episodes"])
    )
    write_manifest(manifest_path, manifest)
    print(json.dumps({
        "complete": manifest["complete"],
        "total_transitions": manifest["total_transitions"],
        "total_level_1_chunks": manifest["total_level_1_chunks"],
        "accepted_episodes": len(manifest["accepted_episodes"]),
        "rejected_attempts": len(manifest["rejected_attempts"]),
        "manifest": str(manifest_path),
    }, indent=2))
    return 0 if manifest["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
