"""Run a small normal/random test for both H-JEPA dataset levels."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wam_dataset.levels import build_level1, validate_level1


DEFAULT_TASKS = ("Lift", "Door", "PickPlaceBread")
CAMERA = "agentview"


def _run_reference(task, seed, output_root, max_steps, random_mode):
    """Collect one accepted reference trajectory and move it into the test set."""
    command = [
        sys.executable,
        str(ROOT / "scripts" / "collect_reference.py"),
        "--task", task,
        "--seed", str(seed),
        "--max-steps", str(max_steps),
    ]
    if not random_mode:
        command.append("--no-camera-variation")
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    source = ROOT / "data" / "controller_test" / "reference" / task / f"trajectory_seed{seed:04d}"
    if result.returncode != 0 or not source.is_dir():
        return None, result.stdout + "\n" + result.stderr
    destination = output_root / task / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    shutil.move(str(source), str(destination))
    return destination, result.stdout


def _image(path):
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def _variation_check(normal_root, random_root):
    """Measure whether randomized input images differ from normal inputs."""
    normal = _image(normal_root / "input" / CAMERA / "000000" / "rgb.png")
    random = _image(random_root / "input" / CAMERA / "000000" / "rgb.png")
    diff = ImageChops.difference(normal, random)
    diff_array = np.asarray(diff, dtype=np.float32)
    return {
        "mean_abs_rgb_difference": float(diff_array.mean()),
        "changed_pixel_fraction": float(np.mean(np.max(diff_array, axis=2) > 0)),
        "different": bool(diff_array.max() > 0),
    }


def _make_sheet(rows, output):
    """Make a quick visual check: normal input, random input, and their diff."""
    tile_w, tile_h = 256, 256
    label_h = 28
    sheet = Image.new("RGB", (tile_w * 3, (tile_h + label_h) * len(rows)), "white")
    draw = ImageDraw.Draw(sheet)
    for row_index, row in enumerate(rows):
        top = row_index * (tile_h + label_h)
        normal = _image(row["normal"] / "input" / CAMERA / "000000" / "rgb.png")
        random = _image(row["random"] / "input" / CAMERA / "000000" / "rgb.png")
        normal = normal.resize((tile_w, tile_h))
        random = random.resize((tile_w, tile_h))
        diff = ImageChops.difference(normal, random)
        sheet.paste(normal, (0, top + label_h))
        sheet.paste(random, (tile_w, top + label_h))
        sheet.paste(diff, (tile_w * 2, top + label_h))
        label = f"{row['task']} | normal input | random input | diff"
        draw.text((4, top + 6), label, fill="black")
    sheet.save(output)


def main():
    parser = argparse.ArgumentParser(description="small Level 0 + Level 1 integrity test")
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--max-steps", type=int, default=1400)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--out", default="data/test_two_levels")
    args = parser.parse_args()

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty test output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    rejected = []
    for task_index, task in enumerate(args.tasks):
        task_roots = {}
        for mode, random_mode in (("normal", False), ("random", True)):
            # Normal keeps the base scene; random keeps the approved variation hooks.
            mode_root = out / "level_0" / mode
            start_seed = 73000 + task_index * 100 + (50 if random_mode else 0)
            trajectory = None
            last_log = ""
            for attempt in range(args.attempts):
                seed = start_seed + attempt
                trajectory, last_log = _run_reference(
                    task, seed, mode_root, args.max_steps, random_mode
                )
                if trajectory is not None:
                    break
                rejected.append({"task": task, "mode": mode, "seed": seed, "log": last_log[-1200:]})
            if trajectory is None:
                raise RuntimeError(f"no valid {mode} trajectory for {task} after {args.attempts} attempts")
            task_roots[mode] = trajectory
            # Level 1 is built only after Level 0 has passed its checks.
            level1_root = out / "level_1" / mode / task / trajectory.name
            build_level1(trajectory, level1_root, chunk_length=8)
            alignment = validate_level1(trajectory, level1_root, chunk_length=8)
            (trajectory / "level_1_test.json").write_text(json.dumps(alignment, indent=2) + "\n")
        variation = _variation_check(task_roots["normal"], task_roots["random"])
        rows.append({"task": task, "normal": task_roots["normal"], "random": task_roots["random"], "variation": variation})

    report = {
        "schema": "wam-0.1-two-level-test-v1",
        "tasks": list(args.tasks),
        "normal_trajectories": len(rows),
        "random_trajectories": len(rows),
        "level_0_valid": len(rows) * 2,
        "level_1_valid": len(rows) * 2,
        "chunk_length": 8,
        "rejected_attempts": rejected,
        "variation_checks": [
            {"task": row["task"], **row["variation"]} for row in rows
        ],
        "rows": [
            {
                "task": row["task"],
                "normal": str(row["normal"]),
                "random": str(row["random"]),
            }
            for row in rows
        ],
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    report_dir = ROOT / "reports" / "two_level_test"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    _make_sheet(rows, report_dir / "contact_sheet.png")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
