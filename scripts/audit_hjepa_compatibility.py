from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT = Path("/home/aditya/robotics/wam-0.1")
FINAL = PROJECT / "data" / "final"
REPORT = PROJECT / "reports" / "hjepa_compatibility_audit.json"
CAMERA_MAP = {
    "wam_core": ("agentview", "frontview", "birdview"),
    "robocasa": ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"),
}
CANONICAL_STATE_SCHEMA = "wam-0.1-panda-state-v1"
CANONICAL_STATE_DIM = 30


def family_for(path: Path) -> str:
    return "robocasa" if "robocasa" in path.parts else "wam_core"


def trajectory_metadata(mode_root: Path):
    # Only inspect the known trajectory depths; frame folders can contain metadata too.
    candidates = list(mode_root.glob("*/*/metadata.json"))
    candidates.extend(mode_root.glob("*/*/*/metadata.json"))
    for path in sorted(set(candidates)):
        if path.parent.name.startswith("segment_"):
            continue
        if not (path.parent / "action.npy").exists():
            continue
        yield path


def inspect_trajectory(meta_path: Path) -> dict:
    root = meta_path.parent
    meta = json.loads(meta_path.read_text())
    family = family_for(root)
    actions = np.load(root / "action.npy", mmap_mode="r")
    state_dims = {}
    for state_path in root.glob("*_state.npy"):
        state_dims[state_path.stem] = int(np.load(state_path, mmap_mode="r").shape[-1])
    state_schema = meta.get("robot_state_schema", "legacy_sim_flattened")
    camera_names = tuple(meta.get("cameras") or CAMERA_MAP[family])
    if family == "robocasa":
        rgb_path = next((root / "frames" / camera_names[0]).glob("*/rgb.png"), None)
    else:
        rgb_path = next((root / "input" / camera_names[0]).glob("*/rgb.png"), None)
    image_size = list(Image.open(rgb_path).size) if rgb_path else None
    return {
        "family": family,
        "task": meta.get("task", root.parent.name),
        # The mode is the directory immediately below level_0.
        "mode": root.parts[root.parts.index("level_0") + 1],
        "trajectory": root.name,
        "transitions": int(actions.shape[0]),
        "action_shape": list(actions.shape),
        "action_min": float(np.min(actions)),
        "action_max": float(np.max(actions)),
        "has_nan_actions": bool(np.isnan(actions).any()),
        "state_dims": state_dims,
        "state_schema": state_schema,
        "canonical_state": state_schema == CANONICAL_STATE_SCHEMA and set(state_dims.values()) == {CANONICAL_STATE_DIM},
        "cameras": list(camera_names),
        "image_size": image_size,
        "success": meta.get("task_success_at_end", meta.get("validation", {}).get("success")),
        "official_done": meta.get("official_done_at_end"),
        "replay_error": meta.get("max_replay_state_error"),
        "level_1_chunks": int(meta.get("level_1_chunks", int(actions.shape[0]) // 8)),
        "absolute_paths": {
            key: value
            for key, value in meta.items()
            if key in {"source_dataset", "source_level_0"} and isinstance(value, str) and value.startswith("/")
        },
    }


def main() -> None:
    rows = []
    for mode_root in sorted((FINAL / "level_0").iterdir()):
        if mode_root.name == "quarantine" or not mode_root.is_dir():
            continue
        rows.extend(inspect_trajectory(path) for path in trajectory_metadata(mode_root))

    groups = defaultdict(list)
    for row in rows:
        groups[(row["mode"], row["family"])].append(row)

    summaries = {}
    warnings = []
    for (mode, family), values in sorted(groups.items()):
        errors = [float(x["replay_error"]) for x in values if x["replay_error"] is not None]
        tails = [x["transitions"] - x["level_1_chunks"] * 8 for x in values]
        state_shapes = sorted({tuple(sorted(x["state_dims"].values())) for x in values})
        summary = {
            "trajectories": len(values),
            "tasks": len({x["task"] for x in values}),
            "transitions": sum(x["transitions"] for x in values),
            "action_dims": sorted({x["action_shape"][1] for x in values}),
            "state_shapes": state_shapes,
            "state_schemas": sorted({x["state_schema"] for x in values}),
            "canonical_state_trajectories": sum(x["canonical_state"] for x in values),
            "legacy_state_trajectories": sum(not x["canonical_state"] for x in values),
            "camera_sets": sorted({tuple(x["cameras"]) for x in values}),
            "image_sizes": sorted({tuple(x["image_size"] or []) for x in values}),
            "nan_actions": sum(x["has_nan_actions"] for x in values),
            "out_of_bounds_trajectories": sum(x["action_min"] < -1.000001 or x["action_max"] > 1.000001 for x in values),
            "success_false": sum(x["success"] is False for x in values),
            "official_done_false": sum(x["official_done"] is False for x in values),
            "level_1_tail_transitions": sum(tails),
            "trajectories_with_level_1_tail": sum(tail > 0 for tail in tails),
        }
        if errors:
            summary["replay_error"] = {
                "count": len(errors),
                "min": min(errors),
                "median": float(np.median(errors)),
                "max": max(errors),
                "gt_1": sum(x > 1 for x in errors),
                "gt_5": sum(x > 5 for x in errors),
            }
        summaries[f"{mode}/{family}"] = summary
        if any(not x["canonical_state"] for x in values):
            warnings.append(
                f"{mode}/{family}: legacy simulator states are auxiliary only; "
                "regenerate this split with the canonical 30D Panda state before using state inputs"
            )
        if family == "robocasa" and errors and max(errors) > 1:
            warnings.append(f"{mode}/{family}: replay state error requires review before claiming exact fidelity")
        if summary["level_1_tail_transitions"]:
            warnings.append(f"{mode}/{family}: incomplete Level 1 tails are excluded from full chunks")

    absolute_path_count = sum(bool(row["absolute_paths"]) for row in rows)
    if absolute_path_count:
        warnings.append(f"{absolute_path_count} trajectories contain absolute source paths; convert them to relative metadata")
    if not (FINAL / "index.json").exists():
        warnings.append("data/final/index.json is missing; create one portable index before upload")
    if (FINAL / "quarantine").exists():
        warnings.append("data/final/quarantine exists; exclude it from the training upload")

    report = {
        "schema": "wam-0.1-hjepa-compatibility-audit-v1",
        "dataset_root": str(FINAL),
        "trajectory_count": len(rows),
        "transition_count": sum(row["transitions"] for row in rows),
        "summaries": summaries,
        "warnings": warnings,
        "status": "needs_packaging_and_fidelity_review" if warnings else "pass",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
