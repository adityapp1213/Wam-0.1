from __future__ import annotations

import json
from pathlib import Path


PROJECT = Path("/home/aditya/robotics/wam-0.1")
FINAL = PROJECT / "data" / "final"
INDEX = FINAL / "index.json"


def rel(path: Path) -> str:
    return path.relative_to(PROJECT).as_posix()


def family_for(path: Path) -> str:
    return "robocasa" if "robocasa" in path.parts else "wam_core"


def mode_for(path: Path, level: str) -> str:
    parts = path.parts
    return parts[parts.index(level) + 1]


def trajectory_meta(level_root: Path):
    # Keep the scan shallow so frame folders are never treated as trajectories.
    patterns = ("*/*/metadata.json", "*/*/*/metadata.json")
    found = set()
    for pattern in patterns:
        found.update(level_root.glob(pattern))
    for path in sorted(found):
        if path.parent.name.startswith("segment_"):
            continue
        if (path.parent / "action.npy").is_file():
            yield path


def segment_meta(level_root: Path):
    patterns = (
        "*/*/*/metadata.json",
        "*/*/*/*/metadata.json",
    )
    found = set()
    for pattern in patterns:
        found.update(level_root.glob(pattern))
    for path in sorted(found):
        if path.parent.name.startswith("segment_"):
            yield path


def level0_entry(meta_path: Path) -> dict:
    root = meta_path.parent
    meta = json.loads(meta_path.read_text())
    action_path = root / "action.npy"
    transitions = int(meta.get("transitions", 0))
    if not transitions:
        import numpy as np

        transitions = int(np.load(action_path, mmap_mode="r").shape[0])
    chunk_length = int(meta.get("chunk_length", 8))
    chunks = int(meta.get("level_1_chunks", transitions // chunk_length))
    return {
        "level": "level_0",
        "source_family": family_for(root),
        "mode": mode_for(root, "level_0"),
        "task": meta.get("task", root.parent.name),
        "trajectory": root.name,
        "trajectory_dir": rel(root),
        "metadata": rel(meta_path),
        "action": rel(action_path),
        "transitions": transitions,
        "chunk_length": chunk_length,
        "level_1_full_chunks": chunks,
        "level_1_tail_transitions": transitions - chunks * chunk_length,
        "cameras": meta.get("cameras", []),
        "image_size": meta.get("image_size", [128, 128]),
        "action_dim": int(meta.get("action_dim", 7)),
        "success": meta.get("task_success_at_end"),
    }


def level1_entry(meta_path: Path) -> dict:
    segment = meta_path.parent
    trajectory = segment.parent
    meta = json.loads(meta_path.read_text())
    if "start_frame" in meta:
        start_frame = meta["start_frame"]
        subgoal_frame = meta["subgoal_frame"]
    else:
        # WAM metadata stores end_transition as an inclusive action index.
        start_frame = meta["start_transition"]
        subgoal_frame = meta["end_transition"] + 1
    return {
        "level": "level_1",
        "source_family": family_for(trajectory),
        "mode": mode_for(trajectory, "level_1"),
        "task": trajectory.parent.name,
        "trajectory": trajectory.name,
        "trajectory_dir": rel(trajectory),
        "segment": segment.name,
        "metadata": rel(meta_path),
        "action_block": rel(segment / "action_block.npy"),
        "input_dir": rel(segment / "input"),
        "subgoal_dir": rel(segment / "subgoal"),
        "start_frame": int(start_frame),
        "subgoal_frame": int(subgoal_frame),
        "chunk_length": int(meta["chunk_length"]),
        "alignment": meta.get(
            "alignment",
            "frame[start] + consecutive actions -> frame[subgoal]",
        ),
    }


def main() -> None:
    level0 = []
    level1 = []
    for level in ("level_0", "level_1"):
        level_root = FINAL / level
        if not level_root.is_dir():
            continue
        for mode_root in sorted(level_root.iterdir()):
            if not mode_root.is_dir() or mode_root.name == "quarantine":
                continue
            if level == "level_0":
                level0.extend(level0_entry(path) for path in trajectory_meta(mode_root))
            else:
                level1.extend(level1_entry(path) for path in segment_meta(mode_root))

    index = {
        "dataset": "wam-0.1",
        "schema": "configs/hjepa_schema_v1.json",
        "index_version": "wam-0.1-index-v1",
        "root": "data/final",
        "levels": {
            "level_0": {
                "meaning": "input frame + one 7D action -> next observation frame",
                "entries": level0,
                "trajectory_count": len(level0),
                "transition_count": sum(item["transitions"] for item in level0),
            },
            "level_1": {
                "meaning": "input frame + action chunk -> future subgoal frame",
                "entries": level1,
                "chunk_count": len(level1),
            },
        },
        "training_notes": {
            "views_are_not_actions": True,
            "level_1_uses_only_complete_chunks": True,
            "exclude": ["data/final/quarantine"],
            "split_unit": "trajectory",
        },
    }
    INDEX.write_text(json.dumps(index, indent=2) + "\n")
    print(
        f"index={INDEX} level_0_trajectories={len(level0)} "
        f"level_0_transitions={index['levels']['level_0']['transition_count']} "
        f"level_1_chunks={len(level1)}"
    )


if __name__ == "__main__":
    main()
