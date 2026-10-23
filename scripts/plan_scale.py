import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    project = json.loads((ROOT / "configs" / "project.json").read_text())
    tasks = json.loads((ROOT / "configs" / "tasks.json").read_text())
    target = int(project["scale"]["target_physical_transitions"])
    views = len(project["capture"]["cameras"])
    task_count = len(tasks)
    base, remainder = divmod(target, task_count)
    allocation = {
        task["name"]: base + int(index < remainder)
        for index, task in enumerate(tasks)
    }
    print(json.dumps({
        "tasks": task_count,
        "target_physical_transitions": target,
        "target_input_action_observation_tuples": target,
        "stored_rgbd_frames": target * views * 2,
        "transitions_per_task": allocation,
        "validation_rule": "only successful RoboSuite episodes enter final data",
        "scaling_rule": "measure valid transition lengths before converting task budgets to episode counts",
    }, indent=2))


if __name__ == "__main__":
    main()
