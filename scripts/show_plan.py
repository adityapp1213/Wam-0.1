import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
project = json.loads((ROOT / "configs/project.json").read_text())
tasks = json.loads((ROOT / "configs/tasks.json").read_text())

print(project["project_name"])
print("Robot:", project["robot"])
print("Controller:", project["controller"]["robosuite_controller"])
print("Action dimension:", project["controller"]["action_dim"])
print("Cameras:", ", ".join(project["capture"]["cameras"]))
print("Pilot episodes:", project["pilot"]["total_episodes"])
print("Scale transitions:", project["scale"]["target_physical_transitions"])
print("\nTasks:")
for index, task in enumerate(tasks, start=1):
    print(f"{index:02d}. {task['name']} ({task['family']})")

