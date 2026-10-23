import copy
import json
import sys
from pathlib import Path

import robosuite as suite
from robosuite.controllers import load_composite_controller_config


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wam_dataset.controllers.pick_place import PickPlaceReferenceController, PickPlaceSequenceReferenceController
from wam_dataset.controllers.simple_tasks import (
    LiftReferenceController,
    NutAssemblyReferenceController,
    NutAssemblySequenceReferenceController,
    StackReferenceController,
)
from wam_dataset.controllers.task_specific import DoorReferenceController, ToolHangReferenceController, WipeReferenceController


TASKS = [
    "Lift", "Stack", "Door", "ToolHang", "Wipe", "PickPlace", "PickPlaceBread",
    "PickPlaceCan", "PickPlaceCereal", "PickPlaceMilk", "PickPlaceSingle",
    "NutAssembly", "NutAssemblyRound", "NutAssemblySingle", "NutAssemblySquare",
]


def make_policy(task, steps):
    if task == "PickPlace":
        return PickPlaceSequenceReferenceController(task, max_steps=steps)
    if task.startswith("PickPlace"):
        return PickPlaceReferenceController(task, max_steps=steps)
    if task == "Lift":
        return LiftReferenceController(task, max_steps=steps)
    if task == "Stack":
        return StackReferenceController(task, max_steps=steps)
    if task == "NutAssembly":
        return NutAssemblySequenceReferenceController(task, max_steps=steps)
    if task.startswith("NutAssembly"):
        return NutAssemblyReferenceController(task, max_steps=steps)
    if task == "Door":
        return DoorReferenceController(task, max_steps=steps)
    if task == "ToolHang":
        return ToolHangReferenceController(task, max_steps=steps)
    if task == "Wipe":
        return WipeReferenceController(task, max_steps=steps)
    raise ValueError(task)


def run_task(task, seed, controller_cfg):
    max_steps = 500 if task == "Wipe" else 900 if task == "ToolHang" else 1400
    env = suite.make(
        env_name=task,
        robots="Panda",
        controller_configs=copy.deepcopy(controller_cfg),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        seed=seed,
        horizon=max(max_steps, 1000),
    )
    try:
        env.reset()
        policy = make_policy(task, max_steps)
        policy.reset(env, seed)
        for step in range(max_steps):
            action = policy.act(env)
            _, _, terminated, _ = env.step(action)
            policy.update(env)
            if terminated or policy.done(env):
                break
        return {"task": task, "seed": seed, "steps": step + 1, **policy.result().as_dict()}
    except Exception as exc:
        return {"task": task, "seed": seed, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        env.close()


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    controller_cfg = load_composite_controller_config(robot="Panda")
    report = [run_task(task, seed, controller_cfg) for task in TASKS]
    out = ROOT / "reports" / f"controller_matrix_seed{seed:04d}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"path": str(out), "passed": sum(bool(item.get("success")) for item in report), "total": len(report)}, indent=2))


if __name__ == "__main__":
    main()
