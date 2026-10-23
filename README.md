# wam-0.1

Panda action-conditioned dataset project for H-JEPA.

The project has two learning levels:

- Level 0: input frame + one 7D Panda action -> observation frame
- Level 1: input frame + action chunk -> future subgoal frame

The current approved test set contains 10 RoboSuite Panda tasks. Each task is tested in normal and randomized modes. Randomization changes the approved scene, camera, lighting, and object conditions while keeping the task controller and fixed white table rule intact.

Level 1 is built from consecutive Level 0 transitions. It never mixes episodes or chooses a random endpoint:

```text
Level 0: input[t] + action[t] -> observation[t]
Level 0: observation[t] == input[t + 1]
Level 1: input[t] + actions[t:t+8] -> observation[t+7]
```

## Project commands

```bash
source ~/robotics/roboenv/bin/activate
cd ~/robotics/wam-0.1
python scripts/show_plan.py
python scripts/test_controller.py
python scripts/test_two_levels.py
python scripts/collect_batch.py --workers 2
```

The two-level test writes only to `data/test_two_levels`. The production directory `data/final` remains empty until the batch collector is implemented and tested.
