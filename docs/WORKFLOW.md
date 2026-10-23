# WAM-0.1 workflow

## What each level means

Level 0 is the atomic training record:

```text
input frame + one 7D Panda action -> observation frame
```

Level 1 is a short-horizon record made from Level 0:

```text
input frame + eight consecutive actions -> future subgoal frame
```

The subgoal is always the observation reached after the final action in the chunk.

## Main folders

```text
wam-0.1/
├── configs/       fixed dataset and task settings
├── data/final/    production dataset; empty until scale collection
├── data/test_two_levels/  small normal/random integrity test
├── docs/          format and workflow notes
├── reports/       visual checks and validation reports
├── scripts/       collection, testing, and reporting commands
└── src/wam_dataset/  reusable actions, variation, controllers, and level checks
```

## Test before scale

Run the small test first:

```bash
source ~/robotics/roboenv/bin/activate
cd ~/robotics/wam-0.1
python scripts/test_two_levels.py
```

The test checks:

- normal and randomized task success;
- valid 7D actions;
- input/observation RGB and depth files;
- continuity between neighboring Level 0 transitions;
- exact Level 1 action slices;
- exact Level 1 input and subgoal images;
- no cross-episode chunks;
- visual differences between normal and randomized inputs.

Only after this passes should the batch collector write to `data/final`.

## Production collection

The collector reads the 10-task list and targets 400,000 transitions split evenly across normal and random modes:

```bash
python scripts/collect_batch.py --workers 2
```

It can be interrupted and rerun. The manifest keeps accepted episodes, rejected attempts, per-bucket transition counts, and Level 1 chunk counts.
