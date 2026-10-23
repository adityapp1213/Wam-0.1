# Data format

## Level 0

One directory represents one trajectory:

```text
trajectory_seed73000/
├── input/<camera>/<step>/rgb.png
├── input/<camera>/<step>/depth.npy
├── observation/<camera>/<step>/rgb.png
├── observation/<camera>/<step>/depth.npy
├── action.npy
├── input_state.npy
├── observation_state.npy
└── metadata.json
```

The alignment is:

```text
input_state[t] + action[t] -> observation_state[t]
observation[t] == input[t + 1]
```

The saved input and observation frames use the same transition index. The final action without a following observation is excluded. The continuity check compares RGB, depth, and simulator state between neighboring transitions.

## Level 1

Each segment is made from consecutive Level 0 transitions:

```text
segment_000000/
├── input/<camera>/000000/rgb.png
├── input/<camera>/000000/depth.npy
├── subgoal/<camera>/000000/rgb.png
├── subgoal/<camera>/000000/depth.npy
├── action_block.npy
├── start_state.npy
├── subgoal_state.npy
└── metadata.json
```

For an eight-action chunk beginning at `t`:

```text
input[t] + actions[t:t+8] -> observation[t+7]
```

The Level 1 input and subgoal images are hardlinks to the exact Level 0 files when possible. This keeps the images available at both levels without storing duplicate image bytes. A short tail at the end of a trajectory remains valid Level 0 data but is not made into an incomplete Level 1 chunk.
