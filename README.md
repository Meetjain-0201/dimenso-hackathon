# Dimenso Hackathon — Egocentric Lab Demo to Unitree G1 in Simulation

Recover human body and hand motion from smart-glasses **head-cam RGB video + head
IMU**, then translate that motion onto a **Unitree G1** humanoid and replay it in
**MuJoCo** simulation. This project is **motion translation only** — human demo in,
robot joint trajectory out, replayed in sim. There is no pick-and-place, object
interaction, or manipulation planning. The specific demonstrated task will be
selected (TBD) after reviewing the actual recordings.

## Repo layout

| Folder | Purpose |
| --- | --- |
| `data_pipeline/` | Perception + IMU sync + dataset build — the core. `inspect.py` (probe a recording), `imu_sync.py` (align IMU to frames, recover head pose), `pose_extract.py` (MediaPipe body/hand landmarks + angles), `stabilize.py` (camera-frame → world-frame via head pose), `build_dataset.py` (assemble clean joint-angle + gripper trajectories). |
| `notebooks/` | Exploration and sanity checks. |
| `method/` | Retargeting + imitation-learning sketch. `retarget.py` (human angles → G1 joint targets), `policy_sketch.md` (IL sketch, not trained). |
| `sim/` | MuJoCo G1 replay. `replay_g1.py` (load G1, replay retargeted trajectory). |
| `report/` | `report.md` and `figures/`. |

## Data

Recordings live under **`data/<task>/<...>/{base.mp4, imu.json}`** at the repo root.
`data/` is **gitignored and NOT committed** — video/IMU stay local. Generated
artifacts (`outputs/`, `*.mp4`, `*.npz`) are excluded too. Model assets
(`sim/assets/g1/`, `sim/assets/inspire/`) and report figures **are** committed.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # numpy, opencv, scipy, matplotlib, mediapipe, mujoco
```

### Offline motion-translation engine (egocentric video + IMU → G1+Inspire replay)

Render uses `MUJOCO_GL=glfw` (osmesa/egl fail on this box — see docs/diagnostics.md).

```bash
# Build the dataset + render the replay mp4 (+ sample frames).
# Default recording: first under data/task_04_pasteur_pipette/
MUJOCO_GL=glfw python -m data_pipeline.run_offline
#   → outputs/task04_dataset.npz (+ .schema.json)   ← the dataset deliverable
#   → outputs/replay_task04.mp4                       ← G1+Inspire replay
#   → report/figures/replay_frame_*.png

# Same, but also drive the LIVE passive viewer in real time:
MUJOCO_GL=glfw python -m data_pipeline.run_offline --live

# Point at any recording explicitly:
MUJOCO_GL=glfw python -m data_pipeline.run_offline data/task_04_pasteur_pipette/<rec> --name task04

# Frame-aligned side-by-side (source video ∥ replay):
MUJOCO_GL=glfw python sim/replay_g1.py --dataset outputs/task04_dataset.npz \
    --compare data/task_04_pasteur_pipette/<rec>/base.mp4 --mp4 outputs/compare_task04.mp4
```

Individual stages are runnable too (`python -m data_pipeline.pose_extract <rec>`, etc.).
Sanity plots: `notebooks/01_explore.ipynb`.
