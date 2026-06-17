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

The dataset lives in **`mentra-pull/`** at the repo root. It is **gitignored and
NOT committed** — recordings (video/IMU) stay local. Generated artifacts
(`data/`, `outputs/`, `*.mp4`, `*.npz`) are likewise excluded.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# TODO: run commands for the pipeline once stubs are implemented, e.g.
#   python -m data_pipeline.inspect <recording>
#   python -m data_pipeline.build_dataset ...
#   python sim/replay_g1.py ...
```
