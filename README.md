# Dimenso Hackathon — Egocentric Lab Demo to Unitree G1 in Simulation

Recover human **hand + arm** motion from smart-glasses **head-cam RGB video + head
IMU**, retarget it onto a **Unitree G1 with five-finger Inspire hands**, and replay it
in **MuJoCo**. **Motion translation only** — human demo in, robot joint trajectory out,
replayed in sim (no object interaction / manipulation planning). Built and validated on
`task_04_pasteur_pipette`. Full write-up: [`report/report.pdf`](report/report.pdf)
(source `report/report.md`); slides: `report/dimenso_slides.pptx`.

## Repo layout

| Folder | Purpose |
| --- | --- |
| `data_pipeline/` | Perception + IMU sync + dataset build (the core). `pose_extract.py` (MediaPipe Hands + One-Euro + curl/pinch/thumb-distances), `imu_sync.py` (resample IMU→frames, recover head orientation), `stabilize.py` (2D wrist→approx 3D target), `build_dataset.py` (assemble + write the documented dataset), `run_offline.py` (one-command runner). |
| `notebooks/` | `01_explore.ipynb` — sanity plots + IMU visualization. |
| `method/` | `retarget.py` (DLS IK → G1+Inspire joint targets), `policy_sketch.md`. |
| `sim/` | `replay_g1.py` (replay / live viewer / side-by-side), `pov_compare.py` (POV IK-diagnosis harness), `assets/` (committed G1+Inspire model + meshes + scene). |
| `report/` | `report.md` + `report.pdf` + `dimenso_slides.pptx` + `figures/`. |
| `docs/` | `diagnostics.md`, `inspire_hand_model.md`, `live_architecture.md`. |

## Setup

Python **3.11**. Install the deps:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # numpy, opencv-python, scipy, matplotlib, mediapipe, mujoco
```
(Developed/verified with `mediapipe 0.10.21`, `mujoco 3.9.0`, `opencv 4.11`, `numpy 1.26`.)

Render uses **`MUJOCO_GL=glfw`** — `osmesa`/`egl` fail on this hardware (see
`docs/diagnostics.md`). Prefix the run commands with it.

## Data (gitignored — you must provide it)

Recordings are **not committed** (`data/` is gitignored). Place the task_04 recording at:

```
data/task_04_pasteur_pipette/<recording_id>/base.mp4
data/task_04_pasteur_pipette/<recording_id>/imu.json
```

The G1+Inspire **model + meshes** (`sim/assets/`) and the **report figures** ARE
committed, so no model/asset setup is needed. **What won't run without the recording:**
`run_offline.py`, `replay_g1.py`, `pov_compare.py`, and the notebook all need
`base.mp4` + `imu.json` present (they read frames + IMU). Everything else (model load,
schema) works without data.

## Reproduce — exact commands

```bash
# 1) Offline engine: build the dataset AND render the replay mp4 (+ sample frames).
#    Default recording = first under data/task_04_pasteur_pipette/.
MUJOCO_GL=glfw python -m data_pipeline.run_offline
#    OUTPUTS: outputs/task04_dataset.npz        ← the dataset deliverable
#             outputs/task04_dataset.schema.json ← committed-format contract (documents every field)
#             outputs/replay_task04.mp4          ← G1+Inspire replay video
#             report/figures/replay_frame_*.png  ← sample replay frames

# 2) Launch the LIVE native MuJoCo viewer playing the trajectory in real time:
MUJOCO_GL=glfw python -m data_pipeline.run_offline --live
#    (opens a MuJoCo window; also writes the dataset + mp4 as in step 1)

# 3) Regenerate the side-by-side comparisons + the 250-frame clip (POV harness):
MUJOCO_GL=glfw python sim/pov_compare.py --clip 250
#    OUTPUTS: report/figures/diag_compare_*.png  ← 10 tagged source∥POV frames
#             sim/assets/diag_compare_grid.png   ← contact sheet (also copied to report/figures/)
#             outputs/tuned_compare_250.mp4      ← 250-frame side-by-side clip
#    (MediaPipe pose is cached to outputs/pose_cache_task04.npz; reruns after an
#     IK-only change are fast. Use --no-cache to force re-extraction.)

# 4) Explore the dataset + regenerate sanity plots / IMU visualization:
MUJOCO_GL=glfw python sim/replay_g1.py --dataset outputs/task04_dataset.npz \
    --compare data/task_04_pasteur_pipette/<rec>/base.mp4 --mp4 outputs/compare_task04.mp4   # optional standalone compare
jupyter notebook notebooks/01_explore.ipynb
#    The notebook reads outputs/task04_dataset.npz and writes report/figures/plot_*.png
#    + report/figures/imu_visualization.png
```

Individual pipeline stages run standalone too, e.g.
`python -m data_pipeline.pose_extract data/task_04_pasteur_pipette/<rec>`.

`outputs/` (npz, mp4) is gitignored; `report/figures/`, `report/report.pdf`,
`report/dimenso_slides.pptx`, and `sim/assets/` are committed.
