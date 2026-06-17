# Dimenso Hackathon Report

Egocentric smart-glasses RGB video + head IMU → recovered human motion →
retargeted onto a Unitree **G1 with five-finger Inspire hands** → replayed in
MuJoCo. **Motion translation only** (no object interaction / manipulation
planning). Task used: `task_04_pasteur_pipette`.

## Problem framing & assumptions
- **In:** one recording = `base.mp4` (egocentric head-cam RGB, 1920×1080 @ ~29.6 fps) + `imu.json` (head IMU, ~100 Hz, `[ts_ms, accel xyz, gyro xyz]`). **Out:** a per-frame G1+Inspire joint trajectory + a MuJoCo replay.
- **Full-body pose is NOT recoverable** from a forward head-cam (diagnostics: only 29% of frames return any MediaPipe Pose; lower-body visibility collapses). So the design is reframed: **legs frozen** at the stand keyframe; **torso/waist** driven by the head IMU; **arms + hands** driven by MediaPipe hand tracking. The G1 has **no neck joint**, so head orientation drives the **3 waist joints only**.
- **RGB-only, no depth** → the 3D wrist target is **approximated** (documented below and in code).
- **No shared video↔IMU clock** → we assume IMU sample 0 = video frame 0 (durations agree to ~0.12 s).

## Data pipeline (the centerpiece)
`data_pipeline/run_offline.py` chains five stages (all joint/actuator names read
from the model via MuJoCo — nothing hardcoded):

1. **`pose_extract.py`** — decode video (real per-frame timestamps), MediaPipe Hands per frame (21 landmarks + handedness + confidence). A **One Euro filter** (Casiez et al., CHI 2012) de-jitters every landmark coordinate; detections are confidence-gated. Derives per-finger **curl** (path-length ratio) and thumb-index **pinch**. Handedness is flipped for the non-mirrored egocentric view.
2. **`imu_sync.py`** — **resamples the jittery IMU onto the video frame timestamps** (not a fixed 10 ms grid); recovers head orientation with a complementary filter (gravity-from-accel for roll/pitch, gyro integration, yaw leak for drift); outputs a per-frame **waist-lean target relative to the recording's first frame**, clamped ±30°.
3. **`stabilize.py`** — lifts the 2D wrist landmark to an **approximate 3D workspace target** on a table plane (image-u → lateral y, image-v → forward x, apparent hand scale + MediaPipe relative z → height/depth). All constants are commented tunables. **Heavily approximate — not a calibrated reconstruction.**
4. **`method/retarget.py`** — **DLS IK** (MuJoCo body Jacobians) drives each present wrist to its 3D target; the **3 waist joints are shared IK DoF** with a **soft bias** to the IMU lean, so reach and lean cooperate. Both arms solved jointly. Finger curl → Inspire finger joints (independent, no coupling). **Per-arm stow** (upper arm down, forearm forward ~90°, fingers open) when a hand isn't seen, with a hold-last-valid window and per-joint **rate limiting** for smooth eases.
5. **`build_dataset.py`** — writes **`outputs/task04_dataset.npz`** + a `.schema.json` documenting every array and approximation. This file is the deliverable; the replay consumes its `qpos` trajectory.

## Method (perception → retargeting → imitation learning → simulation)
Perception (MediaPipe Hands + IMU) → approximate 3D targets → DLS IK + finger/lean
mapping → full G1+Inspire `qpos` per frame → MuJoCo kinematic replay
(`sim/replay_g1.py`: offscreen mp4 + live passive viewer; legs frozen at stand).
**Imitation learning is a sketch only — none was trained** (see `method/policy_sketch.md`); the dataset is structured so behavior cloning *could* consume it later.

## Validation
Recording `task_04_pasteur_pipette` (first rec), 1431 frames @ 29.6 fps:
- **Hand-detection coverage:** right **76.5%**, left **26.5%**, **any-hand 79.6%** → **79.6% of frames use real tracking, 20.4% fall back to stow.**
- **IMU↔video:** durations 48.42 s vs 48.30 s (~0.12 s slack, consistent with the documented sync assumption). Waist lean exercised the full ±30° clamp on roll, ≈0..−30° on pitch.
- **Replay renders:** `outputs/replay_task04.mp4` ✅ and a frame-aligned side-by-side `outputs/compare_task04.mp4` ✅ (`report/figures/compare_frame_*.png`).
- **Live viewer:** `--live` plays the trajectory in real time in a passive MuJoCo viewer ✅ (visually spot-checked against the source video).
- Sanity plots (`notebooks/01_explore.ipynb` → `report/figures/plot_*.png`): wrist trajectory, hand-present timeline, waist-lean vs IMU.

**Measured vs approximate:** coverage, timing, joint trajectories, render success are *measured*. Wrist **depth/3D position** and the **video↔IMU offset** are *approximated* and clearly marked.

## Feasibility & cost proposal
Runs **CPU-only** (no GPU needed): MediaPipe ~0.1 s/frame; IK + render negligible; full recording processes offline in ~3–4 min on the dev laptop. The engine is offline by design — perception is too slow for 30 fps live, so we precompute the trajectory once and replay it in real time. Scaling to many recordings is embarrassingly parallel. No paid services used.

## Limitations & next steps
**Known weak spots (v1 — needs tuning):**
- **Depth is guessed** (RGB-only): reach *direction* tracks well, absolute forward/height is approximate — the most visible source of error. A second camera or a learned monocular-depth prior would fix this.
- **Left arm mostly stowed** (26.5% coverage) — bimanual sub-tasks render one-handed. Better hand re-acquisition / wider FoV would help.
- **IMU yaw drifts** (leaked, no magnetometer); **video↔IMU offset** unverified (~2–3 frames).
- Waist→lean and finger curl→joint gains are hand-tuned, not calibrated.
- Inspire fingers modeled as **independent joints** (real hand is ~6-motor underactuated) — deliberate simplification.

**Next:** calibrate the workspace mapping against a known scene; add a depth/scale prior; detect a shared video/IMU sync event; tune IK weights & rate limits; (later) train a BC policy on the dataset.

## References
- MediaPipe Hands — Google. One Euro filter — Casiez, Roussel, Vogel, CHI 2012, https://gery.casiez.net/1euro/
- G1 base: `mujoco_menagerie/unitree_g1` (`accb6df`). Inspire hands: `unitreerobotics/unitree_ros` `g1_description/inspire_hand` DFQ (`7c40519`). MuJoCo, Google DeepMind.
- See also `docs/diagnostics.md`, `docs/inspire_hand_model.md`.
