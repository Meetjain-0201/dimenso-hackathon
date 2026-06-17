# Diagnostics — environment, G1 joint model, data & hand-detection viability

> **Scope:** diagnostics only. No pipeline code was written and no existing source files were modified. Findings here are the reference for designing the real pipeline.
> Date: 2026-06-17. All Python checks ran in the existing `isaac_sim` conda env (`~/miniconda3/envs/isaac_sim/bin/python`, Python 3.11.14) — nothing was installed.

## TL;DR (headline numbers)

| Question | Answer |
|---|---|
| G1 + **Inspire** hand MJCF available? | **No.** Only a Unitree **Dex3-1 (3-finger)** hand exists locally/clonable. Must source/build an Inspire variant. |
| Working MuJoCo render backend | **glfw** (osmesa & egl both failed) |
| Stow pose render correct? | **Yes, after a convention fix** — G1 elbow zero is already a ~90° L-shape; `elbow=0` (not π/2) gives forearm-forward. |
| Best task for hand tracking | **task_04_pasteur_pipette** — 77% frames ≥1 hand, shortest no-hand gap (~3.4 s) |
| IMU↔video timing | Durations match within **0.06–0.10 s**; no clean sync event → start offset unverified (≈2–3 frames) |
| Head joints on G1 | **0** — model has waist (3 DoF) but **no neck/head joint** |
| Full-body pose from egocentric video | **Not recoverable** — only 29% of frames return any pose; lower-body visibility collapses |

---

## 1. Environment

| Component | Status | Version |
|---|---|---|
| OS | ✅ | Ubuntu 22.04.5 LTS |
| CPU | ✅ | Intel Core Ultra 7 255HX, 20 cores |
| RAM | ✅ | 30 GiB |
| GPU | ✅ visible | NVIDIA RTX 5060 Laptop, 8151 MiB, driver 580.159.03 |
| python | ✅ | 3.11.14 (`isaac_sim` conda env) |
| mujoco | ✅ | 3.9.0 |
| mediapipe | ✅ | 0.10.21 |
| opencv-python | ✅ | 4.11.0.86 |
| numpy | ✅ | 1.26.0 |
| scipy | ✅ | 1.15.3 |
| fastapi | ✅ | 0.115.7 |
| uvicorn | ✅ | 0.29.0 |
| node | ✅ | v22.22.1 |
| npm | ✅ | 10.9.4 |

**CPU-path viability (brief is CPU/laptop-only): GOOD.**
- MediaPipe ran entirely on CPU (it initialized GL on `llvmpipe` software renderer, i.e. no GPU acceleration used) and processed 100 frames per task at 1280×720 in ~11 s (~0.11 s/frame). Comfortable for offline dataset building; borderline-but-OK for ~5–10 fps live preview.
- MuJoCo offscreen render works on CPU/GL via `glfw` (see §4).
- A GPU is present but was **not** required for any diagnostic. Note the perception stack lives in `isaac_sim` and was deliberately reused to avoid a heavy install on a near-full disk (root partition ~90% full).

---

## 2. G1 model + Inspire-hand availability

**`mujoco_menagerie`** was not present; cloned (sparse + shallow, `unitree_g1` only, to save disk) to **`~/Dev/mujoco_menagerie`** — *outside* this repo, not committed.

| File | Path |
|---|---|
| `g1.xml` (no hands) | `~/Dev/mujoco_menagerie/unitree_g1/g1.xml` |
| `g1_with_hands.xml` | `~/Dev/mujoco_menagerie/unitree_g1/g1_with_hands.xml` |
| scene wrappers | `scene.xml`, `scene_with_hands.xml` (add floor + light) |

**Which hand does `g1_with_hands.xml` use?** The **Unitree Dex3-1**, a **3-finger** dexterous hand. Confirmed from the XML body/joint names: each hand has `thumb_{0,1,2}`, `index_{0,1}`, `middle_{0,1}` only (7 joints/hand) — **no ring/pinky**. This is *not* the Inspire hand.

**Do we have a G1 + Inspire-hand MJCF?** **No — must source or build one.**
- `mujoco_menagerie/unitree_g1` → Dex3-1 only.
- `robot_descriptions` package → **not installed**.
- Other local G1 assets found, none with Inspire:
  - `~/motion references/unitree_rl_gym-main/.../g1_description/g1_29dof_with_hand.xml` → also Dex3-1 (thumb/index/middle).
  - `~/Dev/Motion-Copilot/cloud/mjcf/g1_holo_compat.xml` → Unitree **rubber_hand** (fixed, non-articulated).
  - `~/Dev/Motion-Copilot/cloud/g1_inspire_config.py` → a GR00T *data*-modality config that *references* `inspire_left/right_hand` state keys, but it is **not** an MJCF/URDF.
- **Candidate sources to obtain an Inspire MJCF (do NOT build yet):** Unitree teleop repos that ship G1 + Inspire assets (`unitreerobotics/avp_teleoperate` / `xr_teleoperate`), `unitree_ros` URDFs (`g1_description` with Inspire variants) convertible to MJCF, and Inspire-hand URDFs used by `dex-retargeting`. Recommended interim: prototype on `g1_with_hands.xml` (Dex3-1) and swap the hand subtree once an Inspire MJCF is sourced.

---

## 3. G1 Joint Map

Model used: **`g1_with_hands.xml`** (the most complete option locally; the Inspire variant does not exist yet — see §2). Loaded in MuJoCo 3.9.0; everything below is read directly from the model, not guessed.

**Totals:** 44 joints, nq=50, nv=49 DoF, 43 actuators, 45 bodies. Keyframes: stand.


#### BASE/OTHER  (1 joints)

| joint | type | parent body | axis (xyz) | range rad | range deg | actuator (ctrlrange) |
|---|---|---|---|---|---|---|
| floating_base_joint | free | pelvis | — | — | — | — |

#### LEFT LEG  (6 joints)

| joint | type | parent body | axis (xyz) | range rad | range deg | actuator (ctrlrange) |
|---|---|---|---|---|---|---|
| left_hip_pitch_joint | hinge | left_hip_pitch_link | (0,1,0) | [-2.531, +2.880] | [-145, +165] | left_hip_pitch_joint [-2.53,2.88] |
| left_hip_roll_joint | hinge | left_hip_roll_link | (1,0,0) | [-0.524, +2.967] | [-30, +170] | left_hip_roll_joint [-0.52,2.97] |
| left_hip_yaw_joint | hinge | left_hip_yaw_link | (0,0,1) | [-2.758, +2.758] | [-158, +158] | left_hip_yaw_joint [-2.76,2.76] |
| left_knee_joint | hinge | left_knee_link | (0,1,0) | [-0.087, +2.880] | [-5, +165] | left_knee_joint [-0.09,2.88] |
| left_ankle_pitch_joint | hinge | left_ankle_pitch_link | (0,1,0) | [-0.873, +0.524] | [-50, +30] | left_ankle_pitch_joint [-0.87,0.52] |
| left_ankle_roll_joint | hinge | left_ankle_roll_link | (1,0,0) | [-0.262, +0.262] | [-15, +15] | left_ankle_roll_joint [-0.26,0.26] |

#### RIGHT LEG  (6 joints)

| joint | type | parent body | axis (xyz) | range rad | range deg | actuator (ctrlrange) |
|---|---|---|---|---|---|---|
| right_hip_pitch_joint | hinge | right_hip_pitch_link | (0,1,0) | [-2.531, +2.880] | [-145, +165] | right_hip_pitch_joint [-2.53,2.88] |
| right_hip_roll_joint | hinge | right_hip_roll_link | (1,0,0) | [-2.967, +0.524] | [-170, +30] | right_hip_roll_joint [-2.97,0.52] |
| right_hip_yaw_joint | hinge | right_hip_yaw_link | (0,0,1) | [-2.758, +2.758] | [-158, +158] | right_hip_yaw_joint [-2.76,2.76] |
| right_knee_joint | hinge | right_knee_link | (0,1,0) | [-0.087, +2.880] | [-5, +165] | right_knee_joint [-0.09,2.88] |
| right_ankle_pitch_joint | hinge | right_ankle_pitch_link | (0,1,0) | [-0.873, +0.524] | [-50, +30] | right_ankle_pitch_joint [-0.87,0.52] |
| right_ankle_roll_joint | hinge | right_ankle_roll_link | (1,0,0) | [-0.262, +0.262] | [-15, +15] | right_ankle_roll_joint [-0.26,0.26] |

#### WAIST/TORSO  (3 joints)

| joint | type | parent body | axis (xyz) | range rad | range deg | actuator (ctrlrange) |
|---|---|---|---|---|---|---|
| waist_yaw_joint | hinge | waist_yaw_link | (0,0,1) | [-2.618, +2.618] | [-150, +150] | waist_yaw_joint [-2.62,2.62] |
| waist_roll_joint | hinge | waist_roll_link | (1,0,0) | [-0.520, +0.520] | [-30, +30] | waist_roll_joint [-0.52,0.52] |
| waist_pitch_joint | hinge | torso_link | (0,1,0) | [-0.520, +0.520] | [-30, +30] | waist_pitch_joint [-0.52,0.52] |

#### LEFT ARM  (7 joints)

| joint | type | parent body | axis (xyz) | range rad | range deg | actuator (ctrlrange) |
|---|---|---|---|---|---|---|
| left_shoulder_pitch_joint | hinge | left_shoulder_pitch_link | (0,1,0) | [-3.089, +2.670] | [-177, +153] | left_shoulder_pitch_joint [-3.09,2.67] |
| left_shoulder_roll_joint | hinge | left_shoulder_roll_link | (1,0,0) | [-1.588, +2.252] | [-91, +129] | left_shoulder_roll_joint [-1.59,2.25] |
| left_shoulder_yaw_joint | hinge | left_shoulder_yaw_link | (0,0,1) | [-2.618, +2.618] | [-150, +150] | left_shoulder_yaw_joint [-2.62,2.62] |
| left_elbow_joint | hinge | left_elbow_link | (0,1,0) | [-1.047, +2.094] | [-60, +120] | left_elbow_joint [-1.05,2.09] |
| left_wrist_roll_joint | hinge | left_wrist_roll_link | (1,0,0) | [-1.972, +1.972] | [-113, +113] | left_wrist_roll_joint [-1.97,1.97] |
| left_wrist_pitch_joint | hinge | left_wrist_pitch_link | (0,1,0) | [-1.614, +1.614] | [-93, +93] | left_wrist_pitch_joint [-1.61,1.61] |
| left_wrist_yaw_joint | hinge | left_wrist_yaw_link | (0,0,1) | [-1.614, +1.614] | [-93, +93] | left_wrist_yaw_joint [-1.61,1.61] |

#### RIGHT ARM  (7 joints)

| joint | type | parent body | axis (xyz) | range rad | range deg | actuator (ctrlrange) |
|---|---|---|---|---|---|---|
| right_shoulder_pitch_joint | hinge | right_shoulder_pitch_link | (0,1,0) | [-3.089, +2.670] | [-177, +153] | right_shoulder_pitch_joint [-3.09,2.67] |
| right_shoulder_roll_joint | hinge | right_shoulder_roll_link | (1,0,0) | [-2.252, +1.588] | [-129, +91] | right_shoulder_roll_joint [-2.25,1.59] |
| right_shoulder_yaw_joint | hinge | right_shoulder_yaw_link | (0,0,1) | [-2.618, +2.618] | [-150, +150] | right_shoulder_yaw_joint [-2.62,2.62] |
| right_elbow_joint | hinge | right_elbow_link | (0,1,0) | [-1.047, +2.094] | [-60, +120] | right_elbow_joint [-1.05,2.09] |
| right_wrist_roll_joint | hinge | right_wrist_roll_link | (1,0,0) | [-1.972, +1.972] | [-113, +113] | right_wrist_roll_joint [-1.97,1.97] |
| right_wrist_pitch_joint | hinge | right_wrist_pitch_link | (0,1,0) | [-1.614, +1.614] | [-93, +93] | right_wrist_pitch_joint [-1.61,1.61] |
| right_wrist_yaw_joint | hinge | right_wrist_yaw_link | (0,0,1) | [-1.614, +1.614] | [-93, +93] | right_wrist_yaw_joint [-1.61,1.61] |

#### LEFT HAND (fingers)  (7 joints)

| joint | type | parent body | axis (xyz) | range rad | range deg | actuator (ctrlrange) |
|---|---|---|---|---|---|---|
| left_hand_thumb_0_joint | hinge | left_hand_thumb_0_link | (0,1,0) | [-1.047, +1.047] | [-60, +60] | left_hand_thumb_0_joint [-1.05,1.05] |
| left_hand_thumb_1_joint | hinge | left_hand_thumb_1_link | (0,0,1) | [-0.724, +1.047] | [-42, +60] | left_hand_thumb_1_joint [-0.72,1.05] |
| left_hand_thumb_2_joint | hinge | left_hand_thumb_2_link | (0,0,1) | [+0.000, +1.745] | [+0, +100] | left_hand_thumb_2_joint [0.00,1.75] |
| left_hand_middle_0_joint | hinge | left_hand_middle_0_link | (0,0,1) | [-1.571, +0.000] | [-90, +0] | left_hand_middle_0_joint [-1.57,0.00] |
| left_hand_middle_1_joint | hinge | left_hand_middle_1_link | (0,0,1) | [-1.745, +0.000] | [-100, +0] | left_hand_middle_1_joint [-1.75,0.00] |
| left_hand_index_0_joint | hinge | left_hand_index_0_link | (0,0,1) | [-1.571, +0.000] | [-90, +0] | left_hand_index_0_joint [-1.57,0.00] |
| left_hand_index_1_joint | hinge | left_hand_index_1_link | (0,0,1) | [-1.745, +0.000] | [-100, +0] | left_hand_index_1_joint [-1.75,0.00] |

#### RIGHT HAND (fingers)  (7 joints)

| joint | type | parent body | axis (xyz) | range rad | range deg | actuator (ctrlrange) |
|---|---|---|---|---|---|---|
| right_hand_thumb_0_joint | hinge | right_hand_thumb_0_link | (0,1,0) | [-1.047, +1.047] | [-60, +60] | right_hand_thumb_0_joint [-1.05,1.05] |
| right_hand_thumb_1_joint | hinge | right_hand_thumb_1_link | (0,0,1) | [-1.047, +0.724] | [-60, +42] | right_hand_thumb_1_joint [-1.05,0.72] |
| right_hand_thumb_2_joint | hinge | right_hand_thumb_2_link | (0,0,1) | [-1.745, +0.000] | [-100, +0] | right_hand_thumb_2_joint [-1.75,0.00] |
| right_hand_middle_0_joint | hinge | right_hand_middle_0_link | (0,0,1) | [+0.000, +1.571] | [+0, +90] | right_hand_middle_0_joint [0.00,1.57] |
| right_hand_middle_1_joint | hinge | right_hand_middle_1_link | (0,0,1) | [+0.000, +1.745] | [+0, +100] | right_hand_middle_1_joint [0.00,1.75] |
| right_hand_index_0_joint | hinge | right_hand_index_0_link | (0,0,1) | [+0.000, +1.571] | [+0, +90] | right_hand_index_0_joint [0.00,1.57] |
| right_hand_index_1_joint | hinge | right_hand_index_1_link | (0,0,1) | [+0.000, +1.745] | [+0, +100] | right_hand_index_1_joint [0.00,1.75] |

**Stand keyframe (`stand`) — non-zero arm/leg values of note:** `*_shoulder_pitch=0.20`, `left_shoulder_roll=+0.20`/`right=-0.20`, `*_elbow=1.28`, `*_hip_pitch≈-0.10`, `*_knee≈0.30`, `*_ankle_pitch≈-0.20` (legs slightly flexed standing stance). All finger joints `0` (open). Full vector stored in the model's `key_qpos`.

**Critical structural findings:**
- **No HEAD/neck joint exists.** Groups present: 2 legs (6 each), waist/torso (3: yaw/roll/pitch), 2 arms (7 each), 2 Dex3 hands (7 each), plus the floating base. ⇒ The design's "HEAD driven by head IMU → head joints" has **no actuator to map onto**; head-IMU orientation can only drive the **3 waist joints**. Flag for design.
- Per-arm DoF = 7 (shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw). Human arm from MediaPipe gives ~shoulder+elbow+wrist ≈ similar count, so arm IK is tractable; the **hand** DoF mismatch (human 21 landmarks → Dex3 7 / Inspire 6–12) is the harder mapping.

---

## 4. Render backend

Offscreen 640×480 RGB render of `scene_with_hands.xml` after `mj_forward`, backends tried in order:

| `MUJOCO_GL` | Result |
|---|---|
| **glfw** | ✅ **frame produced** (mean px 68.4) → use this |
| osmesa | ❌ failed (`glGetError` on None — OSMesa lib not available) |
| egl | ❌ failed (`EGLError`) |

**Use `MUJOCO_GL=glfw`.** This fixes the streaming/render path. Note: the system is **Wayland** (`DISPLAY=:1`, XWayland present); the interactive `mujoco.viewer` works but glfw-on-Wayland reports a HiDPI framebuffer smaller than the window (scene renders into a sub-region) — cosmetic, forcing the X11/XWayland backend or 100% display scale resolves it. → `docs/diag_g1_render.png`

---

## 5. Pose-convention verification (home + stow)

- **Home / `stand` keyframe** rendered correctly — upright G1, arms slightly forward. → `docs/diag_pose_home.png`
- **Stow pose** (per spec: upper arm straight down, forearm forward horizontal, ~90° elbow, fingers open):

  **⚠️ Convention surprise found and corrected.** The G1 **elbow zero is already a ~90° L-shape**, not full extension. Sweeping the elbow with `shoulder_pitch=0`:
  | elbow (rad) | forearm direction (world) |
  |---|---|
  | 0.0 | `[+1.00, 0, -0.05]` → **forward horizontal** ✅ |
  | +0.5 | `[+0.85, 0, -0.53]` |
  | +1.0 | `[+0.49, 0, -0.87]` |
  | +1.571 | `[-0.05, 0, -1.00]` → points **down** (over-flexed) |

  So the naive `elbow=π/2` is **wrong** (it curls the forearm back down). Correct stow values (both arms): `shoulder_pitch=shoulder_roll=shoulder_yaw=0`, **`elbow=0`**, `wrist_*=0`, all finger joints `=0`. Verified geometry: upper arm `[+0.09,+0.03,-1.00]` (down), forearm `[+1.00,±0.01,-0.05]` (forward), **elbow angle 82°** (≈90°; residual from a slight upper-arm forward lean). Render looks correct. → `docs/diag_pose_stow.png`
  - **Sign notes for the builder:** `elbow` axis +Y, positive flexes forearm *downward* from the L-rest. `shoulder_pitch` axis +Y, positive swings upper arm *backward/up*. `shoulder_roll` is mirrored L/R (left range `[-1.59,+2.25]`, right `[-2.25,+1.59]`) — abduction sign is opposite per side. Finger flexion sign is also mirrored L/R (left index/middle ranges `[-1.57,0]`, right `[0,+1.57]`): **0 = open** on both, but the curl direction's sign flips between hands.

---

## 6. Data inventory

`~/Dev/dimenso/data/` exists and is gitignored (`.gitignore` line `data/` confirmed; video never committed). Layout: `data/<task>/<task_id>/<VID_...>/{base.mp4, imu.json}`.

| Task | Recordings |
|---|---|
| task_01_sample_weighing | 6 |
| task_02_wellplate_movement | 7 |
| task_03_pipetting | 6 |
| task_04_pasteur_pipette | 4 |
| **Total** | **23** |

---

## 7. Video / IMU stats + timing offset

Deep-inspected the first recording of **each** task. Video is uniformly **1920×1080 @ ~29.6 fps**. IMU JSON schema: `{version, sampleCount, samplingRateHz, startTimeNs, durationMs, samples}` where **`samples` is `N×7` = `[timestamp_ms, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]`** (accel in m/s² — rest magnitude ≈9.8 confirmed; gyro in rad/s). `durationMs` == the **last** sample's timestamp (device-uptime ms), **not** a duration; `startTimeNs` is a *separate* ns clock.

| Task (rec 1) | Video | IMU | IMU rate (dt) | video−IMU dur |
|---|---|---|---|---|
| task_01 | 1299 f / 43.87 s | 4394 samp / 43.93 s | 100.0 Hz (10.0 ms, jitter 4–16) | **−0.06 s** |
| task_02 | 965 f / 32.60 s | 3271 samp / 32.68 s | 100.1 Hz (9.99 ms, jitter 1–18) | **−0.08 s** |
| task_03 | 980 f / 33.10 s | 3321 samp / 33.20 s | 100.0 Hz (10.0 ms, jitter 3–17) | **−0.10 s** |
| task_04 | 1431 f / 48.34 s | 4849 samp / 48.42 s | 100.1 Hz (9.99 ms, jitter 1–18) | **−0.08 s** |

Example raw IMU timestamps (task_02, ms): first5 `[1160274, 1160287, 1160294, 1160303, 1160313]`, last5 `[1192913, 1192923, 1192933, 1192946, 1192953]`.

**Timing alignment (measured only, not fixed):**
- The IMU stream is consistently **~0.06–0.10 s longer** than the video (≈2–3 video frames), i.e. video is slightly truncated relative to IMU.
- IMU sampling is stable ~100 Hz but **not** uniform (per-sample dt ranges 1–18 ms) ⇒ the pipeline must **resample IMU to video-frame timestamps**, not assume a fixed 10 ms grid.
- There is **no shared absolute clock and no sync event** in the files: IMU timestamps are device-uptime ms; the video carries no per-frame absolute timestamp here (only a wall-clock string in the folder name). So the precise start **offset between the two clocks is unverifiable from the data** — durations matching to ~0.1 s suggests near-synchronous capture, but a ~2–3 frame start offset cannot be ruled out. **Do not assume a clean zero offset.** A short shared motion event (e.g. an initial head nod visible in both video and gyro) would be needed to pin it down.

---

## 8. Hand-detection rates + task recommendation

MediaPipe Hands (`static_image_mode`, `max_num_hands=2`, conf≥0.5) over 100 frames evenly spread across the first recording of each task (frames resized to 1280×720). Annotated sample per task: `docs/diag_hands_<task>.png`.

| Task | ≥1 hand | 2 hands | mean conf | longest no-hand run |
|---|---|---|---|---|
| task_01_sample_weighing | 34.0% | 0.0% | 0.949 | 14 samp (~6.1 s) |
| task_02_wellplate_movement | 42.0% | 21.0% | 0.951 | 18 samp (~5.9 s) |
| task_03_pipetting | 49.0% | 21.0% | 0.953 | 16 samp (~5.3 s) |
| **task_04_pasteur_pipette** | **77.0%** | 18.0% | **0.976** | **7 samp (~3.4 s)** |

**Recommendation: build/validate on `task_04_pasteur_pipette` first.** It has by far the highest hand-visibility (77% vs 34–49%), the highest detection confidence, and the shortest blackout gap. When a hand is detected confidence is high everywhere (~0.95+), so the limiter is *presence in frame*, not detector quality.
**Stow/hold sizing:** worst-case no-hand gaps reach **~6 s** (task_01/02) — the per-arm stow/hold logic must hold gracefully for multi-second dropouts, with hysteresis to avoid flicker between stow and tracking.

---

## 9. Full-body Pose sanity check

MediaPipe Pose over the same 100-frame sample of **task_04**:

| Metric | Value |
|---|---|
| Frames returning any pose result | **29 / 100 (29%)** |
| Mean visibility — shoulders (11,12) | 0.994 |
| Mean visibility — hips (23,24) | 0.964 |
| Mean visibility — knees (25,26) | 0.460 |
| Mean visibility — ankles (27,28) | 0.225 |

**Interpretation:** A forward head-cam cannot see the wearer's own torso/legs, so Pose returns *nothing* 71% of the time. When it does fire, the high shoulder/hip "visibility" is **MediaPipe hallucinating** a plausible upper torso it cannot actually observe (not trustworthy ground truth), while the lower body honestly degrades (knees 0.46 → ankles 0.22). **Conclusion: full-body pose is NOT recoverable from egocentric video** — confirming the locked design (legs frozen in a standing stance; torso/waist driven by head IMU; arms/hands from hand tracking).

---

## 10. Open questions / risks

1. **No Inspire-hand MJCF yet** (§2). Decide: source one (Unitree teleop / `unitree_ros` URDF→MJCF) vs prototype on Dex3-1 and swap later. Finger-retarget mapping differs (Dex3 7-DoF, 3-finger vs Inspire 6-DoF/finger-coupled, 5-finger).
2. **No head joint on G1** (§3). Head-IMU orientation must drive the **3 waist joints** (yaw/roll/pitch, each limited to ±30° for roll/pitch, ±150° yaw). Head pitch/roll beyond ±30° will saturate the waist — decide clamping/scaling policy.
3. **IMU→video offset unverifiable** (§7). No sync event in the data. Risk: a 2–3 frame misalignment between head motion and hand motion. Mitigation: detect a shared motion event, or accept ~0 offset and document the uncertainty.
4. **IMU drift.** Pure accel+gyro integration for head *translation* will drift; for the design only head *orientation* → waist is needed, so prefer a gravity-referenced orientation filter (complementary/Madgwick) and avoid double-integrating accel for position.
5. **Hand presence is the bottleneck** (§8), not detector quality. Multi-second dropouts (up to ~6 s) require robust stow/hold + hysteresis. Even task_04 is "2 hands" only 18% of the time → bimanual sub-tasks will often be one-handed in the data.
6. **CPU-only live streaming** (§1): offline is comfortable (~0.11 s/frame); real-time >10 fps with Hands + IK + render on CPU is tight — may need frame-skipping or lower preview resolution.
7. **Render path is Wayland-quirky** (§4): glfw works headless-ish but the interactive viewer mis-scales on Wayland HiDPI; standardize on `MUJOCO_GL=glfw` and document the X11 fallback for live viewing.

### Artifacts referenced
`docs/diag_g1_render.png` · `docs/diag_pose_home.png` · `docs/diag_pose_stow.png` · `docs/diag_hands_task_01_sample_weighing.png` · `docs/diag_hands_task_02_wellplate_movement.png` · `docs/diag_hands_task_03_pipetting.png` · `docs/diag_hands_task_04_pasteur_pipette.png`
