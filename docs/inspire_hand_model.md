# G1 + five-finger Inspire hand — MuJoCo model

**VERDICT — Inspire ready: YES** — both five-finger Inspire DFQ hands graft cleanly onto the menagerie G1 wrists, the model loads in MuJoCo, and open / mid-curl / fist render correctly with symmetric, inward finger curl on both hands.

> Status: **asset only, ready but NOT wired into any pipeline.** The motion pipeline is built on the Dex3 hand separately; this model is staged for later use.

## Source

| Piece | Source | Commit |
|---|---|---|
| G1 base (arms+wrists, no hands) | `mujoco_menagerie/unitree_g1/g1.xml` | `accb6df40a9a1d1e49eff88157f6818b63a49335` |
| Inspire DFQ hands (L+R URDF + 26 STL meshes) | `unitreerobotics/unitree_ros` → `robots/g1_description/inspire_hand/DFQ_{left,right}_hand.urdf` | `7c40519e02d7dd16c06b25fe3fa3b67fdeb7cd74` |

Both repos are external (under `~/Dev/`), not vendored here. Only the meshes used were copied into the repo: G1 link STLs → `sim/assets/g1/` (35 files), Inspire STLs → `sim/assets/inspire/` (26 files: `{L,R}_hand_base_link` + `Link11..22_{L,R}`).

Outputs: `sim/assets/g1_inspire.xml` (combined model), `sim/assets/scene_g1_inspire.xml` (+ ground plane + light).

## Base + build

- **Base:** menagerie `g1.xml` — 29-actuated-DoF G1 with full arms **including wrist roll/pitch/yaw** and **no articulated hand** (it only drew a fixed `rubber_hand` visual geom, which was removed). The `stand` keyframe and all leg/waist/arm position actuators are kept intact.
- **Graft:** each Inspire hand attached to the corresponding `*_wrist_yaw_link` body. The Inspire DFQ URDF root link is literally named `left_wrist_yaw_link` / `right_wrist_yaw_link` — identical to the menagerie G1 — so the URDF mount transform applies directly.
- **Both chiralities** come straight from `unitree_ros` (separate L and R URDFs); **no manual mirroring was needed or done** — the L/R mirror is baked into the URDF joint axes and origins (e.g. left thumb-pitch axis `(0,0,-1)` vs right `(0,0,1)`).

### Mount transforms (wrist → hand base), from the URDF
| Hand | Parent body | pos (m) | rpy (rad) | quat (wxyz) |
|---|---|---|---|---|
| Left  | `left_wrist_yaw_link`  | `0.0415 0 0` | `0 0 +π/2` | `0.7071 0 0 0.7071` |
| Right | `right_wrist_yaw_link` | `0.0415 0 0` | `π 0 −π/2` | `0 −0.7071 0.7071 0` |

These are exposed as the `pos`/`quat` of the `L_hand_base_link` / `R_hand_base_link` bodies at the top of each grafted hand block in `g1_inspire.xml` for easy tuning.

## Totals (DoF)

| | Count |
|---|---|
| Actuated DoF (`nu`) | **53** = 29 (G1: 12 leg + 3 waist + 14 arm) + 24 (12 fingers × 2 hands) |
| `nq` | 60 (53 hinge + 7 free-floating base) |
| `nv` | 59 |
| Joints (`njnt`) | 54 (1 free + 53 hinge) |

Each Inspire hand = **12 revolute joints**: thumb ×4 (proximal_yaw, proximal_pitch, intermediate, distal), index/middle/ring/pinky ×2 each (proximal, intermediate).

## Joint map (grouped)

Axes are in each joint's local frame; ranges read from the loaded model. Every joint below has one independent `position` actuator (`kp=2`) whose `ctrlrange` equals the joint range.

### LEFT ARM (7)
| joint | axis | range rad | range deg | ctrl |
|---|---|---|---|---|
| left_shoulder_pitch_joint | (0,1,0) | [-3.089,+2.670] | [-177,+153] | [-3.089, 2.670] |
| left_shoulder_roll_joint | (1,0,0) | [-1.588,+2.252] | [-91,+129] | [-1.588, 2.252] |
| left_shoulder_yaw_joint | (0,0,1) | [-2.618,+2.618] | [-150,+150] | [-2.618, 2.618] |
| left_elbow_joint | (0,1,0) | [-1.047,+2.094] | [-60,+120] | [-1.047, 2.094] |
| left_wrist_roll_joint | (1,0,0) | [-1.972,+1.972] | [-113,+113] | [-1.972, 1.972] |
| left_wrist_pitch_joint | (0,1,0) | [-1.614,+1.614] | [-93,+93] | [-1.614, 1.614] |
| left_wrist_yaw_joint | (0,0,1) | [-1.614,+1.614] | [-93,+93] | [-1.614, 1.614] |

### RIGHT ARM (7)
| joint | axis | range rad | range deg | ctrl |
|---|---|---|---|---|
| right_shoulder_pitch_joint | (0,1,0) | [-3.089,+2.670] | [-177,+153] | [-3.089, 2.670] |
| right_shoulder_roll_joint | (1,0,0) | [-2.252,+1.588] | [-129,+91] | [-2.252, 1.588] |
| right_shoulder_yaw_joint | (0,0,1) | [-2.618,+2.618] | [-150,+150] | [-2.618, 2.618] |
| right_elbow_joint | (0,1,0) | [-1.047,+2.094] | [-60,+120] | [-1.047, 2.094] |
| right_wrist_roll_joint | (1,0,0) | [-1.972,+1.972] | [-113,+113] | [-1.972, 1.972] |
| right_wrist_pitch_joint | (0,1,0) | [-1.614,+1.614] | [-93,+93] | [-1.614, 1.614] |
| right_wrist_yaw_joint | (0,0,1) | [-1.614,+1.614] | [-93,+93] | [-1.614, 1.614] |

### WAIST (3)
| joint | axis | range rad | range deg | ctrl |
|---|---|---|---|---|
| waist_yaw_joint | (0,0,1) | [-2.618,+2.618] | [-150,+150] | [-2.618, 2.618] |
| waist_roll_joint | (1,0,0) | [-0.520,+0.520] | [-30,+30] | [-0.520, 0.520] |
| waist_pitch_joint | (0,1,0) | [-0.520,+0.520] | [-30,+30] | [-0.520, 0.520] |

*(Legs — 12 joints — are unchanged from the base; omitted here for brevity, present in the model.)*

### LEFT HAND fingers (12)
| joint | axis | range rad | range deg | ctrl |
|---|---|---|---|---|
| L_thumb_proximal_yaw_joint | (0,0,1) | [-0.100,+1.300] | [-6,+74] | [-0.100, 1.300] |
| L_thumb_proximal_pitch_joint | (0,0,-1) | [-0.100,+0.600] | [-6,+34] | [-0.100, 0.600] |
| L_thumb_intermediate_joint | (0,0,-1) | [0.000,+0.800] | [0,+46] | [0.000, 0.800] |
| L_thumb_distal_joint | (0,0,-1) | [0.000,+1.200] | [0,+69] | [0.000, 1.200] |
| L_index_proximal_joint | (0,0,-1) | [0.000,+1.700] | [0,+97] | [0.000, 1.700] |
| L_index_intermediate_joint | (0,0,-1) | [0.000,+1.700] | [0,+97] | [0.000, 1.700] |
| L_middle_proximal_joint | (0,0,-1) | [0.000,+1.700] | [0,+97] | [0.000, 1.700] |
| L_middle_intermediate_joint | (0,0,-1) | [0.000,+1.700] | [0,+97] | [0.000, 1.700] |
| L_ring_proximal_joint | (0,0,-1) | [0.000,+1.700] | [0,+97] | [0.000, 1.700] |
| L_ring_intermediate_joint | (0,0,-1) | [0.000,+1.700] | [0,+97] | [0.000, 1.700] |
| L_pinky_proximal_joint | (0,0,-1) | [0.000,+1.700] | [0,+97] | [0.000, 1.700] |
| L_pinky_intermediate_joint | (0,0,-1) | [0.000,+1.700] | [0,+97] | [0.000, 1.700] |

### RIGHT HAND fingers (12)
Identical joint names with `R_` prefix and **mirrored axes** (thumb-pitch/intermediate/distal and all proximal/intermediate axes flip sign vs left: e.g. `R_index_proximal_joint` axis `(0,0,1)` vs left `(0,0,-1)`). Ranges are identical to the left hand. Same independent `position` actuators, ctrl = joint range.

## Poses & convention verification

Rendered with `MUJOCO_GL=glfw`, offscreen 640×480, from the `stand` keyframe (legs in standing stance) + arms in stow (`shoulders=0, elbow=0, wrists=0`).

- **`sim/assets/diag_inspire_open.png`** — fingers relaxed/open: all finger joints at their **lower limit** (0 rad for the 8 flexion joints; thumb yaw/pitch at −0.1 rad), i.e. fully extended. This is also the default in the `stand` keyframe (finger DoF padded to 0). ✅ looks correct.
- **`sim/assets/diag_inspire_mid.png`** — mid-curl: each finger joint at 50% of its range. ✅ looks correct.
- **`sim/assets/diag_inspire_fist.png`** — full fist: each finger joint at its upper limit. ✅ looks correct — both hands close into compact fists.

**Curl direction & mirror — verified, no backwards signs.** Setting the same positive curl fraction on left and right finger joints flexes **both** hands inward symmetrically (the sign flip lives in the mirrored axes, so equal commands give a mirror-correct grasp). Numeric check: left and right index `intermediate→palm` distance drops identically from **0.172 m (open) → 0.137 m (fist)** — inward curl, perfectly symmetric. No joint rendered hyperextended/backwards.

## Known simplification — underactuation

The real Inspire RH56-DFQ hand is **underactuated: ~6 motors drive its 12 finger joints** (each finger's proximal+intermediate flex together via a linkage; the thumb's distal segments are coupled). This model exposes **all 12 joints per hand as independent position actuators** (24 total) — a deliberate simplification. For our use (recovering open/close + finger curl from egocentric video and replaying it), independent joints are sufficient and simpler to drive; we are **not** modeling the 6→12 mechanical coupling. If physically-faithful actuation is later required, add MuJoCo `<equality>`/tendon couplings to tie each finger's two joints (and reduce the actuator count to ~6/hand).

## How to load
```python
import mujoco                       # MUJOCO_GL=glfw
m = mujoco.MjModel.from_xml_path("sim/assets/scene_g1_inspire.xml")
d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0)   # 'stand'
```
Artifacts: `diag_inspire_open.png`, `diag_inspire_mid.png`, `diag_inspire_fist.png` (in `sim/assets/`).
