"""Assemble + write the per-frame robot-usable dataset (the deliverable).

Pipeline stage 5. Collects the outputs of pose/imu/stabilize/retarget into a
single documented dataset: outputs/<name>_dataset.npz plus a JSON sidecar that
documents every field (name, shape, units, frame, derivation). The .npz is
gitignored, so the schema.json is the committed contract. The replay
(sim/replay_g1.py) consumes the `qpos` trajectory directly.

Everything is per video frame and time-aligned to the video frame timestamps.
"""
from __future__ import annotations
import json
import pathlib
import numpy as np


def actuated_columns(m):
    """Names + qpos addresses of the 53 actuated (hinge) joints, free base excluded."""
    import mujoco
    names, qadr = [], []
    for i in range(m.njnt):
        if int(m.jnt_type[i]) == int(mujoco.mjtJoint.mjJNT_HINGE):
            names.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i))
            qadr.append(int(m.jnt_qposadr[i]))
    return names, qadr


def assemble(pose, targets, lean, qpos_traj, imu_meta, joint_targets, act_names, model_meta):
    """Pack everything into a flat dict of arrays (one row per video frame)."""
    T = pose["n_frames"]
    sl = slice(0, T)
    data = {
        "frame_index": np.arange(T),
        "timestamps": pose["timestamps"][sl],
        "fps": np.array(pose["fps"]),
        "image_width": np.array(pose["width"]),
        "image_height": np.array(pose["height"]),
        # ── robot motion ──
        "qpos": qpos_traj[sl],                                   # (T, nq) full incl free base
        "joint_targets": joint_targets[sl],                      # (T, 53) actuated only, named
        "waist_lean": lean[sl],                                  # (T,3) [roll,pitch,yaw] rad (clamped IK input)
        # ── IMU layer ──
        "imu_accel": imu_meta["accel"][sl],                      # (T,3) m/s² resampled to frame times
        "imu_gyro": imu_meta["gyro"][sl],                        # (T,3) rad/s resampled
        "head_roll": imu_meta["head_roll"][sl],                  # (T,) recovered head orientation,
        "head_pitch": imu_meta["head_pitch"][sl],                #       relative to frame 0, UNclamped (rad)
        "head_yaw": imu_meta["head_yaw"][sl],
    }
    if "waist_yaw_joint" in act_names:
        data["waist_yaw_applied"] = joint_targets[sl, act_names.index("waist_yaw_joint")]
    # ── perception layer, per arm ──
    for s in ("left", "right"):
        d = pose[s]
        data[f"landmarks_{s}"] = d["landmarks"][sl]              # (T,21,3) normalized x,y,z (RAW MediaPipe)
        data[f"present_{s}"] = d["present"][sl]                  # (T,) bool — real tracking
        data[f"stow_{s}"] = ~d["present"][sl]                    # (T,) bool — arm stowed (no detection)
        data[f"conf_{s}"] = d["conf"][sl]
        data[f"curl_{s}"] = d["curl"][sl]                        # (T,5)
        data[f"thumb_dists_{s}"] = d.get("thumb_dists", np.full((T, 4), np.nan))[sl]  # (T,4)
        data[f"pinch_{s}"] = d["pinch"][sl]                      # (T,)
        data[f"wrist_uv_{s}"] = d["wrist_uv"][sl]                # (T,2) normalized
        data[f"wrist_target_{s}"] = targets[s][sl]              # (T,3) approx 3D robot/table frame
    return data


def save(data, qpos_layout, act_names, model_meta, out_npz):
    out_npz = pathlib.Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **data,
                        qpos_joint_names=np.array(qpos_layout["joint_names"]),
                        actuated_joint_names=np.array(act_names))
    schema = {
        "file": out_npz.name,
        "frames": int(len(data["timestamps"])),
        "fps": float(data["fps"]),
        "frame_alignment": "every array is per video frame, indexed by frame_index / timestamps "
                            "(video clock zeroed at frame 0). IMU is resampled onto these times.",
        "model": model_meta,
        "arrays": {
            "frame_index": {"shape": "(T,)", "units": "—", "desc": "0..T-1 video frame index"},
            "timestamps": {"shape": "(T,)", "units": "s", "frame": "video clock (zeroed at frame 0)",
                           "desc": "per-frame timestamp from the decoder"},
            "fps": {"shape": "scalar", "units": "Hz", "desc": "source video frame rate"},
            "image_width/height": {"shape": "scalar", "units": "px", "desc": "source resolution; "
                                   "image-pixel landmark coords = normalized × (width,height)"},
            "qpos": {"shape": "(T, nq=60)", "units": "rad (+7 free-base pos/quat)",
                     "frame": "MuJoCo model", "desc": "full replay-ready joint targets; columns named "
                     "in qpos_joint_names (free base first, then 53 hinges). Legs frozen at stand; "
                     "waist_yaw+arms from DLS IK; waist_roll/pitch locked 0; fingers from retarget."},
            "joint_targets": {"shape": "(T, 53)", "units": "rad", "frame": "MuJoCo model",
                              "desc": "the 53 ACTUATED joint angles only, columns named in "
                              "actuated_joint_names (self-describing)."},
            "actuated_joint_names": {"shape": "(53,)", "desc": "ordered names for joint_targets columns "
                                     "(leg×12, waist_yaw/roll/pitch, arm×14, Inspire fingers×24)."},
            "qpos_joint_names": {"shape": "(54,)", "desc": "joint names in qpos order (free base first)."},
            "waist_lean": {"shape": "(T,3)", "units": "rad", "frame": "robot",
                           "desc": "[roll,pitch,yaw] head-IMU lean target (clamped ±30°); the IK input. "
                           "Note: roll/pitch are now LOCKED to 0 by the tuned retargeter — only yaw is used."},
            "waist_yaw_applied": {"shape": "(T,)", "units": "rad", "desc": "the waist_yaw actually "
                                  "commanded (from joint_targets) — what the torso turn ended up being."},
            "imu_accel": {"shape": "(T,3)", "units": "m/s²", "frame": "head IMU",
                          "desc": "accelerometer resampled onto frame times (not a fixed 10 ms grid)."},
            "imu_gyro": {"shape": "(T,3)", "units": "rad/s", "frame": "head IMU",
                         "desc": "gyroscope resampled onto frame times."},
            "head_roll/head_pitch/head_yaw": {"shape": "(T,)", "units": "rad",
                "desc": "recovered head orientation (complementary filter: accel-gravity roll/pitch + "
                "gyro integration; yaw leaked), RELATIVE to frame 0, UNclamped. (waist_lean is the "
                "clamped form actually fed to IK.)"},
            "landmarks_{left,right}": {"shape": "(T,21,3)", "units": "normalized [0,1] x,y + relative z",
                "frame": "image", "desc": "RAW MediaPipe Hands 21 landmarks (One-Euro filtered). "
                "Pixel coords = x·width, y·height. NaN when that hand is absent."},
            "present_{left,right}": {"shape": "(T,)", "units": "bool",
                "desc": "hand detected above confidence gate = REAL tracking this frame for that arm."},
            "stow_{left,right}": {"shape": "(T,)", "units": "bool",
                "desc": "= ~present: arm is stowed (no detection). The retargeter holds the last valid "
                "pose for a few frames before easing fully to the arm-down stow."},
            "conf_{left,right}": {"shape": "(T,)", "desc": "MediaPipe handedness/detection score."},
            "curl_{left,right}": {"shape": "(T,5)", "units": "[0,1]",
                "desc": "per-finger curl [thumb,index,middle,ring,pinky] (path-length ratio)."},
            "thumb_dists_{left,right}": {"shape": "(T,4)", "units": "normalized by hand scale",
                "desc": "thumb-tip↔fingertip distance for [index,middle,ring,pinky] — the descriptor "
                "the tuned retarget uses to flex the thumb into pinches."},
            "pinch_{left,right}": {"shape": "(T,)", "units": "normalized image", "desc": "thumb-tip↔"
                "index-tip distance (drives thumb opposition/yaw)."},
            "wrist_uv_{left,right}": {"shape": "(T,2)", "units": "normalized image",
                "desc": "filtered wrist (landmark 0) image coords."},
            "wrist_target_{left,right}": {"shape": "(T,3)", "units": "m", "frame": "robot/table",
                "desc": "APPROXIMATE 3D wrist target. X/Z from table-plane + hand-scale depth proxy; "
                "Y (lateral) tracks the real detected wrist separation (tuned). NOT metric depth."},
        },
        "handedness_note": "MediaPipe labels handedness assuming a mirrored selfie image; for the "
                           "non-mirrored egocentric cam we FLIP it, so 'left'/'right' here = the "
                           "demonstrator's (and robot's) actual side. Raw flipped label is encoded by "
                           "which arm's arrays are populated (not stored separately).",
        "approximations": [
            "DEPTH: wrist 3D X/Z approximated from image position + hand scale + MediaPipe z. Not metric.",
            "IMU-VIDEO SYNC: no shared clock; IMU sample 0 assumed == video frame 0 (~2-3 frame unknown).",
            "FINGERS: 12 independent joints/hand (real Inspire is ~6-motor underactuated).",
            "WAIST: head orientation relative to frame 0; torso kept upright (roll/pitch=0), yaw-only.",
        ],
    }
    out_json = out_npz.with_suffix(".schema.json")
    out_json.write_text(json.dumps(schema, indent=2))
    return out_npz, out_json


def qpos_layout_from_model(m):
    import mujoco
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) or f"j{i}" for i in range(m.njnt)]
    return {"joint_names": names, "nq": int(m.nq)}


def main():
    print(__doc__)
    print("Build the dataset via: python -m data_pipeline.run_offline <recording>")


if __name__ == "__main__":
    main()
