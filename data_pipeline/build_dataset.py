"""Assemble + write the per-frame robot-usable dataset (the deliverable).

Pipeline stage 5. Collects the outputs of pose/imu/stabilize/retarget into a
single documented dataset: outputs/<name>_dataset.npz plus a JSON sidecar that
documents the schema. This file is THE dataset deliverable; the replay
(sim/replay_g1.py) consumes the `qpos` trajectory directly.
"""
from __future__ import annotations
import json
import pathlib
import numpy as np


def assemble(pose, targets, lean, qpos_traj, model_meta):
    """Pack everything into a flat dict of arrays (one row per video frame)."""
    T = pose["n_frames"]
    sl = slice(0, T)
    data = {
        "timestamps": pose["timestamps"][sl],
        "fps": np.array(pose["fps"]),
        "waist_lean": lean[sl],                                  # (T,3) roll,pitch,yaw rad
        "qpos": qpos_traj[sl],                                   # (T, nq) replay-ready targets
    }
    for s in ("left", "right"):
        data[f"present_{s}"] = pose[s]["present"][sl]
        data[f"conf_{s}"] = pose[s]["conf"][sl]
        data[f"wrist_target_{s}"] = targets[s][sl]               # (T,3) approx 3D
        data[f"curl_{s}"] = pose[s]["curl"][sl]                  # (T,5)
        data[f"pinch_{s}"] = pose[s]["pinch"][sl]                # (T,)
        data[f"wrist_uv_{s}"] = pose[s]["wrist_uv"][sl]          # (T,2)
    return data


def save(data, qpos_layout, model_meta, out_npz):
    out_npz = pathlib.Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **data, qpos_joint_names=np.array(qpos_layout["joint_names"]))
    schema = {
        "file": out_npz.name,
        "frames": int(len(data["timestamps"])),
        "fps": float(data["fps"]),
        "model": model_meta,
        "arrays": {
            "timestamps": "(T,) seconds, video frame times (zeroed at frame 0)",
            "fps": "scalar source video fps",
            "waist_lean": "(T,3) head-IMU lean target [roll,pitch,yaw] rad, clamped",
            "qpos": "(T, nq) full G1+Inspire joint position targets, replay-ready "
                    "(legs frozen at stand; waist+arms from IK; fingers from curl)",
            "qpos_joint_names": "(njnt,) joint names in qpos order (free base first)",
            "present_{left,right}": "(T,) bool, hand tracked this frame",
            "conf_{left,right}": "(T,) detection confidence",
            "wrist_target_{left,right}": "(T,3) APPROXIMATE 3D wrist target (RGB-only, no depth)",
            "curl_{left,right}": "(T,5) per-finger curl [thumb,index,middle,ring,pinky] in [0,1]",
            "pinch_{left,right}": "(T,) thumb-index pinch distance (normalized image units)",
            "wrist_uv_{left,right}": "(T,2) wrist image coords (filtered)",
        },
        "approximations": [
            "DEPTH: wrist 3D target is approximated from 2D image position + apparent "
            "hand scale + MediaPipe relative z. NOT metric (single RGB stream).",
            "IMU-VIDEO SYNC: no shared clock; IMU sample 0 assumed == video frame 0. "
            "Durations agree to ~0.06-0.10 s (diagnostics), so ~2-3 frame unknown offset.",
            "HANDEDNESS: MediaPipe selfie-label flipped for the non-mirrored egocentric view.",
            "FINGERS: 12 independent joints/hand driven (real Inspire is ~6-motor "
            "underactuated); curl mapped uniformly to each finger's joints.",
            "WAIST: head orientation is RELATIVE to the recording's first frame "
            "(initial head pose = neutral). roll/pitch clamped +/-30deg.",
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
