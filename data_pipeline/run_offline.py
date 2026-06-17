"""Offline motion-translation engine — single-command runner (stages 1→6).

Takes ONE recording (base.mp4 + imu.json) and produces:
  (1) a documented robot-usable dataset  → outputs/<name>_dataset.npz (+ schema.json)
  (2) a MuJoCo replay of the G1+Inspire   → outputs/replay_<name>.mp4 + sample frames
With --live it additionally drives the passive viewer in real time.

Chain: pose_extract → stabilize → imu_sync → retarget → build_dataset → replay.

The replay steps run as SEPARATE subprocesses so the offscreen Renderer and the
live passive viewer never share a GL context in one process (MUJOCO_GL=glfw).
"""
from __future__ import annotations
import argparse
import os
import pathlib
import subprocess
import sys
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_REC = REPO / "data/task_04_pasteur_pipette"
MODEL = REPO / "sim/assets/scene_g1_inspire.xml"


def first_recording(task_dir):
    cands = sorted(p.parent for p in pathlib.Path(task_dir).glob("*/*/base.mp4"))
    if not cands:
        cands = sorted(p.parent for p in pathlib.Path(task_dir).glob("**/base.mp4"))
    if not cands:
        sys.exit(f"ERROR: no base.mp4 under {task_dir}")
    return cands[0]


def build_dataset(recording, name, max_frames):
    from data_pipeline.pose_extract import extract_pose
    from data_pipeline.stabilize import stabilize_targets
    from data_pipeline.imu_sync import imu_to_waist
    from data_pipeline.build_dataset import assemble, save, qpos_layout_from_model
    from method.retarget import Retargeter

    rec = pathlib.Path(recording)
    print(f"[run] recording: {rec}")
    print("[run] stage 1/5: pose_extract (MediaPipe Hands + One Euro)...")
    pose = extract_pose(rec / "base.mp4", max_frames=max_frames)
    T = pose["n_frames"]

    print("[run] stage 2/5: stabilize (2D wrist → approx 3D target)...")
    targets = stabilize_targets(pose)

    print("[run] stage 3/5: imu_sync (resample IMU → waist lean)...")
    lean, imeta = imu_to_waist(rec / "imu.json", pose["timestamps"][:T])
    print(f"        imu: {imeta}")

    print("[run] stage 4/5: retarget (DLS IK + fingers + stow)...")
    rt = Retargeter(MODEL)
    nq = rt.m.nq
    traj = np.zeros((T, nq))
    used = {"left": 0, "right": 0}
    for fi in range(T):
        present = {s: bool(pose[s]["present"][fi]) for s in ("left", "right")}
        for s in ("left", "right"):
            used[s] += present[s]
        q = rt.solve_frame(
            {s: targets[s][fi] for s in ("left", "right")},
            lean[fi],
            present,
            {s: pose[s]["curl"][fi] for s in ("left", "right")},
            {s: float(pose[s]["pinch"][fi]) for s in ("left", "right")},
            {s: pose[s]["thumb_dists"][fi] for s in ("left", "right")},
        )
        traj[fi] = q
        if fi % 200 == 0:
            print(f"  [retarget] frame {fi}/{T}")

    print("[run] stage 5/5: build_dataset (write npz + schema)...")
    model_meta = {
        "model": "sim/assets/scene_g1_inspire.xml",
        "g1_base": "mujoco_menagerie unitree_g1/g1.xml @accb6df",
        "inspire": "unitree_ros g1_description/inspire_hand DFQ @7c40519",
        "nu": int(rt.m.nu), "nq": int(rt.m.nq),
    }
    data = assemble(pose, targets, lean, traj, model_meta)
    out_npz = REPO / "outputs" / f"{name}_dataset.npz"
    npz, js = save(data, qpos_layout_from_model(rt.m), model_meta, out_npz)
    # validation summary
    cov = {s: float(pose[s]["present"][:T].mean()) for s in ("left", "right")}
    any_present = np.array([pose["left"]["present"][:T] | pose["right"]["present"][:T]]).mean()
    print(f"[run] dataset → {npz}  (+ {js.name})")
    print(f"[run] coverage: left={cov['left']*100:.1f}%  right={cov['right']*100:.1f}%  "
          f"any-hand={any_present*100:.1f}%  frames_with_real_tracking={any_present*100:.1f}%  "
          f"stow={100-any_present*100:.1f}%")
    return npz


def run_replay(npz, live):
    env = dict(os.environ, MUJOCO_GL="glfw")
    replay = str(REPO / "sim/replay_g1.py")
    print("[run] replay → mp4 (offscreen)...")
    mp4 = str(REPO / "outputs" / f"replay_{npz.stem.replace('_dataset','')}.mp4")
    r = subprocess.run([sys.executable, replay, "--dataset", str(npz), "--mp4", mp4,
                        "--figures", str(REPO / "report/figures")], env=env)
    mp4_ok = (r.returncode == 0 and pathlib.Path(mp4).exists())
    print(f"[run] mp4 {'OK' if mp4_ok else 'FAILED'}: {mp4}")
    if live:
        print("[run] launching LIVE passive viewer (separate process)...")
        subprocess.run([sys.executable, replay, "--dataset", str(npz), "--live"], env=env)
    return mp4_ok


def main():
    ap = argparse.ArgumentParser(description="Offline motion-translation engine")
    ap.add_argument("recording", nargs="?", default=None,
                    help="recording dir (default: first under data/task_04_pasteur_pipette)")
    ap.add_argument("--name", default="task04")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--live", action="store_true", help="also drive the passive viewer")
    ap.add_argument("--no-mp4", action="store_true", help="skip mp4/replay (dataset only)")
    args = ap.parse_args()

    rec = args.recording or first_recording(DEFAULT_REC)
    npz = build_dataset(rec, args.name, args.max_frames)
    if not args.no_mp4 or args.live:
        run_replay(npz, args.live)
    print("[run] done.")


if __name__ == "__main__":
    main()
