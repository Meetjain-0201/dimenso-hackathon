"""Replay a retargeted G1+Inspire joint trajectory in MuJoCo.

Loads sim/assets/scene_g1_inspire.xml, freezes the legs at the stand keyframe,
and plays back the per-frame `qpos` trajectory from an offline-engine dataset
(.npz). Kinematic replay (set qpos + mj_forward, no dynamics) — matches how the
diagnostics/asset renders were validated.

Two modes (run as SEPARATE invocations — a live passive viewer and the offscreen
Renderer can fight over the GL context in one process):
  --mp4 PATH   offscreen render → mp4 (640x480, MUJOCO_GL=glfw) + sample frames
  --live       mujoco.viewer.launch_passive window playing in ~real time
               (kills the stale pre-Inspire viewer PID first)

Model G1 base sourced from mujoco_menagerie; Inspire hands grafted from
unitree_ros (see docs/inspire_hand_model.md).
"""
from __future__ import annotations
import argparse
import pathlib
import time
import numpy as np
import mujoco

SCENE = pathlib.Path(__file__).resolve().parent / "assets" / "scene_g1_inspire.xml"
CAM = dict(azimuth=150, elevation=-18, distance=2.0, lookat=[0.12, 0.0, 0.92])
STALE_VIEWER_PID = 440742   # pre-Inspire interactive viewer Meet had open


def _load():
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)        # stand: legs in standing stance
    # leg joints stay frozen at these stand values throughout replay
    leg_q = {i: d.qpos[m.jnt_qposadr[i]] for i in range(m.njnt)
             if any(k in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) or "")
                    for k in ("hip", "knee", "ankle"))}
    return m, d, leg_q


def _apply(m, d, q, leg_q):
    d.qpos[:] = q
    for i, v in leg_q.items():                  # re-assert frozen legs
        d.qpos[m.jnt_qposadr[i]] = v
    mujoco.mj_forward(m, d)


def render_mp4(npz, out_mp4, fig_dir):
    import cv2
    data = np.load(npz, allow_pickle=True)
    traj = data["qpos"]; fps = float(data["fps"]) or 30.0
    m, d, leg_q = _load()
    r = mujoco.Renderer(m, height=480, width=640)
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(m, cam)
    cam.azimuth = CAM["azimuth"]; cam.elevation = CAM["elevation"]
    cam.distance = CAM["distance"]; cam.lookat[:] = CAM["lookat"]
    out_mp4 = pathlib.Path(out_mp4); out_mp4.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 480))
    fig_dir = pathlib.Path(fig_dir); fig_dir.mkdir(parents=True, exist_ok=True)
    T = len(traj)
    sample_idx = set(np.linspace(0, T - 1, 4).astype(int))
    for i in range(T):
        _apply(m, d, traj[i], leg_q)
        r.update_scene(d, camera=cam)
        rgb = r.render()
        vw.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if i in sample_idx:
            from PIL import Image
            n = sorted(sample_idx).index(i)
            Image.fromarray(rgb).save(fig_dir / f"replay_frame_{n}.png")
        if i % 200 == 0:
            print(f"  [replay] frame {i}/{T}")
    vw.release()
    print(f"[replay] wrote {out_mp4} ({T} frames @ {fps:.1f} fps) + 4 sample frames → {fig_dir}")


def render_compare(npz, src_video, out_mp4, fig_dir):
    """Side-by-side mp4: source egocentric video (left) ∥ G1+Inspire replay (right).

    Frame-aligned 1:1 (the trajectory has one qpos row per video frame), so this is
    the honest visual check of the motion translation."""
    import cv2
    data = np.load(npz, allow_pickle=True)
    traj = data["qpos"]; fps = float(data["fps"]) or 30.0
    m, d, leg_q = _load()
    r = mujoco.Renderer(m, height=480, width=640)
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(m, cam)
    cam.azimuth = CAM["azimuth"]; cam.elevation = CAM["elevation"]
    cam.distance = CAM["distance"]; cam.lookat[:] = CAM["lookat"]
    cap = cv2.VideoCapture(str(src_video))
    out_mp4 = pathlib.Path(out_mp4); out_mp4.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 480))
    fig_dir = pathlib.Path(fig_dir); fig_dir.mkdir(parents=True, exist_ok=True)
    T = len(traj); sample = set(np.linspace(0, T - 1, 3).astype(int))
    for i in range(T):
        ok, frame = cap.read()
        left = cv2.resize(frame, (640, 480)) if ok else np.zeros((480, 640, 3), np.uint8)
        _apply(m, d, traj[i], leg_q)
        r.update_scene(d, camera=cam)
        right = cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR)
        combo = np.hstack([left, right])
        cv2.putText(combo, "egocentric video", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(combo, "G1 + Inspire replay", (652, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        vw.write(combo)
        if i in sample:
            n = sorted(sample).index(i)
            cv2.imwrite(str(fig_dir / f"compare_frame_{n}.png"), combo)
        if i % 200 == 0:
            print(f"  [compare] frame {i}/{T}")
    cap.release(); vw.release()
    print(f"[replay] wrote side-by-side {out_mp4} + 3 compare frames → {fig_dir}")


def play_live(npz, loops=2):
    import os, signal
    import mujoco.viewer
    # kill the stale pre-Inspire interactive viewer if still running
    try:
        os.kill(STALE_VIEWER_PID, signal.SIGTERM)
        print(f"[replay] killed stale viewer PID {STALE_VIEWER_PID}")
    except ProcessLookupError:
        print(f"[replay] stale viewer PID {STALE_VIEWER_PID} not running (ok)")
    except Exception as e:
        print(f"[replay] could not signal PID {STALE_VIEWER_PID}: {e}")

    data = np.load(npz, allow_pickle=True)
    traj = data["qpos"]; fps = float(data["fps"]) or 30.0
    dt = 1.0 / fps
    m, d, leg_q = _load()
    print(f"[replay] LIVE: {len(traj)} frames @ {fps:.1f} fps, {loops} loop(s). Close window to stop.")
    with mujoco.viewer.launch_passive(m, d) as v:
        for _ in range(loops):
            for i in range(len(traj)):
                if not v.is_running():
                    return
                t0 = time.time()
                _apply(m, d, traj[i], leg_q)
                v.sync()
                time.sleep(max(0.0, dt - (time.time() - t0)))
        # hold final pose until closed
        while v.is_running():
            mujoco.mj_forward(m, d); v.sync(); time.sleep(0.05)


def main():
    ap = argparse.ArgumentParser(description="Replay a G1+Inspire qpos trajectory")
    ap.add_argument("--dataset", default="outputs/task04_dataset.npz")
    ap.add_argument("--mp4", default=None, help="output mp4 path (offscreen render mode)")
    ap.add_argument("--figures", default="report/figures", help="dir for sample frames")
    ap.add_argument("--live", action="store_true", help="drive a passive viewer in real time")
    ap.add_argument("--compare", default=None, help="source base.mp4 → side-by-side compare mp4")
    ap.add_argument("--loops", type=int, default=2)
    args = ap.parse_args()
    if args.compare:
        out = args.mp4 or "outputs/compare_task04.mp4"
        render_compare(args.dataset, args.compare, out, args.figures)
    elif args.live:
        play_live(args.dataset, loops=args.loops)
    else:
        out = args.mp4 or "outputs/replay_task04.mp4"
        render_mp4(args.dataset, out, args.figures)


if __name__ == "__main__":
    main()
