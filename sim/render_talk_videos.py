"""Presentation render — full-length talk videos from the task_04 recording.

RENDERING ONLY — no pipeline/IK changes. Runs the (tuned) pipeline once
(MediaPipe pose cached) to get the per-frame qpos, then renders the WHOLE
recording, frame-aligned, from the scene's named cameras.

  VIDEO 1  outputs/talk_sidebyside_full.mp4   source+overlay ∥ robot pov_head
  VIDEO 2  outputs/talk_quad_full.mp4         2×2: source | pov_head / top_view | iso_view

Both at the source fps. Source frames are read sequentially (fast). MUJOCO_GL=glfw.

Usage:
  MUJOCO_GL=glfw python sim/render_talk_videos.py [recording] --video {1,2,both}
"""
from __future__ import annotations
import argparse, pathlib, sys, time
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sim.pov_compare import run_pipeline, draw_hand_overlay, render_pov   # reuse harness


def _label(img, txt):
    import cv2
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, txt, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return img


def _cam(rt, name):
    import mujoco
    return mujoco.mj_name2id(rt.m, mujoco.mjtObj.mjOBJ_CAMERA, name)


def render(rec, pose, traj, rt, which, fps):
    import cv2
    T = pose["n_frames"]
    R = __import__("mujoco").Renderer(rt.m, height=480, width=640)
    cams = {n: _cam(rt, n) for n in ("pov_head", "top_view", "iso_view")}
    cap = cv2.VideoCapture(str(rec / "base.mp4"))
    results = {}

    def source_frame(fi):
        ok, fr = cap.read()
        base = cv2.resize(fr, (640, 480)) if ok else np.zeros((480, 640, 3), np.uint8)
        return _label(draw_hand_overlay(base, pose, fi), "Source + MediaPipe")

    if which in ("1", "both"):
        out = REPO / "outputs/talk_sidebyside_full.mp4"
        vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 480))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        t0 = time.time()
        for fi in range(T):
            left = source_frame(fi)
            right = _label(cv2.cvtColor(render_pov(rt, traj[fi], cams["pov_head"], R), cv2.COLOR_RGB2BGR), "Robot POV")
            vw.write(np.hstack([left, right]))
            if fi % 200 == 0: print(f"  [v1] {fi}/{T}")
        vw.release()
        results["video1"] = (out, T, fps, time.time() - t0)
        print(f"[talk] VIDEO 1 → {out}  ({T} frames @ {fps:.2f} fps)")

    if which in ("2", "both"):
        out = REPO / "outputs/talk_quad_full.mp4"
        vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 960))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        t0 = time.time()
        for fi in range(T):
            src = source_frame(fi)
            pov = _label(cv2.cvtColor(render_pov(rt, traj[fi], cams["pov_head"], R), cv2.COLOR_RGB2BGR), "Robot POV")
            top = _label(cv2.cvtColor(render_pov(rt, traj[fi], cams["top_view"], R), cv2.COLOR_RGB2BGR), "Top")
            iso = _label(cv2.cvtColor(render_pov(rt, traj[fi], cams["iso_view"], R), cv2.COLOR_RGB2BGR), "Isometric")
            quad = np.vstack([np.hstack([src, pov]), np.hstack([top, iso])])
            vw.write(quad)
            if fi % 100 == 0: print(f"  [v2] {fi}/{T}")
        vw.release()
        results["video2"] = (out, T, fps, time.time() - t0)
        print(f"[talk] VIDEO 2 → {out}  ({T} frames @ {fps:.2f} fps)")

    cap.release()
    return results


def main():
    ap = argparse.ArgumentParser(description="Render full-length presentation videos")
    ap.add_argument("recording", nargs="?", default=None)
    ap.add_argument("--video", choices=["1", "2", "both"], default="both")
    args = ap.parse_args()
    rec = args.recording or sorted(p.parent for p in (REPO / "data/task_04_pasteur_pipette").glob("*/*/base.mp4"))[0]
    rec = pathlib.Path(rec)
    pose, targets, lean, traj, rt = run_pipeline(rec, "task04", use_cache=True)
    fps = pose["fps"] or 29.6
    res = render(rec, pose, traj, rt, args.video, fps)
    for k, (out, n, f, dt) in res.items():
        sz = out.stat().st_size / 1e6
        print(f"[talk] {k}: {out.name}  {n} frames  {n/f:.1f}s  {sz:.1f} MB  rendered in {dt:.0f}s")


if __name__ == "__main__":
    main()
