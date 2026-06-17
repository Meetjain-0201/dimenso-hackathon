"""POV side-by-side test harness — spot IK problems against the real recording.

DIAGNOSIS ONLY — does not change any pipeline/IK logic. Runs the existing
pipeline on one recording, picks 10 diagnostic frames spread across the timeline
(each tagged with a behavior), and for each builds a side-by-side:

    LEFT  : source video frame + MediaPipe hand overlay
    RIGHT : G1+Inspire posed at that frame's retargeted qpos (real IMU lean
            applied), rendered from the head-mounted "pov_head" camera

Outputs: report/figures/diag_compare_<NN>_<tag>.png (+ a contact sheet
sim/assets/diag_compare_grid.png). Rerun after each tuning change and diff by eye.

MediaPipe pose extraction is cached (outputs/pose_cache_<name>.npz) so reruns
after an IK-only tuning change skip the slow perception step. Use --no-cache to
force re-extraction.  MUJOCO_GL=glfw.
"""
from __future__ import annotations
import argparse, pathlib, sys
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
SCENE = REPO / "sim/assets/scene_g1_inspire.xml"
N_FRAMES = 10                       # spread across the recording (user: 10 for first tuning)
PINCH_PCTL = 15                     # bottom-% pinch distance counts as a "pinch"
HEADTILT_DEG = 10.0                 # |waist-pitch target| above this = "headtilt"


def _pose_cache_path(name): return REPO / "outputs" / f"pose_cache_{name}.npz"


def get_pose(recording, name, use_cache=True):
    from data_pipeline.pose_extract import extract_pose, attach_thumb_dists
    cache = _pose_cache_path(name)
    if use_cache and cache.exists():
        print(f"[pov] loading cached pose {cache.name}")
        z = np.load(cache, allow_pickle=True)
        pose = {"fps": float(z["fps"]), "n_frames": int(z["n_frames"]),
                "width": int(z["width"]), "height": int(z["height"]),
                "timestamps": z["timestamps"]}
        for s in ("left", "right"):
            pose[s] = {k[len(s) + 1:]: z[k] for k in z.files if k.startswith(s + "_")}
        attach_thumb_dists(pose)     # compute CHANGE-4 descriptor from cached landmarks
        return pose
    pose = extract_pose(pathlib.Path(recording) / "base.mp4")
    flat = {"fps": pose["fps"], "n_frames": pose["n_frames"], "width": pose["width"],
            "height": pose["height"], "timestamps": pose["timestamps"]}
    for s in ("left", "right"):
        for k, v in pose[s].items():
            flat[f"{s}_{k}"] = v
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **flat)
    print(f"[pov] cached pose → {cache.name}")
    return pose


def run_pipeline(recording, name, use_cache):
    """Returns pose, targets, lean(T,3), qpos_traj(T,nq), Retargeter."""
    from data_pipeline.stabilize import stabilize_targets
    from data_pipeline.imu_sync import imu_to_waist
    from method.retarget import Retargeter
    pose = get_pose(recording, name, use_cache)
    T = pose["n_frames"]
    targets = stabilize_targets(pose)
    lean, _ = imu_to_waist(pathlib.Path(recording) / "imu.json", pose["timestamps"][:T])
    rt = Retargeter(SCENE)
    traj = np.zeros((T, rt.m.nq))
    for fi in range(T):
        present = {s: bool(pose[s]["present"][fi]) for s in ("left", "right")}
        traj[fi] = rt.solve_frame(
            {s: targets[s][fi] for s in ("left", "right")}, lean[fi], present,
            {s: pose[s]["curl"][fi] for s in ("left", "right")},
            {s: float(pose[s]["pinch"][fi]) for s in ("left", "right")},
            {s: pose[s]["thumb_dists"][fi] for s in ("left", "right")})
    return pose, targets, lean, traj, rt


def select_frames(pose, targets, lean):
    """10 frames covering varied behaviors AND spread across the timeline.

    Fill a per-behavior quota by extremity, but require each pick to be at least
    MIN_GAP frames from every other pick (temporal spread); backfill any unmet
    quota with time-spread reach frames.
    """
    T = pose["n_frames"]
    MIN_GAP = max(20, T // 25)            # ≥ ~1 s between chosen frames
    pr, pl = pose["right"]["present"][:T], pose["left"]["present"][:T]
    pinch = pose["right"]["pinch"][:T]
    pitch = np.abs(np.rad2deg(lean[:T, 1]))
    tx = targets["right"][:T, 0]
    fin = np.isfinite(pinch)
    pinch_thr = np.nanpercentile(pinch[fin], PINCH_PCTL) if fin.any() else 0
    txmid = np.nanmedian(tx[pr]) if pr.any() else 0.3

    def tag(fi):
        if pr[fi] and pl[fi]: return "bimanual"
        if not pr[fi]: return "stow"                      # primary (right) hand gone
        if fin[fi] and pinch[fi] <= pinch_thr: return "pinch"
        if pitch[fi] >= HEADTILT_DEG: return "headtilt"
        return "reach-far" if tx[fi] >= txmid else "reach-near"

    cats = {}
    for fi in range(T):
        cats.setdefault(tag(fi), []).append(fi)

    def rank_key(cat, fi):                                # smaller = picked first
        if cat == "pinch": return pinch[fi]
        if cat == "headtilt": return -pitch[fi]
        if cat == "reach-far": return -tx[fi]
        if cat == "reach-near": return tx[fi]
        return abs(fi - T / 2)                            # bimanual/stow: central-ish

    chosen = []
    ok = lambda fi: all(abs(fi - c) >= MIN_GAP for c, _ in chosen)
    quota = [("pinch", 2), ("bimanual", 2), ("stow", 2), ("headtilt", 1),
             ("reach-far", 2), ("reach-near", 1)]
    for cat, q in quota:
        n = 0
        for fi in sorted(cats.get(cat, []), key=lambda f: rank_key(cat, f)):
            if n >= q: break
            if ok(fi): chosen.append((fi, cat)); n += 1
    # backfill to N_FRAMES with any hand-present frames, spread in time
    for fi in range(0, T, max(1, MIN_GAP)):
        if len(chosen) >= N_FRAMES: break
        if (pr[fi] or pl[fi]) and ok(fi): chosen.append((fi, tag(fi)))
    return sorted(chosen)[:N_FRAMES]


def draw_hand_overlay(img, pose, fi):
    import cv2
    import mediapipe as mp
    H, W = img.shape[:2]
    HC = mp.solutions.hands.HAND_CONNECTIONS
    colors = {"right": (0, 255, 0), "left": (0, 180, 255)}
    for s in ("right", "left"):
        if not pose[s]["present"][fi]:
            continue
        lm = pose[s]["landmarks"][fi]            # (21,3) normalized
        pts = [(int(x * W), int(y * H)) for x, y, _ in lm]
        for a, b in HC:
            cv2.line(img, pts[a], pts[b], colors[s], 2)
        for p in pts:
            cv2.circle(img, p, 3, colors[s], -1)
    return img


def render_pov(rt, qpos, cam_id, renderer):
    import mujoco
    d = rt.d
    d.qpos[:] = qpos
    mujoco.mj_forward(rt.m, d)
    renderer.update_scene(d, camera=cam_id)
    return renderer.render()                     # RGB


def render_clip(rec, pose, traj, rt, n, out_mp4):
    """Render the best-right-hand-coverage contiguous n-frame window as a
    side-by-side mp4 (source+overlay ∥ POV replay). Returns (start, end, coverage)."""
    import cv2, mujoco
    T = pose["n_frames"]; fps = pose["fps"] or 30.0
    pr = pose["right"]["present"][:T].astype(int)
    n = min(n, T)
    csum = np.concatenate([[0], np.cumsum(pr)])
    win = csum[n:] - csum[:-n]                       # right-present count per window
    start = int(np.argmax(win)); end = start + n
    cov = float(win[start]) / n
    cam = mujoco.mj_name2id(rt.m, mujoco.mjtObj.mjOBJ_CAMERA, "pov_head")
    R = mujoco.Renderer(rt.m, height=480, width=640)
    cap = cv2.VideoCapture(str(rec / "base.mp4"))
    out_mp4 = pathlib.Path(out_mp4); out_mp4.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 480))
    for fi in range(start, end):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi); ok, fr = cap.read()
        left = draw_hand_overlay(cv2.resize(fr, (640, 480)), pose, fi) if ok else np.zeros((480, 640, 3), np.uint8)
        right = cv2.cvtColor(render_pov(rt, traj[fi], cam, R), cv2.COLOR_RGB2BGR)
        combo = np.hstack([left, right])
        cv2.rectangle(combo, (0, 0), (1280, 28), (0, 0, 0), -1)
        cv2.putText(combo, "SOURCE + MediaPipe", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.putText(combo, f"POV replay (tuned)  f={fi}  t={fi/fps:.2f}s", (650, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        vw.write(combo)
        if (fi - start) % 50 == 0:
            print(f"  [clip] {fi-start}/{n}")
    cap.release(); vw.release()
    print(f"[pov] clip → {out_mp4}  frames {start}-{end-1} (t {start/fps:.1f}-{end/fps:.1f}s)  "
          f"right-hand coverage {cov*100:.1f}%")
    return start, end, cov


def main():
    import cv2, mujoco
    ap = argparse.ArgumentParser(description="POV side-by-side IK diagnosis harness")
    ap.add_argument("recording", nargs="?", default=None,
                    help="recording dir (default: first under data/task_04_pasteur_pipette)")
    ap.add_argument("--name", default="task04")
    ap.add_argument("--no-cache", action="store_true", help="force MediaPipe re-extraction")
    ap.add_argument("--clip", type=int, default=0,
                    help="also render an N-frame best-coverage side-by-side mp4")
    ap.add_argument("--clip-out", default="outputs/tuned_compare_250.mp4")
    args = ap.parse_args()
    rec = args.recording
    if rec is None:
        rec = sorted(p.parent for p in (REPO / "data/task_04_pasteur_pipette").glob("*/*/base.mp4"))[0]
    rec = pathlib.Path(rec)

    pose, targets, lean, traj, rt = run_pipeline(rec, args.name, not args.no_cache)
    chosen = select_frames(pose, targets, lean)
    fps = pose["fps"]
    print("\n[pov] chosen frames (time-spread, tagged):")
    for fi, t in chosen:
        print(f"   frame {fi:4d}  t={fi/fps:6.2f}s  tag={t}")

    cam_id = mujoco.mj_name2id(rt.m, mujoco.mjtObj.mjOBJ_CAMERA, "pov_head")
    renderer = mujoco.Renderer(rt.m, height=480, width=640)
    cap = cv2.VideoCapture(str(rec / "base.mp4"))
    figdir = REPO / "report/figures"; figdir.mkdir(parents=True, exist_ok=True)
    panels = []
    for n, (fi, tg) in enumerate(chosen):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi)); ok, frame = cap.read()
        left = cv2.resize(frame, (640, 480)) if ok else np.zeros((480, 640, 3), np.uint8)
        left = draw_hand_overlay(left, pose, fi)
        right = cv2.cvtColor(render_pov(rt, traj[fi], cam_id, renderer), cv2.COLOR_RGB2BGR)
        combo = np.hstack([left, right])
        label = f"t={fi/fps:.2f}s  [{tg}]"
        cv2.rectangle(combo, (0, 0), (1280, 30), (0, 0, 0), -1)
        cv2.putText(combo, "SOURCE + MediaPipe", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(combo, f"POV replay   {label}", (650, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        out = figdir / f"diag_compare_{n:02d}_{tg}.png"
        cv2.imwrite(str(out), combo)
        panels.append(combo)
    cap.release()

    # contact sheet: 2 columns × 5 rows, each panel downscaled
    cell = [cv2.resize(p, (640, 240)) for p in panels]
    rows = [np.hstack(cell[i:i + 2]) for i in range(0, len(cell), 2)]
    grid = np.vstack(rows)
    grid_path = REPO / "sim/assets/diag_compare_grid.png"
    cv2.imwrite(str(grid_path), grid)
    print(f"[pov] wrote {len(panels)} comparisons → report/figures/diag_compare_*.png")
    print(f"[pov] contact sheet → {grid_path}")

    # ── targeted diagnostics ────────────────────────────────────────────────
    print_thumb_diag(rt, traj, chosen, pose)
    print_headtilt_diag(rt, traj, chosen, lean)
    print_mismatch_diag(rt, traj, chosen, pose)

    if args.clip:
        render_clip(rec, pose, traj, rt, args.clip, REPO / args.clip_out)


def print_mismatch_diag(rt, traj, chosen, pose):
    """Quantify the two mismatches spotted by eye (torso tilt, hand separation).
    DIAGNOSIS ONLY — surfaces the gap for the IK/stabilize tuning pass, no fix here."""
    import mujoco
    bid = lambda n: mujoco.mj_name2id(rt.m, mujoco.mjtObj.mjOBJ_BODY, n)
    print("\n[pov] === torso tilt + hand-separation (robot vs human) — for tuning ===")
    print("   frame  tag         waist_roll  waist_pitch | robot_hand_sep   human_wrist_sep(img-norm)")
    for fi, tg in chosen:
        rt.d.qpos[:] = traj[fi]; mujoco.mj_forward(rt.m, rt.d)
        roll = np.rad2deg(traj[fi][rt.qadr("waist_roll_joint")])
        pitch = np.rad2deg(traj[fi][rt.qadr("waist_pitch_joint")])
        rsep = float(np.linalg.norm(rt.d.xpos[bid("L_hand_base_link")] - rt.d.xpos[bid("R_hand_base_link")]))
        if pose["left"]["present"][fi] and pose["right"]["present"][fi]:
            hs = f"{float(np.linalg.norm(pose['left']['wrist_uv'][fi] - pose['right']['wrist_uv'][fi])):.3f}"
        else:
            hs = "n/a (1 hand)"
        print(f"   {fi:4d}  {tg:11s} {roll:+7.1f}    {pitch:+7.1f}   |   {rsep:.3f} m        {hs}")
    print("   note: robot hand-sep is driven by stabilize.py Y_SPAN/SIDE_BIAS (not the real")
    print("         hand distance); waist_roll reaching ±30° is the visible torso tilt.")


def print_thumb_diag(rt, traj, chosen, pose):
    pinch_frames = [fi for fi, t in chosen if t == "pinch"]
    print("\n[pov] === PINCH-FRAME thumb vs finger joint check (is the thumb moving?) ===")
    if not pinch_frames:
        print("   (no pinch frame selected this run)")
        return
    qa = lambda j: rt.qadr(j)
    for fi in pinch_frames:
        q = traj[fi]
        print(f"   frame {fi} (pinch={pose['right']['pinch'][fi]:.3f}):")
        for grp in ("thumb_proximal_yaw", "thumb_proximal_pitch", "thumb_intermediate", "thumb_distal",
                    "index_proximal", "index_intermediate", "middle_proximal"):
            jn = f"R_{grp}_joint"
            print(f"      R_{grp:22s} = {q[qa(jn)]:+.3f} rad")


def print_headtilt_diag(rt, traj, chosen, lean):
    ht = [fi for fi, t in chosen if t == "headtilt"]
    print("\n[pov] === HEAD-TILT frame: IMU pitch → waist-pitch target ===")
    if not ht:
        print("   (no headtilt frame selected; showing the max-pitch frame instead)")
        ht = [int(np.argmax(np.abs(lean[:, 1])))]
    qa = rt.qadr
    for fi in ht:
        imu_pitch = np.rad2deg(lean[fi, 1])
        wp = np.rad2deg(traj[fi][qa("waist_pitch_joint")])
        print(f"   frame {fi}: IMU pitch target = {imu_pitch:+.1f}°  →  commanded waist_pitch = {wp:+.1f}°")


if __name__ == "__main__":
    main()
