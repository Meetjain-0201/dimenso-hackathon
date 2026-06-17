"""Approximate a 3D wrist target in the robot workspace from a 2D wrist landmark.

Pipeline stage 3. RGB-only, NO depth — so the 3D wrist target is APPROXIMATED.
We map the wrist's image (u,v) onto a horizontal table plane in front of the
robot, and use apparent hand scale (+ MediaPipe's weak relative z) as a
height/depth proxy.

============================  HEAVILY APPROXIMATE  ============================
There is no metric depth in a single RGB stream. The mapping below is a
deliberately simple, monotonic guess so the replay is plausible and temporally
coherent — it is NOT a calibrated reconstruction. Every constant is a tunable.
Mapping (camera looks forward-and-down from the head):
  * image u (0=left .. 1=right)   → robot lateral  y  (left arm = +y)
  * image v (0=top  .. 1=bottom)  → robot forward  x  (top=far, bottom=near)
  * hand scale (bbox size) + mp z → height/forward refinement (bigger=closer)
The wrist target is additionally biased to the correct side of the body so the
two arms don't cross.
=============================================================================
"""
from __future__ import annotations
import numpy as np

# ── tunables (workspace geometry, metres, robot base frame) ─────────────────
PLANE_HEIGHT = 0.75      # table-plane height z (m) the hands nominally work over
Z_FROM_SCALE = 0.18      # how much apparent hand scale lifts the target above plane
X_NEAR, X_FAR = 0.18, 0.55   # forward reach mapped from image-v (near..far)
Y_SPAN = 0.45            # total lateral span mapped from image-u
SIDE_BIAS = 0.10         # push each wrist toward its own side (m) to avoid crossing
SCALE_REF = 0.22         # reference hand bbox size (norm units) ≈ "comfortable" depth
SCALE_GAIN = 0.20        # forward (x) shift per unit (scale-SCALE_REF)


def _hand_scale(landmarks):
    """Apparent hand size in normalized image units (bbox diagonal). NaN-safe."""
    lm = landmarks[:, :2]
    if np.any(np.isnan(lm)):
        return np.nan
    span = lm.max(0) - lm.min(0)
    return float(np.hypot(*span))


def wrist_to_target(side, wrist_uv, landmarks, mp_z=None):
    """Map one frame's wrist (u,v) + hand to an approximate 3D target (x,y,z).

    side : 'left' or 'right' (robot arm). Returns (3,) or NaNs if no detection.
    """
    if np.any(np.isnan(wrist_uv)):
        return np.full(3, np.nan)
    u, v = float(wrist_uv[0]), float(wrist_uv[1])
    scale = _hand_scale(landmarks)

    # lateral: image-left → +y (robot left). u in [0,1] → y in [+Y/2 .. -Y/2]
    y = (0.5 - u) * Y_SPAN
    # forward: top of image = far. v in [0,1] (top..bottom) → x in [X_FAR..X_NEAR]
    x = X_FAR + (X_NEAR - X_FAR) * v
    # scale refinement: a bigger hand reads as closer → pull forward target in
    if not np.isnan(scale):
        x += (scale - SCALE_REF) * SCALE_GAIN
        z = PLANE_HEIGHT + np.clip((scale - SCALE_REF) / SCALE_REF, -1, 1) * Z_FROM_SCALE
    else:
        z = PLANE_HEIGHT
    # weak mp relative-z prior: more negative z (closer to cam) → slightly nearer
    if mp_z is not None and not np.isnan(mp_z):
        x += float(np.clip(mp_z, -0.2, 0.2)) * 0.1
    # keep each wrist on its own side
    y += SIDE_BIAS if side == "left" else -SIDE_BIAS
    return np.array([x, y, z])


def stabilize_targets(pose):
    """Per-frame 3D wrist targets for both arms. Returns {'left':(T,3),'right':(T,3)}."""
    T = pose["n_frames"]
    out = {}
    for side in ("left", "right"):
        d = pose[side]
        tg = np.full((T, 3), np.nan)
        for fi in range(T):
            if d["present"][fi]:
                mpz = d["landmarks"][fi, 0, 2]
                tg[fi] = wrist_to_target(side, d["wrist_uv"][fi], d["landmarks"][fi], mpz)
        out[side] = tg
    return out


def main():
    print(__doc__)
    print("Run via data_pipeline/run_offline.py — this stage needs pose_extract output.")


if __name__ == "__main__":
    main()
