"""Per-frame hand tracking from egocentric RGB video (MediaPipe Hands).

Pipeline stage 1. Decodes the video, runs MediaPipe Hands per frame (21
landmarks/hand + handedness + confidence), de-jitters the landmarks with a
One Euro filter, confidence-gates detections, and derives a per-finger curl
descriptor + thumb-index pinch distance.

Output (per frame, per robot arm side 'left'/'right'):
    landmarks  (T,21,3)  filtered normalized image coords (x,y in [0,1], z rel)
    present    (T,)      bool, hand detected & above confidence gate this frame
    conf       (T,)      handedness/detection score
    curl       (T,5)     per-finger curl in [0,1]  (thumb,index,middle,ring,pinky)
    pinch      (T,)      thumb-tip↔index-tip distance (normalized image units)
    wrist_uv   (T,2)     filtered wrist (landmark 0) image coords, for stabilize.py

Approximations / assumptions (documented honestly):
  * HANDEDNESS: MediaPipe labels handedness assuming a MIRRORED (selfie) image.
    An egocentric head-cam is NOT mirrored, so we FLIP the label to map the
    demonstrator's physical hand to the same-side robot arm. Tunable below.
  * No depth here — only 2D + MediaPipe's weak relative z. Depth is approximated
    later in stabilize.py.

One Euro filter: Casiez, Roussel, Vogel, "1€ Filter: A Simple Speed-based Low-pass
Filter for Noisy Input in Interactive Systems", CHI 2012. https://gery.casiez.net/1euro/
"""
from __future__ import annotations
import math
import numpy as np

# ── tunables ────────────────────────────────────────────────────────────────
FLIP_HANDEDNESS = True   # egocentric (non-mirrored) → flip MediaPipe's selfie label
CONF_GATE = 0.5          # min handedness score to accept a detection
CURL_GAIN = 1.6          # scales raw curl ratio so a full fist ≈ 1.0 (see _finger_curl)
# One Euro params (per normalized-coord units, sampled per frame):
EURO_MINCUTOFF = 1.5     # lower → smoother but more lag
EURO_BETA = 0.05         # speed coefficient; higher → less lag on fast motion
EURO_DCUTOFF = 1.0

_FINGERS = {  # mediapipe landmark indices: (base, [joint, joint, tip])
    "thumb":  (1, [2, 3, 4]),
    "index":  (5, [6, 7, 8]),
    "middle": (9, [10, 11, 12]),
    "ring":   (13, [14, 15, 16]),
    "pinky":  (17, [18, 19, 20]),
}


class OneEuroFilter:
    """Scalar 1€ filter (Casiez et al. 2012). One instance per signal channel."""

    def __init__(self, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self._x_prev is None:
            self._x_prev, self._t_prev = x, t
            return x
        dt = max(t - self._t_prev, 1e-6)
        dx = (x - self._x_prev) / dt
        a_d = self._alpha(self.dcutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self._dx_prev
        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self._x_prev
        self._x_prev, self._dx_prev, self._t_prev = x_hat, dx_hat, t
        return x_hat


def _finger_curl(lm, base, joints):
    """Curl in [0,1]: 0 straight, 1 fully bent. Path-length ratio (uses x,y,z)."""
    pts = [lm[base]] + [lm[j] for j in joints]
    straight = np.linalg.norm(pts[-1] - pts[0])
    path = sum(np.linalg.norm(pts[i + 1] - pts[i]) for i in range(len(pts) - 1))
    if path < 1e-6:
        return 0.0
    raw = 1.0 - straight / path            # 0 straight, →1 curled
    return float(np.clip(raw * CURL_GAIN, 0.0, 1.0))


def extract_pose(video_path, conf_gate=CONF_GATE, max_frames=None, progress=True):
    """Run MediaPipe Hands over the video. Returns dict of per-frame arrays."""
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        nf = min(nf, max_frames)

    sides = ("left", "right")
    out = {
        "fps": fps, "n_frames": nf, "width": int(cap.get(3)), "height": int(cap.get(4)),
        "timestamps": np.zeros(nf),
    }
    for s in sides:
        out[s] = {
            "landmarks": np.full((nf, 21, 3), np.nan),
            "present": np.zeros(nf, bool),
            "conf": np.zeros(nf),
            "curl": np.zeros((nf, 5)),
            "pinch": np.full(nf, np.nan),
            "wrist_uv": np.full((nf, 2), np.nan),
        }
    # per-side, per-landmark, per-axis One Euro filters
    euros = {s: [[OneEuroFilter(EURO_MINCUTOFF, EURO_BETA, EURO_DCUTOFF) for _ in range(3)]
                 for _ in range(21)] for s in sides}

    flip = {"Left": "right", "Right": "left"} if FLIP_HANDEDNESS else {"Left": "left", "Right": "right"}

    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                        min_detection_confidence=conf_gate,
                        min_tracking_confidence=0.5) as hands:
        for fi in range(nf):
            ok, frame = cap.read()
            if not ok:
                out["n_frames"] = fi
                break
            # prefer real per-frame PTS; fall back to index/fps
            t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            t = t_ms / 1000.0 if t_ms and t_ms > 0 else fi / fps
            out["timestamps"][fi] = t

            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.multi_hand_landmarks and res.multi_handedness:
                for hlms, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                    label = hd.classification[0].label
                    score = hd.classification[0].score
                    if score < conf_gate:
                        continue
                    side = flip.get(label, "left")
                    raw = np.array([[p.x, p.y, p.z] for p in hlms.landmark])  # (21,3)
                    filt = np.empty_like(raw)
                    for li in range(21):
                        for ax in range(3):
                            filt[li, ax] = euros[side][li][ax](raw[li, ax], t)
                    d = out[side]
                    d["landmarks"][fi] = filt
                    d["present"][fi] = True
                    d["conf"][fi] = score
                    d["wrist_uv"][fi] = filt[0, :2]
                    for k, (fn, (base, joints)) in enumerate(_FINGERS.items()):
                        d["curl"][fi, k] = _finger_curl(filt, base, joints)
                    d["pinch"][fi] = float(np.linalg.norm(filt[4] - filt[8]))
            if progress and fi % 200 == 0:
                print(f"  [pose] frame {fi}/{nf}")
    cap.release()
    for s in sides:
        cov = out[s]["present"][: out["n_frames"]].mean() * 100
        print(f"  [pose] {s}: {cov:.1f}% frames present")
    return out


def main():
    import argparse, pathlib
    ap = argparse.ArgumentParser(description="Extract hand tracking from a recording's base.mp4")
    ap.add_argument("recording", help="path to recording dir (containing base.mp4)")
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()
    vid = pathlib.Path(args.recording) / "base.mp4"
    out = extract_pose(vid, max_frames=args.max_frames)
    print(f"fps={out['fps']:.2f} frames={out['n_frames']} "
          f"L={out['left']['present'].mean()*100:.0f}% R={out['right']['present'].mean()*100:.0f}%")


if __name__ == "__main__":
    main()
