"""Head IMU → per-frame waist-lean target, resampled onto video frame times.

Pipeline stage 2. Loads imu.json (samples = [ts_ms, ax,ay,az, gx,gy,gz];
accel m/s², gyro rad/s, ~100 Hz but jittery), resamples it onto the VIDEO frame
timestamps (NOT a fixed 10 ms grid), and recovers head orientation over time.
Outputs a per-frame waist lean target (roll, pitch, yaw) in radians, clamped.

Orientation recovery (complementary filter):
  * roll/pitch reference from the gravity direction in the accelerometer
    (low-frequency, drift-free but noisy under linear acceleration);
  * gyro integrated for high-frequency change, fused with the accel reference
    (alpha≈0.98) so short accelerations don't corrupt tilt;
  * yaw has NO absolute reference (no magnetometer) → gyro-integrated with a slow
    leak toward zero to bound drift. Treated as least-trustworthy.

Lean is expressed RELATIVE TO THE FIRST FRAME's head orientation (the recording's
initial pose is taken as "neutral, zero lean"). This sidesteps the unknown IMU
axis-to-world calibration: we only need the CHANGE in head tilt.

SYNC ASSUMPTION (documented): there is no shared clock between video and IMU.
We assume IMU sample 0 == video frame 0 (best effort) and map IMU time
t_imu = (ts_ms - ts_ms[0]) / 1000 onto the video timeline. Diagnostics showed
the two durations agree to ~0.06–0.10 s, so this is a small (≈2–3 frame) unknown.
"""
from __future__ import annotations
import json
import numpy as np

# ── tunables ────────────────────────────────────────────────────────────────
COMP_ALPHA = 0.98        # complementary filter weight on gyro vs accel (roll/pitch)
YAW_LEAK = 0.97          # per-frame decay of integrated yaw toward 0 (drift bound)
CLAMP_RP_DEG = 30.0      # waist roll/pitch clamp (G1 waist hardware limit)
CLAMP_YAW_DEG = 45.0     # waist yaw clamp (conservative; joint allows ±150°)


def _accel_rp(ax, ay, az):
    """Roll/pitch (rad) from the gravity vector measured by the accelerometer."""
    roll = np.arctan2(ay, az)
    pitch = np.arctan2(-ax, np.sqrt(ay * ay + az * az))
    return roll, pitch


def imu_to_waist(imu_path, frame_times):
    """Return (T,3) waist lean targets [roll,pitch,yaw] (rad), one per video frame.

    frame_times : (T,) video frame timestamps in seconds (from pose_extract).
    """
    j = json.load(open(imu_path))
    s = np.asarray(j["samples"], dtype=np.float64)
    t_imu = (s[:, 0] - s[0, 0]) / 1000.0          # device-uptime ms → s, zeroed
    acc = s[:, 1:4]
    gyr = s[:, 4:7]
    ft = np.asarray(frame_times, dtype=np.float64)
    ft = ft - ft[0]                               # zero video clock too (sync assumption)

    # Resample IMU channels onto the (jittery) video frame timestamps.
    acc_f = np.column_stack([np.interp(ft, t_imu, acc[:, k]) for k in range(3)])
    gyr_f = np.column_stack([np.interp(ft, t_imu, gyr[:, k]) for k in range(3)])

    T = len(ft)
    roll = np.zeros(T); pitch = np.zeros(T); yaw = np.zeros(T)
    r0, p0 = _accel_rp(*acc_f[0])
    roll[0], pitch[0] = r0, p0
    for i in range(1, T):
        dt = max(ft[i] - ft[i - 1], 1e-3)
        # gyro prediction (body rates ~ roll/pitch/yaw rates; small-angle approx)
        roll_g = roll[i - 1] + gyr_f[i, 0] * dt
        pitch_g = pitch[i - 1] + gyr_f[i, 1] * dt
        # accel correction
        roll_a, pitch_a = _accel_rp(*acc_f[i])
        roll[i] = COMP_ALPHA * roll_g + (1 - COMP_ALPHA) * roll_a
        pitch[i] = COMP_ALPHA * pitch_g + (1 - COMP_ALPHA) * pitch_a
        # yaw: integrate gyro_z, leak toward 0 (no absolute reference)
        yaw[i] = YAW_LEAK * (yaw[i - 1] + gyr_f[i, 2] * dt)

    # relative to initial orientation → "lean" from the recording's start pose
    roll -= roll[0]; pitch -= pitch[0]; yaw -= yaw[0]

    rp = np.deg2rad(CLAMP_RP_DEG); yc = np.deg2rad(CLAMP_YAW_DEG)
    lean = np.column_stack([
        np.clip(roll, -rp, rp),
        np.clip(pitch, -rp, rp),
        np.clip(yaw, -yc, yc),
    ])
    # `lean` (the IK input) is unchanged. The meta dict additionally exposes the
    # per-frame resampled IMU + recovered orientation for the dataset/figures.
    meta = dict(n_imu=len(s), imu_dur=float(t_imu[-1]), video_dur=float(ft[-1]),
                mean_dt_ms=float(np.mean(np.diff(t_imu)) * 1000),
                accel=acc_f, gyro=gyr_f,                 # (T,3) resampled to frame times
                head_roll=roll, head_pitch=pitch, head_yaw=yaw)  # (T,) relative, unclamped
    return lean, meta


def main():
    import argparse, pathlib
    ap = argparse.ArgumentParser(description="Resample IMU → per-frame waist lean")
    ap.add_argument("recording")
    args = ap.parse_args()
    rec = pathlib.Path(args.recording)
    # standalone: fabricate frame times from video fps to demo the resampler
    import cv2
    cap = cv2.VideoCapture(str(rec / "base.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS); nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    ft = np.arange(nf) / fps
    lean, meta = imu_to_waist(rec / "imu.json", ft)
    print("meta:", meta)
    print(f"lean roll/pitch/yaw deg ranges: "
          f"{np.rad2deg(lean.min(0)).round(1)} .. {np.rad2deg(lean.max(0)).round(1)}")


if __name__ == "__main__":
    main()
