"""Synchronize the head IMU stream to video frames and recover head pose.

Intent
------
Take the ~100 Hz head IMU (accelerometer + gyroscope) and the video frame
timestamps observed in `inspect.py`, and produce a per-frame, time-aligned
estimate of head/camera pose:

    * resample / align IMU samples to each video frame's timestamp
      (handle differing start times, jitter, and dropped samples — no clean
      fixed offset is assumed)
    * integrate gyroscope for orientation and accelerometer for translation
      (gravity removal + drift handling) to recover head/camera pose over time

Output is intended to feed `stabilize.py`, which uses head pose to move
camera-frame hand/arm motion into a stable world frame.

This module is a STUB: it states intent and contains no real logic yet.
"""


def main() -> None:
    raise NotImplementedError("imu_sync.py is a stub — no logic implemented yet")


if __name__ == "__main__":
    main()
