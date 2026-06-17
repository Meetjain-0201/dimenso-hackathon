"""Transform camera-frame motion into a stable world/table frame.

Intent
------
The head-cam moves with the demonstrator, so raw MediaPipe landmarks live in a
constantly-moving camera frame. Using the head/camera pose recovered in
`imu_sync.py`, transform the per-frame hand/arm landmarks (`pose_extract.py`)
out of the camera frame and into a stable world / table-fixed frame:

    * apply the inverse of camera pose to landmark positions per frame
    * yield hand/arm trajectories that are stationary when the demonstrator's
      hands are stationary, even while the head turns

This is what makes retargeting meaningful: the robot should reproduce world-frame
hand motion, not the apparent motion induced by head movement.

This module is a STUB: it states intent and contains no real logic yet.
"""


def main() -> None:
    raise NotImplementedError("stabilize.py is a stub — no logic implemented yet")


if __name__ == "__main__":
    main()
