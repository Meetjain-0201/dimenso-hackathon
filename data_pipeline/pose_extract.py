"""Extract human body + hand landmarks per frame with MediaPipe.

Intent
------
Run MediaPipe Pose and Hands on every decoded video frame to recover the
human demonstrator's motion in the camera frame:

    * body landmarks (MediaPipe Pose) for arms/torso
    * wrist + finger landmarks (MediaPipe Hands) for each detected hand
    * derived joint angles (e.g. shoulder/elbow/wrist) computed from landmarks

Outputs per-frame landmark sets and joint angles consumed by `stabilize.py`
(to move them into a world frame) and `build_dataset.py` (to assemble
trajectories). Detection gaps / low-confidence frames are expected and should
be surfaced rather than silently filled.

This module is a STUB: it states intent and contains no real logic yet.
"""


def main() -> None:
    raise NotImplementedError("pose_extract.py is a stub — no logic implemented yet")


if __name__ == "__main__":
    main()
