"""Retarget recovered human joint angles onto Unitree G1 joint targets.

Intent
------
Map the per-frame human joint-angle trajectory produced by
`data_pipeline/build_dataset.py` onto joint targets for the Unitree G1
humanoid, so the robot reproduces the demonstrated motion in simulation.

Open problem — human -> humanoid DoF mismatch
---------------------------------------------
The human arm/hand DoF recovered from MediaPipe do NOT correspond one-to-one to
the G1's actuated joints (different kinematic chains, joint limits, link
lengths, and a non-anthropomorphic hand/gripper). Retargeting therefore is not a
direct copy of angles; it requires a chosen correspondence and likely IK and/or
optimization to respect the G1's limits while preserving end-effector intent.
The exact mapping is left open here and is part of the method to be designed.

This module is a STUB: it states intent and contains no real logic yet.
"""


def main() -> None:
    raise NotImplementedError("retarget.py is a stub — no logic implemented yet")


if __name__ == "__main__":
    main()
