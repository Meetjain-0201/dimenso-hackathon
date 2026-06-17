"""Replay a retargeted joint trajectory on the Unitree G1 in MuJoCo.

Intent
------
Load the Unitree G1 humanoid in MuJoCo and play back the retargeted joint
trajectory produced by `method/retarget.py`, so the recovered human motion can
be visually verified on the robot in simulation (motion translation only — no
object interaction).

    * load the G1 MJCF model
    * step the simulation while commanding the retargeted per-frame joint targets
    * render / record the replay for the report

TODO: source the Unitree G1 MJCF model from mujoco_menagerie
      (https://github.com/google-deepmind/mujoco_menagerie, `unitree_g1/`)
      and reference it here; it is not vendored in this repo.

This module is a STUB: it states intent and contains no real logic yet.
"""


def main() -> None:
    raise NotImplementedError("replay_g1.py is a stub — no logic implemented yet")


if __name__ == "__main__":
    main()
