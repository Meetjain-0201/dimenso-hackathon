"""Assemble clean per-frame trajectories into documented output files.

Intent
------
Combine the stabilized, per-frame human motion into a single clean dataset that
the `method/` retargeting code can consume:

    * per-frame joint-angle trajectory (body + arm + hand angles)
    * a gripper open/close signal derived from hand landmarks
      (e.g. thumb-to-fingertip distance thresholded into open/closed)
    * aligned timestamps / frame indices

Output formats: .npz (numeric arrays) and/or .json (metadata + schema).

Documented schema (draft — finalize once a recording is inspected)
------------------------------------------------------------------
    timestamps   : float[T]            seconds, frame-aligned
    joint_angles : float[T, J]         radians; J joints, names in `joint_names`
    joint_names  : str[J]              ordered angle labels
    gripper      : float[T] or int[T]  open/close signal in [0,1] or {0,1}
    fps          : float               source video fps
    meta         : dict                recording id, source paths, units, frame

This module is a STUB: it states intent and contains no real logic yet.
"""


def main() -> None:
    raise NotImplementedError("build_dataset.py is a stub — no logic implemented yet")


if __name__ == "__main__":
    main()
