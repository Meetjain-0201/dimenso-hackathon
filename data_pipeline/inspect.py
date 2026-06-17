"""Inspect a single raw recording before building anything on top of it.

Intent
------
Load ONE recording (head-cam RGB video + head IMU JSON) and report what we
actually have, so downstream alignment is grounded in observed data rather
than assumptions:

    * decode the video and report frame count, resolution, and measured fps
    * load the IMU JSON and report sample count and nominal rate (~100 Hz)
    * print the RAW timestamps for both streams (first/last, span, cadence)

We must NOT assume a clean, fixed video<->IMU offset. The two streams may start
at different times, drift, or drop samples. Quantifying that mismatch here is
itself part of the task and feeds directly into `imu_sync.py`.

This module is a STUB: it states intent and contains no real logic yet.
"""


def main() -> None:
    raise NotImplementedError("inspect.py is a stub — no logic implemented yet")


if __name__ == "__main__":
    main()
