"""Retarget recovered human motion onto G1 + Inspire joint targets (DLS IK).

Pipeline stage 4. Per frame, given approximate 3D wrist targets, a waist lean
target, per-arm hand-present flags and per-finger curl, produce a full
G1+Inspire qpos target:

  * Damped-least-squares IK (MuJoCo body Jacobians) drives each PRESENT wrist
    (hand-base body) to its 3D target. Both arms are solved JOINTLY because they
    share the 3 waist joints, which are included as shared IK DoF.
  * The waist is SOFT-biased toward the head-IMU lean target (an extra weighted
    task row), so torso lean and arm reach cooperate rather than fight.
  * Fingers: per-finger curl → Inspire finger joint angles (independent joints,
    no 6→12 coupling, matching the model). Thumb opposition from pinch.
  * STOW per arm when its hand isn't present: upper arm down, forearm forward
    (elbow≈0 → ~90° L), wrists 0, fingers relaxed open. A hold-last-valid window
    bridges brief dropouts before easing to stow.
  * Safety: per-joint rate limiting so motion eases (esp. on dropout/stow).

Names/indices are all read from the model — nothing hardcoded.
"""
from __future__ import annotations
import numpy as np
import mujoco

# ── tunables ────────────────────────────────────────────────────────────────
DLS_LAMBDA = 0.12        # IK damping (larger = stabler, slower convergence)
IK_ITERS = 8             # Levenberg iterations per frame
WAIST_BIAS_W = 0.4       # weight of the soft waist→lean task in the IK stack
HOLD_FRAMES = 6          # frames to hold last-valid arm pose before stowing
RATE_ARM = 0.18          # max |Δ| per frame (rad) for arm/waist joints
RATE_FINGER = 0.35       # max |Δ| per frame (rad) for finger joints
PINCH_REF = 0.12         # pinch (norm units) below which thumb fully opposes

_ARM_J = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
          "wrist_roll", "wrist_pitch", "wrist_yaw"]
_WAIST_J = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
STOW = {"shoulder_pitch": 0.0, "shoulder_roll": 0.0, "shoulder_yaw": 0.0,
        "elbow": 0.0, "wrist_roll": 0.0, "wrist_pitch": 0.0, "wrist_yaw": 0.0}


class Retargeter:
    def __init__(self, model_xml):
        self.m = mujoco.MjModel.from_xml_path(str(model_xml))
        self.d = mujoco.MjData(self.m)
        m = self.m
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        self.qadr = lambda n: m.jnt_qposadr[jid(n)]
        self.dadr = lambda n: m.jnt_dofadr[jid(n)]
        self.jrange = lambda n: m.jnt_range[jid(n)]

        self.arm_j = {s: [f"{s}_{j}_joint" for j in _ARM_J] for s in ("left", "right")}
        self.waist_j = list(_WAIST_J)
        self.hand_body = {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                          f"{'L' if s == 'left' else 'R'}_hand_base_link") for s in ("left", "right")}
        # finger joints grouped by side, by mediapipe finger
        self.finger_j = {}
        for s, pfx in (("left", "L"), ("right", "R")):
            self.finger_j[s] = {
                "thumb": [f"{pfx}_thumb_proximal_pitch_joint", f"{pfx}_thumb_intermediate_joint",
                          f"{pfx}_thumb_distal_joint"],
                "thumb_yaw": [f"{pfx}_thumb_proximal_yaw_joint"],
                "index": [f"{pfx}_index_proximal_joint", f"{pfx}_index_intermediate_joint"],
                "middle": [f"{pfx}_middle_proximal_joint", f"{pfx}_middle_intermediate_joint"],
                "ring": [f"{pfx}_ring_proximal_joint", f"{pfx}_ring_intermediate_joint"],
                "pinky": [f"{pfx}_pinky_proximal_joint", f"{pfx}_pinky_intermediate_joint"],
            }
        # stand keyframe = base pose (legs frozen here forever)
        mujoco.mj_resetDataKeyframe(m, self.d, 0)
        self.q = self.d.qpos.copy()
        self.miss = {"left": HOLD_FRAMES + 1, "right": HOLD_FRAMES + 1}
        self.last_arm_q = {s: np.array([self.q[self.qadr(j)] for j in self.arm_j[s]]) for s in ("left", "right")}

    # ── finger curl → joint targets ─────────────────────────────────────────
    def _set_fingers(self, q, side, curl, pinch):
        fg = self.finger_j[side]
        order = {"thumb": 0, "index": 1, "middle": 2, "ring": 3, "pinky": 4}
        for fname, k in order.items():
            for j in fg[fname]:
                lo, hi = self.jrange(j)
                q[self.qadr(j)] = lo + float(np.clip(curl[k], 0, 1)) * (hi - lo)
        # thumb opposition from pinch: tighter pinch → more yaw toward index
        opp = 1.0 - float(np.clip((pinch if np.isfinite(pinch) else PINCH_REF) / PINCH_REF, 0, 1))
        for j in fg["thumb_yaw"]:
            lo, hi = self.jrange(j)
            q[self.qadr(j)] = lo + opp * (hi - lo)

    def _open_fingers(self, q, side):
        for fname, js in self.finger_j[side].items():
            for j in js:
                lo, hi = self.jrange(j)
                q[self.qadr(j)] = lo          # lower limit = relaxed open

    # ── DLS IK over waist + present arms ──────────────────────────────────────
    def _ik(self, q, active, targets, lean):
        m, d = self.m, self.d
        dofs = [self.dadr(j) for j in self.waist_j]
        for s in active:
            dofs += [self.dadr(j) for j in self.arm_j[s]]
        dofs = np.array(dofs)
        waist_dofs = [self.dadr(j) for j in self.waist_j]
        waist_q_idx = [self.qadr(j) for j in self.waist_j]
        jacp = np.zeros((3, m.nv))
        for _ in range(IK_ITERS):
            d.qpos[:] = q
            mujoco.mj_forward(m, d)
            rows_J, rows_e = [], []
            for s in active:
                mujoco.mj_jacBody(m, d, jacp, None, self.hand_body[s])
                rows_J.append(jacp[:, dofs])
                rows_e.append(targets[s] - d.xpos[self.hand_body[s]])
            # soft waist→lean bias rows (selector on waist dofs)
            sel = np.zeros((3, len(dofs)))
            for r, wd in enumerate(waist_dofs):
                sel[r, list(dofs).index(wd)] = WAIST_BIAS_W
            rows_J.append(sel)
            rows_e.append(WAIST_BIAS_W * (lean - q[waist_q_idx]))
            J = np.vstack(rows_J); e = np.concatenate(rows_e)
            JJt = J @ J.T + (DLS_LAMBDA ** 2) * np.eye(J.shape[0])
            dq = J.T @ np.linalg.solve(JJt, e)
            # apply + clamp to joint ranges
            qd = q.copy()
            for k, dof in enumerate(dofs):
                # map dof back to its qpos adr (hinge: qadr == find joint with this dof)
                jadr = self._dof_to_qadr(dof)
                qd[jadr] += dq[k]
            self._clamp(qd, active)
            q = qd
        return q

    def _dof_to_qadr(self, dof):
        if not hasattr(self, "_dof2q"):
            self._dof2q = {}
            for jn in self.waist_j + self.arm_j["left"] + self.arm_j["right"]:
                self._dof2q[self.dadr(jn)] = self.qadr(jn)
        return self._dof2q[dof]

    def _clamp(self, q, active):
        for jn in self.waist_j + sum((self.arm_j[s] for s in active), []):
            lo, hi = self.jrange(jn)
            a = self.qadr(jn)
            q[a] = np.clip(q[a], lo, hi)

    # ── main per-frame entry ─────────────────────────────────────────────────
    def solve_frame(self, targets, lean, present, curl, pinch):
        """targets:{side:(3,)} lean:(3,) present:{side:bool} curl:{side:(5,)} pinch:{side:float}.
        Returns a full qpos target (rate-limited from the previous frame)."""
        q_des = self.q.copy()                      # warm start; legs/base stay frozen
        active = [s for s in ("left", "right") if present[s] and np.all(np.isfinite(targets[s]))]

        if active:
            q_des = self._ik(q_des, active, targets, lean)
        else:
            # no arms: still lean the waist toward the IMU target
            for r, j in enumerate(self.waist_j):
                lo, hi = self.jrange(j); q_des[self.qadr(j)] = np.clip(lean[r], lo, hi)

        # per-arm: commit IK result / hold-last-valid / ease to stow
        for s in ("left", "right"):
            if s in active:
                self.miss[s] = 0
                self.last_arm_q[s] = np.array([q_des[self.qadr(j)] for j in self.arm_j[s]])
            else:
                self.miss[s] += 1
                if self.miss[s] <= HOLD_FRAMES:    # hold last valid briefly
                    for j, v in zip(self.arm_j[s], self.last_arm_q[s]):
                        q_des[self.qadr(j)] = v
                else:                               # ease to stow
                    for jk, jn in zip(_ARM_J, self.arm_j[s]):
                        q_des[self.qadr(jn)] = STOW[jk]
            # fingers
            if present[s]:
                self._set_fingers(q_des, s, curl[s], pinch[s])
            else:
                self._open_fingers(q_des, s)

        # ── rate-limit every controlled joint from the previous committed pose ──
        q_new = self.q.copy()
        def limit(joints, cap):
            for jn in joints:
                a = self.qadr(jn)
                q_new[a] = self.q[a] + np.clip(q_des[a] - self.q[a], -cap, cap)
        limit(self.waist_j + self.arm_j["left"] + self.arm_j["right"], RATE_ARM)
        for s in ("left", "right"):
            limit(sum(self.finger_j[s].values(), []), RATE_FINGER)
        self.q = q_new
        return q_new.copy()


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Retargeter self-test (loads model, one stow frame)")
    ap.add_argument("--model", default="sim/assets/scene_g1_inspire.xml")
    args = ap.parse_args()
    rt = Retargeter(args.model)
    q = rt.solve_frame({"left": np.full(3, np.nan), "right": np.full(3, np.nan)},
                       np.zeros(3), {"left": False, "right": False},
                       {"left": np.zeros(5), "right": np.zeros(5)},
                       {"left": np.nan, "right": np.nan})
    print(f"OK: model nu={rt.m.nu} nq={rt.m.nq}; stow qpos len={len(q)}")


if __name__ == "__main__":
    main()
