#!/usr/bin/env python3
"""Smoke-test the compiled MJCF.

Loads ``ddsm115_4wd.xml``, prints nq/nu/mass/wheel-radius, lets the robot
settle for 1 s, then runs a forward-drive episode and a left-yaw episode and
checks the qualitative sign of each.
"""

from __future__ import annotations

import sys

import mujoco
import numpy as np


XML = "ddsm115_4wd.xml"


def step_for(m, d, seconds: float) -> None:
    for _ in range(int(seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)


def main() -> int:
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)

    wheel_r = float(m.geom_size[
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "wheel_br_col"), 0])
    print(f"nq={m.nq}  nu={m.nu}  nbody={m.nbody}")
    print(f"total mass = {m.body_mass.sum():.4f} kg")
    print(f"wheel radius = {wheel_r:.5f} m")

    ctrl_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                  for i in range(m.nu)]
    cmap = {n: i for i, n in enumerate(ctrl_names)}

    # 1. Settle for 1 s under gravity.
    step_for(m, d, 1.0)
    settle_z = float(d.qpos[2])
    print(f"After 1s settle: z={settle_z:.5f} m (expected ~{wheel_r:.5f} m)")

    # 2. Forward drive: ctrl = +1 on all four motors -> robot moves +Y.
    d.ctrl[:] = 1.0
    y0 = float(d.qpos[1])
    step_for(m, d, 2.0)
    dy = float(d.qpos[1]) - y0
    print(f"Forward (ctrl=+1 all) for 2s: dy = {dy:+.4f} m")
    assert dy > 0.05, f"expected positive Y motion, got {dy}"

    # 3. Left yaw: right side +1, left side -1.  Reset, settle, then drive.
    mujoco.mj_resetData(m, d)
    step_for(m, d, 1.0)
    d.ctrl[cmap["motor_wheel_fr"]] = +1
    d.ctrl[cmap["motor_wheel_br"]] = +1
    d.ctrl[cmap["motor_wheel_fl"]] = -1
    d.ctrl[cmap["motor_wheel_bl"]] = -1
    step_for(m, d, 2.0)
    wz = float(d.qvel[5])
    print(f"Yaw cmd for 2s: omega_z = {wz:+.4f} rad/s = {np.degrees(wz):+.2f} deg/s")
    assert wz > 0.1, f"expected positive yaw rate, got {wz}"

    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
