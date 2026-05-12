"""Velocity-tracking task: track a commanded body-frame (vx, omega_z).

Skid-steer locomotion benchmark. A command pair ``(vx_cmd, omega_z_cmd)`` is
sampled at episode reset (and optionally re-sampled mid-episode). The agent
must match those targets in the body frame. Reward uses Gaussian shaping on
the two errors plus mild control / smoothness costs.

Note on the chassis convention from the MJCF: **+Y is forward** in the
chassis frame. The base obs returns body-frame velocity ``v_body`` with
``v_body[1]`` as the forward component. We treat that as ``vx_actual`` in
this task (the conventional "x = forward" for locomotion command).
"""

from __future__ import annotations

import jax
import jax.numpy as jp
from mujoco import mjx

from .base import DDSM115Base, quat_to_rot_mat


VX_RANGE = (-1.0, 1.0)         # m/s
WZ_RANGE = (-1.5, 1.5)         # rad/s
TRACK_SIGMA_V = 0.25
TRACK_SIGMA_W = 0.5
CTRL_COST_WEIGHT = 0.001
SMOOTH_WEIGHT = 0.005
ALIVE_REWARD = 0.05
FLIP_PENALTY = -20.0


def _body_velocity(data: mjx.Data, free_qpos_adr: int, free_qvel_adr: int) -> jp.ndarray:
    quat = data.qpos[free_qpos_adr + 3:free_qpos_adr + 7]
    v_world = data.qvel[free_qvel_adr:free_qvel_adr + 3]
    w_world = data.qvel[free_qvel_adr + 3:free_qvel_adr + 6]
    R = quat_to_rot_mat(quat)
    return R.T @ v_world, R.T @ w_world


class VelocityTracking(DDSM115Base):
    task_obs_size = 2   # (vx_cmd, wz_cmd) appended to base obs

    def _default_metrics(self) -> dict:
        z = jp.zeros(())
        return {"vx_reward": z, "wz_reward": z, "vx_err": z, "wz_err": z}

    def _task_reset_info(self, rng: jax.Array) -> dict:
        rng_v, rng_w = jax.random.split(rng)
        vx_cmd = jax.random.uniform(rng_v, minval=VX_RANGE[0], maxval=VX_RANGE[1])
        wz_cmd = jax.random.uniform(rng_w, minval=WZ_RANGE[0], maxval=WZ_RANGE[1])
        return {
            "vx_cmd": vx_cmd,
            "wz_cmd": wz_cmd,
            "prev_action": jp.zeros(4),
        }

    def _task_obs(self, data: mjx.Data, info: dict) -> jp.ndarray:
        return jp.array([info["vx_cmd"], info["wz_cmd"]])

    def _task_reward(self, data_prev, data, action, info):
        v_body, w_body = _body_velocity(
            data, self._free_qpos_adr, self._free_qvel_adr)
        # chassis +Y is forward (see module docstring)
        vx_actual = v_body[1]
        wz_actual = w_body[2]
        vx_err = vx_actual - info["vx_cmd"]
        wz_err = wz_actual - info["wz_cmd"]
        vx_reward = jp.exp(-(vx_err * vx_err) / (TRACK_SIGMA_V ** 2))
        wz_reward = jp.exp(-(wz_err * wz_err) / (TRACK_SIGMA_W ** 2))
        ctrl_cost = CTRL_COST_WEIGHT * jp.sum(jp.square(action))
        smooth_cost = SMOOTH_WEIGHT * jp.sum(jp.square(action - info["prev_action"]))
        flipped = self._is_flipped(data)
        flip_pen = jp.where(flipped, FLIP_PENALTY, 0.0)
        reward = vx_reward + wz_reward + ALIVE_REWARD - ctrl_cost - smooth_cost + flip_pen
        info["prev_action"] = action
        metrics = {
            "vx_reward": vx_reward,
            "wz_reward": wz_reward,
            "vx_err": vx_err,
            "wz_err": wz_err,
        }
        return reward, metrics
