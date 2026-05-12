"""Drive-to-goal task: navigate the DDSM115 4WD-A to a random xy goal.

A goal is sampled at distance ``[1, 3]`` m from the spawn at a random
heading. The episode ends when the robot is within ``goal_radius`` of the
goal (success), flips over (fail), or the episode-length cap fires.

Observation appends the goal vector in the robot body frame (2 values):
``(dx_forward, dy_right)``. Reward is shaped progress + alive bonus -
control cost + success bonus.
"""

from __future__ import annotations

import jax
import jax.numpy as jp
from mujoco import mjx

from .base import DDSM115Base, quat_to_rot_mat


GOAL_MIN_R = 1.0
GOAL_MAX_R = 3.0
GOAL_RADIUS = 0.15
SUCCESS_BONUS = 50.0
FLIP_PENALTY = -20.0
ALIVE_REWARD = 0.05
CTRL_COST_WEIGHT = 0.001
PROGRESS_WEIGHT = 5.0


def _goal_vec_body(data: mjx.Data, goal_xy: jp.ndarray, free_qpos_adr: int) -> jp.ndarray:
    quat = data.qpos[free_qpos_adr + 3:free_qpos_adr + 7]
    pos = data.qpos[free_qpos_adr:free_qpos_adr + 3]
    R = quat_to_rot_mat(quat)
    delta_world = jp.array([goal_xy[0] - pos[0], goal_xy[1] - pos[1], 0.0])
    delta_body = R.T @ delta_world
    return delta_body[:2]   # (forward = chassis +Y? -- careful: we expose body-frame components)


class DriveToGoal(DDSM115Base):
    task_obs_size = 2

    def _default_metrics(self) -> dict:
        z = jp.zeros(())
        return {"progress": z, "dist": z, "success": z}

    def _task_reset_info(self, rng: jax.Array) -> dict:
        rng_r, rng_a = jax.random.split(rng)
        r = jax.random.uniform(rng_r, minval=GOAL_MIN_R, maxval=GOAL_MAX_R)
        ang = jax.random.uniform(rng_a, minval=-jp.pi, maxval=jp.pi)
        goal_xy = jp.array([r * jp.cos(ang), r * jp.sin(ang)])
        info = {
            "goal_xy": goal_xy,
            "prev_dist": r,
            "success": jp.zeros((), dtype=jp.bool_),
        }
        return info

    def _task_obs(self, data: mjx.Data, info: dict) -> jp.ndarray:
        return _goal_vec_body(data, info["goal_xy"], self._free_qpos_adr)

    def _task_reward(self, data_prev, data, action, info):
        pos = data.qpos[self._free_qpos_adr:self._free_qpos_adr + 3]
        dist = jp.linalg.norm(pos[:2] - info["goal_xy"])
        prev_dist = info["prev_dist"]
        progress = (prev_dist - dist) * PROGRESS_WEIGHT
        ctrl_cost = CTRL_COST_WEIGHT * jp.sum(jp.square(action))
        reached = dist < GOAL_RADIUS
        success_bonus = jp.where(reached, SUCCESS_BONUS, 0.0)
        flipped = self._is_flipped(data)
        flip_pen = jp.where(flipped, FLIP_PENALTY, 0.0)
        reward = progress + ALIVE_REWARD - ctrl_cost + success_bonus + flip_pen
        # Stash updated prev_dist + success into info via metrics (info is
        # mutated by the caller in step()).
        info["prev_dist"] = dist
        info["success"] = jp.logical_or(info["success"], reached)
        metrics = {
            "progress": progress,
            "dist": dist,
            "success": info["success"].astype(jp.float32),
        }
        return reward, metrics

    def _task_done(self, data: mjx.Data, info: dict) -> jp.ndarray:
        return info["success"]
