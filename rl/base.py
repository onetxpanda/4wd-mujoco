"""Common MJX env base for DDSM115 4WD-A tasks.

Brax-compatible: subclasses return ``brax.envs.base.State`` and expose
``observation_size`` / ``action_size`` so brax PPO can train on them
directly without further wrapping.
"""

from __future__ import annotations

import os
from typing import Tuple

import jax
import jax.numpy as jp
import mujoco
from brax.envs.base import Env, State
from mujoco import mjx


# Default control parameters.
DEFAULT_N_FRAMES = 10        # physics substeps per env.step (0.002 * 10 = 50 Hz)
DEFAULT_EPISODE_LEN = 500    # steps per episode  (50 Hz * 10 s)
MOTOR_CTRL_LIMIT = 3.0       # ctrlrange="-3 3" in the MJCF
ACTION_SCALE = MOTOR_CTRL_LIMIT


def find_mjcf() -> str:
    """Locate ``ddsm115_4wd.xml`` next to the repo root.

    Search order: ``$DDSM115_MJCF``, repo root (one level above this file),
    cwd. Raises FileNotFoundError if not found.
    """
    env_path = os.environ.get("DDSM115_MJCF")
    if env_path and os.path.exists(env_path):
        return env_path
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "..", "ddsm115_4wd.xml"),
        os.path.join(os.getcwd(), "ddsm115_4wd.xml"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    raise FileNotFoundError("ddsm115_4wd.xml not found; set DDSM115_MJCF")


def quat_to_rot_mat(q: jp.ndarray) -> jp.ndarray:
    """MuJoCo-order quat (w, x, y, z) -> 3x3 rotation matrix."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    return jp.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


class DDSM115Base(Env):
    """MJX env for the DDSM115 4WD-A robot.

    Subclasses implement :meth:`_task_obs`, :meth:`_task_reward`,
    :meth:`_task_reset_info`, and :meth:`_task_done`.
    """

    # Length of the task-specific observation slice; subclasses override.
    task_obs_size: int = 0

    def __init__(
        self,
        mjcf_path: str | None = None,
        n_frames: int = DEFAULT_N_FRAMES,
        episode_len: int = DEFAULT_EPISODE_LEN,
    ):
        path = mjcf_path or find_mjcf()
        self._mj_model = mujoco.MjModel.from_xml_path(path)
        self._mjx_model = mjx.put_model(self._mj_model)
        self._n_frames = int(n_frames)
        self._episode_len = int(episode_len)

        # Cache joint / sensor / body addresses for fast obs assembly.
        m = self._mj_model

        # Free joint qpos is the first 7 entries (pos 3 + quat 4); qvel 6.
        root_jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "root")
        self._free_qpos_adr = int(m.jnt_qposadr[root_jid])
        self._free_qvel_adr = int(m.jnt_dofadr[root_jid])

        # Wheel hinge qvel addresses, in a canonical order.
        wheel_names = ["wheel_fr", "wheel_fl", "wheel_br", "wheel_bl"]
        self._wheel_qvel_adrs = jp.array([
            int(m.jnt_dofadr[
                mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{n}_hinge")])
            for n in wheel_names
        ])

        # Actuator order matches the order they appear in the MJCF.
        # Build a permutation mapping our wheel order to actuator indices.
        actuator_names = [
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(m.nu)
        ]
        self._act_perm = jp.array([
            actuator_names.index(f"motor_{n}") for n in wheel_names
        ])

        self._base_body_id = int(mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_BODY, "base_link"))

        # qpos0 / qvel0 captured once for resetting.
        self._qpos0 = jp.array(m.qpos0)
        self._qvel0 = jp.zeros(m.nv)

    # --- brax.envs.base.Env interface -----------------------------------

    @property
    def action_size(self) -> int:
        return 4

    @property
    def observation_size(self) -> int:
        return 4 + 1 + 3 + 3 + 4 + self.task_obs_size

    @property
    def backend(self) -> str:
        return "mjx"

    # --- physics helpers ------------------------------------------------

    def _initial_data(self) -> mjx.Data:
        data = mjx.make_data(self._mjx_model)
        data = data.replace(qpos=self._qpos0, qvel=self._qvel0)
        return mjx.forward(self._mjx_model, data)

    def _step_physics(self, data: mjx.Data, action: jp.ndarray) -> mjx.Data:
        # Permute action[fr, fl, br, bl] -> actuator order, scale to ctrl.
        action = jp.clip(action, -1.0, 1.0)
        ctrl = jp.zeros(self._mj_model.nu)
        ctrl = ctrl.at[self._act_perm].set(action * ACTION_SCALE)
        data = data.replace(ctrl=ctrl)
        # Substep n_frames times.
        def body(d, _):
            return mjx.step(self._mjx_model, d), None
        data, _ = jax.lax.scan(body, data, jp.arange(self._n_frames))
        return data

    def _base_obs(self, data: mjx.Data) -> jp.ndarray:
        """Proprioception: orientation + height + body-frame velocities + wheels."""
        qpos = data.qpos
        qvel = data.qvel
        quat = qpos[self._free_qpos_adr + 3: self._free_qpos_adr + 7]
        z = qpos[self._free_qpos_adr + 2:self._free_qpos_adr + 3]
        # World-frame linear/angular velocity.
        v_world = qvel[self._free_qvel_adr:self._free_qvel_adr + 3]
        w_world = qvel[self._free_qvel_adr + 3:self._free_qvel_adr + 6]
        # Rotate into body frame.
        R = quat_to_rot_mat(quat)
        v_body = R.T @ v_world
        w_body = R.T @ w_world
        wheel_w = qvel[self._wheel_qvel_adrs]
        return jp.concatenate([quat, z, v_body, w_body, wheel_w])

    def _is_flipped(self, data: mjx.Data) -> jp.ndarray:
        """True if the chassis +Z has tilted past ~60 deg from world +Z."""
        quat = data.qpos[self._free_qpos_adr + 3:self._free_qpos_adr + 7]
        R = quat_to_rot_mat(quat)
        up = R[:, 2]
        return up[2] < 0.5  # cos(60 deg)

    # --- subclass hooks -------------------------------------------------

    def _task_reset_info(self, rng: jax.Array) -> dict:
        """Return per-episode task state (e.g. goal position, command)."""
        raise NotImplementedError

    def _task_obs(self, data: mjx.Data, info: dict) -> jp.ndarray:
        """Task-specific obs slice. Length must equal :attr:`task_obs_size`."""
        raise NotImplementedError

    def _task_reward(
        self, data_prev: mjx.Data, data: mjx.Data, action: jp.ndarray, info: dict,
    ) -> Tuple[jp.ndarray, dict]:
        """Return (reward_scalar, metrics_dict)."""
        raise NotImplementedError

    def _task_done(self, data: mjx.Data, info: dict) -> jp.ndarray:
        return jp.array(False)

    def _default_metrics(self) -> dict:
        """All keys that :meth:`_task_reward` will return, seeded with zeros.

        Brax's training wrappers carry ``state.metrics`` through ``lax.scan``
        and require the pytree structure to stay invariant; pre-seeding here
        keeps reset and step in sync.
        """
        return {}

    # --- brax env methods -----------------------------------------------

    def reset(self, rng: jax.Array) -> State:
        rng, sub = jax.random.split(rng)
        data = self._initial_data()
        info = self._task_reset_info(sub)
        info["step"] = jp.zeros((), dtype=jp.int32)
        info["rng"] = rng
        obs = jp.concatenate([self._base_obs(data), self._task_obs(data, info)])
        return State(
            pipeline_state=data,
            obs=obs,
            reward=jp.zeros(()),
            done=jp.zeros(()),
            metrics={"reward": jp.zeros(()), **self._default_metrics()},
            info=info,
        )

    def step(self, state: State, action: jp.ndarray) -> State:
        data_prev = state.pipeline_state
        data = self._step_physics(data_prev, action)

        reward, metrics = self._task_reward(data_prev, data, action, state.info)
        flipped = self._is_flipped(data)
        task_done = self._task_done(data, state.info)
        step = state.info["step"] + 1
        timed_out = step >= self._episode_len
        done = jp.logical_or(jp.logical_or(flipped, task_done), timed_out).astype(jp.float32)

        info = dict(state.info)
        info["step"] = step
        # Subclasses may update info via metrics return; merge them.

        obs = jp.concatenate([self._base_obs(data), self._task_obs(data, info)])
        metrics = {"reward": reward, **metrics}
        return state.replace(
            pipeline_state=data, obs=obs, reward=reward, done=done,
            metrics=metrics, info=info,
        )
