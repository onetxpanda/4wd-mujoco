#!/usr/bin/env python3
"""Smoke-test both MJX envs: JIT-compile reset/step and run a short rollout.

Run from the repo root::

    python3 -m rl.test_envs

This confirms the envs trace, compile, and produce sane numbers; it does not
train anything.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jp

from . import DriveToGoal, VelocityTracking


def rollout(env, name: str, steps: int = 50) -> None:
    print(f"--- {name} (obs_dim={env.observation_size}, act_dim={env.action_size}) ---")
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)

    rng = jax.random.PRNGKey(0)
    t0 = time.time()
    state = reset(rng)
    state.obs.block_until_ready()
    print(f"  reset JIT+run: {time.time() - t0:.2f}s, obs.shape={state.obs.shape}")

    # Single step trace
    action = jp.zeros(env.action_size)
    t0 = time.time()
    state = step(state, action)
    state.obs.block_until_ready()
    print(f"  first step JIT+run: {time.time() - t0:.2f}s, reward={float(state.reward):+.4f}")

    # Time per step after JIT
    t0 = time.time()
    for _ in range(steps):
        state = step(state, action)
    state.obs.block_until_ready()
    dt = (time.time() - t0) / steps
    print(f"  {steps} more steps: {dt * 1e3:.2f}ms / step")

    # Drive at +1 forward command-equivalent (ctrl=+1 all): expect +Y motion.
    state = reset(rng)
    a = jp.ones(env.action_size)
    for _ in range(100):
        state = step(state, a)
    pos = state.pipeline_state.qpos[:3]
    print(f"  after 100 steps ctrl=+1: pos=({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})")


def main() -> None:
    rollout(DriveToGoal(), "DriveToGoal")
    rollout(VelocityTracking(), "VelocityTracking")
    print("OK")


if __name__ == "__main__":
    main()
