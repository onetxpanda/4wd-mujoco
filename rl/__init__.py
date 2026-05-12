"""MJX-based RL environments for the DDSM115 4WD-A.

Two task envs share a common base:
  * :class:`DriveToGoal`        -- navigate to a random xy goal
  * :class:`VelocityTracking`   -- track a commanded (vx, omega_z)

Both subclass :class:`DDSM115Base`, a brax-compatible MJX env that owns the
compiled MJCF and the physics-stepping loop. Each step advances physics
``n_frames`` times so the control rate is decoupled from the integrator
timestep.

Conventions:
  * action[i] in [-1, 1] is scaled to the motor ctrlrange of [-3, 3]
  * observation is a flat ``jax.Array``; task-specific extras are appended
  * reward and termination are entirely jax-side so the whole env JITs
"""

from .base import DDSM115Base, find_mjcf
from .drive_to_goal import DriveToGoal
from .velocity_tracking import VelocityTracking

__all__ = ["DDSM115Base", "DriveToGoal", "VelocityTracking", "find_mjcf"]
