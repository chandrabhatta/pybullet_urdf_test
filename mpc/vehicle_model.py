"""
Vehicle kinematic models for RRT planning.
Supports car-only (bicycle model) and car+trailer systems.
"""

import numpy as np
from typing import Tuple, List


class BicycleModel:
    """
    Simple bicycle model for car-only planning.

    State: [x, y, theta]
        x, y: rear axle position
        theta: heading angle

    Control: [v, delta]
        v: forward velocity (m/s)
        delta: steering angle (rad)
    """

    def __init__(self, wheelbase: float = 2.5, max_steer: float = 0.5):
        self.L = wheelbase  # Wheelbase (m)
        self.max_steer = max_steer  # Max steering angle (rad)

    def dynamics(self, state: np.ndarray, control: Tuple[float, float]) -> np.ndarray:
        """
        Compute state derivatives.

        Args:
            state: [x, y, theta]
            control: (v, delta)

        Returns:
            [dx/dt, dy/dt, dtheta/dt]
        """
        x, y, theta = state
        v, delta = control

        # Clamp steering angle
        delta = np.clip(delta, -self.max_steer, self.max_steer)

        dx = v * np.cos(theta)
        dy = v * np.sin(theta)
        dtheta = v * np.tan(delta) / self.L

        return np.array([dx, dy, dtheta])

    def simulate(self, state: np.ndarray, control: Tuple[float, float],
                 dt: float = 0.1, duration: float = 0.5) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Forward simulate the vehicle.

        Args:
            state: Initial state [x, y, theta]
            control: (v, delta) velocity and steering
            dt: Integration timestep
            duration: Total simulation time

        Returns:
            (final_state, trajectory) where trajectory is list of intermediate states
        """
        state = np.array(state, dtype=float)
        trajectory = [state.copy()]

        steps = int(duration / dt)
        for _ in range(steps):
            # Simple Euler integration
            dstate = self.dynamics(state, control)
            state = state + dstate * dt

            # Normalize angle to [-pi, pi]
            state[2] = np.arctan2(np.sin(state[2]), np.cos(state[2]))

            trajectory.append(state.copy())

        return state, trajectory

    def state_dim(self) -> int:
        return 3

    def get_car_corners(self, state: np.ndarray,
                        length: float = 3.0, width: float = 1.6) -> np.ndarray:
        """
        Get the 4 corners of the car rectangle in world coordinates.

        Args:
            state: [x, y, theta] - rear axle position and heading
            length: Car length (m)
            width: Car width (m)

        Returns:
            4x2 array of corner positions
        """
        x, y, theta = state

        # Car extends from rear axle
        # Rear is at origin, front is at +length in local frame
        # Actually, let's center the box on the rear axle
        rear_offset = 0.5  # How far rear extends behind axle
        front_offset = length - rear_offset

        # Local corners (relative to rear axle, before rotation)
        local_corners = np.array([
            [-rear_offset, -width/2],   # Rear left
            [front_offset, -width/2],   # Front left
            [front_offset, width/2],    # Front right
            [-rear_offset, width/2],    # Rear right
        ])

        # Rotation matrix
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([[cos_t, -sin_t],
                      [sin_t, cos_t]])

        # Transform to world coordinates
        world_corners = (R @ local_corners.T).T + np.array([x, y])

        return world_corners


class CarTrailerModel:
    """
    Car + Trailer kinematic model.

    State: [x1, y1, theta0, theta1]
        x1, y1: trailer rear axle position
        theta0: tractor heading
        theta1: trailer heading

    Control: [v, delta]
        v: forward velocity at tractor rear axle (m/s)
        delta: steering angle (rad)
    """

    def __init__(self, L0: float = 2.5, L1: float = 3.0, max_steer: float = 0.5):
        self.L0 = L0  # Tractor wheelbase
        self.L1 = L1  # Trailer length (hitch to rear axle)
        self.max_steer = max_steer

    def dynamics(self, state: np.ndarray, control: Tuple[float, float]) -> np.ndarray:
        """
        Compute state derivatives for car+trailer system.

        The kinematics are derived from the constraint that the trailer
        hitch point moves with the tractor rear axle.
        """
        x1, y1, theta0, theta1 = state
        v, delta = control

        # Clamp steering
        delta = np.clip(delta, -self.max_steer, self.max_steer)

        # Angle difference
        beta = theta0 - theta1

        # Tractor dynamics
        dtheta0 = v * np.tan(delta) / self.L0

        # Trailer dynamics (from hitch constraint)
        dtheta1 = v * np.sin(beta) / self.L1

        # Trailer rear axle velocity
        dx1 = v * np.cos(beta) * np.cos(theta1)
        dy1 = v * np.cos(beta) * np.sin(theta1)

        return np.array([dx1, dy1, dtheta0, dtheta1])

    def simulate(self, state: np.ndarray, control: Tuple[float, float],
                 dt: float = 0.1, duration: float = 0.5) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Forward simulate the car+trailer system.
        """
        state = np.array(state, dtype=float)
        trajectory = [state.copy()]

        steps = int(duration / dt)
        for _ in range(steps):
            dstate = self.dynamics(state, control)
            state = state + dstate * dt

            # Normalize angles
            state[2] = np.arctan2(np.sin(state[2]), np.cos(state[2]))
            state[3] = np.arctan2(np.sin(state[3]), np.cos(state[3]))

            trajectory.append(state.copy())

        return state, trajectory

    def state_dim(self) -> int:
        return 4

    def get_tractor_position(self, state: np.ndarray) -> np.ndarray:
        """
        Get tractor rear axle position from trailer state.
        """
        x1, y1, theta0, theta1 = state

        # Hitch is at L1 ahead of trailer rear axle
        hitch_x = x1 + self.L1 * np.cos(theta1)
        hitch_y = y1 + self.L1 * np.sin(theta1)

        # Tractor rear axle is at hitch (simplified - in reality there's an offset)
        return np.array([hitch_x, hitch_y, theta0])

    def get_car_corners(self, state: np.ndarray,
                        length: float = 3.0, width: float = 1.6) -> np.ndarray:
        """Get tractor corners."""
        tractor_state = self.get_tractor_position(state)
        x, y, theta = tractor_state

        rear_offset = 0.5
        front_offset = length - rear_offset

        local_corners = np.array([
            [-rear_offset, -width/2],
            [front_offset, -width/2],
            [front_offset, width/2],
            [-rear_offset, width/2],
        ])

        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

        return (R @ local_corners.T).T + np.array([x, y])

    def get_trailer_corners(self, state: np.ndarray,
                            length: float = 2.0, width: float = 1.4) -> np.ndarray:
        """Get trailer corners."""
        x1, y1, theta0, theta1 = state

        rear_offset = 0.3
        front_offset = length - rear_offset

        local_corners = np.array([
            [-rear_offset, -width/2],
            [front_offset, -width/2],
            [front_offset, width/2],
            [-rear_offset, width/2],
        ])

        cos_t, sin_t = np.cos(theta1), np.sin(theta1)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

        return (R @ local_corners.T).T + np.array([x1, y1])
