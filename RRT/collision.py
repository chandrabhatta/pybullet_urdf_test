"""
Collision checking for lane-change planning.
Simple rectangular bounds checking against lane boundaries.
"""

import numpy as np
from typing import List, Optional

try:
    from .vehicle_model import BicycleModel, CarTrailerModel
except ImportError:
    from vehicle_model import BicycleModel, CarTrailerModel


# Lane boundaries (Dutch 2-lane highway)
LANE_Y_MIN = 0.0    # Inner edge
LANE_Y_MAX = 7.0    # Outer edge
ROAD_X_MIN = -10.0  # Allow some space behind start
ROAD_X_MAX = 400.0  # Road length

# Vehicle dimensions
CAR_LENGTH = 3.0
CAR_WIDTH = 1.6
TRAILER_LENGTH = 2.0
TRAILER_WIDTH = 1.4

# Safety margins
SAFETY_MARGIN = 0.2  # Extra buffer around vehicles


class CollisionChecker:
    """
    Collision checker for lane-change scenarios.
    Checks vehicle rectangles against lane boundaries and obstacles.
    """

    def __init__(self,
                 y_min: float = LANE_Y_MIN,
                 y_max: float = LANE_Y_MAX,
                 x_min: float = ROAD_X_MIN,
                 x_max: float = ROAD_X_MAX,
                 safety_margin: float = SAFETY_MARGIN):
        self.y_min = y_min + safety_margin
        self.y_max = y_max - safety_margin
        self.x_min = x_min
        self.x_max = x_max
        self.obstacles = []  # List of (x, y, length, width, yaw) tuples

    def add_obstacle(self, x: float, y: float, length: float = 3.0,
                     width: float = 1.6, yaw: float = 0.0):
        """
        Add a rectangular obstacle at (x, y).

        Args:
            x, y: Center position of obstacle
            length: Length of obstacle (along heading)
            width: Width of obstacle (perpendicular to heading)
            yaw: Heading angle of obstacle
        """
        self.obstacles.append((x, y, length, width, yaw))

    def clear_obstacles(self):
        """Remove all obstacles."""
        self.obstacles = []

    def _get_obstacle_corners(self, x: float, y: float, length: float,
                               width: float, yaw: float) -> np.ndarray:
        """Get the 4 corners of an obstacle rectangle."""
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        half_l = length / 2
        half_w = width / 2

        # Corners in local frame
        local_corners = np.array([
            [half_l, half_w],
            [half_l, -half_w],
            [-half_l, -half_w],
            [-half_l, half_w]
        ])

        # Rotate and translate to world frame
        corners = np.zeros((4, 2))
        for i, (lx, ly) in enumerate(local_corners):
            corners[i, 0] = x + lx * cos_yaw - ly * sin_yaw
            corners[i, 1] = y + lx * sin_yaw + ly * cos_yaw

        return corners

    def _rectangles_overlap(self, corners1: np.ndarray, corners2: np.ndarray) -> bool:
        """
        Check if two rectangles overlap using Separating Axis Theorem (SAT).

        Args:
            corners1, corners2: 4x2 arrays of corner positions

        Returns:
            True if rectangles overlap
        """
        def get_axes(corners):
            axes = []
            for i in range(4):
                edge = corners[(i + 1) % 4] - corners[i]
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm > 1e-6:
                    axes.append(normal / norm)
            return axes

        def project(corners, axis):
            projections = np.dot(corners, axis)
            return np.min(projections), np.max(projections)

        # Get all axes to test
        axes = get_axes(corners1) + get_axes(corners2)

        for axis in axes:
            min1, max1 = project(corners1, axis)
            min2, max2 = project(corners2, axis)

            # Check for separation
            if max1 < min2 or max2 < min1:
                return False  # Found separating axis, no collision

        return True  # No separating axis found, rectangles overlap

    def check_obstacle_collision(self, corners: np.ndarray) -> bool:
        """
        Check if vehicle corners collide with any obstacle.

        Args:
            corners: 4x2 array of vehicle corner positions

        Returns:
            True if collision detected (NOT collision-free)
        """
        for obs in self.obstacles:
            x, y, length, width, yaw = obs
            obs_corners = self._get_obstacle_corners(x, y, length, width, yaw)
            if self._rectangles_overlap(corners, obs_corners):
                return True  # Collision!
        return False  # No collision

    def point_in_bounds(self, x: float, y: float) -> bool:
        """Check if a single point is within lane bounds."""
        return (self.x_min <= x <= self.x_max and
                self.y_min <= y <= self.y_max)

    def corners_in_bounds(self, corners: np.ndarray) -> bool:
        """
        Check if all corners of a rectangle are within bounds.

        Args:
            corners: 4x2 array of corner positions

        Returns:
            True if all corners are within lane boundaries
        """
        for corner in corners:
            if not self.point_in_bounds(corner[0], corner[1]):
                return False
        return True

    def check_car(self, state: np.ndarray, model: BicycleModel) -> bool:
        """
        Check if car is collision-free.

        Args:
            state: [x, y, theta]
            model: Vehicle model for getting corners

        Returns:
            True if collision-free
        """
        corners = model.get_car_corners(state, CAR_LENGTH, CAR_WIDTH)
        if not self.corners_in_bounds(corners):
            return False
        if self.check_obstacle_collision(corners):
            return False
        return True

    def check_car_trailer(self, state: np.ndarray, model: CarTrailerModel) -> bool:
        """
        Check if car+trailer is collision-free.

        Args:
            state: [x1, y1, theta0, theta1]
            model: Vehicle model

        Returns:
            True if both car and trailer are collision-free
        """
        # Check tractor bounds
        car_corners = model.get_car_corners(state, CAR_LENGTH, CAR_WIDTH)
        if not self.corners_in_bounds(car_corners):
            return False
        # Check tractor obstacles
        if self.check_obstacle_collision(car_corners):
            return False

        # Check trailer bounds
        trailer_corners = model.get_trailer_corners(state, TRAILER_LENGTH, TRAILER_WIDTH)
        if not self.corners_in_bounds(trailer_corners):
            return False
        # Check trailer obstacles
        if self.check_obstacle_collision(trailer_corners):
            return False

        return True

    def check_trajectory(self, trajectory: List[np.ndarray],
                         model, is_trailer: bool = False) -> bool:
        """
        Check if an entire trajectory is collision-free.

        Args:
            trajectory: List of states
            model: Vehicle model
            is_trailer: True if using car+trailer model

        Returns:
            True if entire trajectory is collision-free
        """
        for state in trajectory:
            if is_trailer:
                if not self.check_car_trailer(state, model):
                    return False
            else:
                if not self.check_car(state, model):
                    return False
        return True

    def check_state(self, state: np.ndarray, model, is_trailer: bool = False) -> bool:
        """
        Generic state check.
        """
        if is_trailer:
            return self.check_car_trailer(state, model)
        else:
            return self.check_car(state, model)


def create_default_checker() -> CollisionChecker:
    """Create a collision checker with default lane boundaries."""
    return CollisionChecker()
