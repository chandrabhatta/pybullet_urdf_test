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
    Checks vehicle rectangles against lane boundaries.
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
        return self.corners_in_bounds(corners)

    def check_car_trailer(self, state: np.ndarray, model: CarTrailerModel) -> bool:
        """
        Check if car+trailer is collision-free.

        Args:
            state: [x1, y1, theta0, theta1]
            model: Vehicle model

        Returns:
            True if both car and trailer are collision-free
        """
        # Check tractor
        car_corners = model.get_car_corners(state, CAR_LENGTH, CAR_WIDTH)
        if not self.corners_in_bounds(car_corners):
            return False

        # Check trailer
        trailer_corners = model.get_trailer_corners(state, TRAILER_LENGTH, TRAILER_WIDTH)
        if not self.corners_in_bounds(trailer_corners):
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
