"""
Evaluate RRT path with full metrics.

This module provides functions to compute curvature change and risk factor
metrics on completed paths for comparison between RRT and MPC planners.
"""

import numpy as np
from typing import List, Dict, Any, Tuple

try:
    from .metrics import pathLength, curvatureChange
    from .risk import riskFactor
except ImportError:
    from metrics import pathLength, curvatureChange
    from risk import riskFactor


def evaluate_path(
    path: List[np.ndarray],
    obstacles: List[Tuple[float, float]],
    dt: float = 0.1,
    velocity: float = 5.0
) -> Dict[str, Any]:
    """
    Compute all metrics for a given path.

    Args:
        path: List of states [x, y, theta0, theta1] or [x, y, theta]
        obstacles: List of obstacle positions [(x, y), ...]
        dt: Time step between path points
        velocity: Constant velocity (m/s)

    Returns:
        Dict with path_length, curvature_change, risk_factor
    """
    if path is None or len(path) < 2:
        return {
            'path_length': 0.0,
            'curvature_change': 0.0,
            'risk_factor': 0.0,
            'valid': False,
        }

    # Extract positions for path length
    positions = [[s[0], s[1]] for s in path]

    # Path length
    length = pathLength(positions)

    # Compute velocities for curvature
    linear_vels = [velocity] * len(path)
    angular_vels = []
    for i in range(len(path) - 1):
        dtheta = path[i + 1][2] - path[i][2]  # theta0 change
        # Normalize angle difference
        dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
        omega = dtheta / dt
        angular_vels.append(omega)
    angular_vels.append(angular_vels[-1])  # Repeat last

    # Time steps
    time_steps = [i * dt for i in range(len(path))]

    # Curvature change
    try:
        cc = curvatureChange(linear_vels, angular_vels, time_steps, positions)
    except Exception:
        cc = 0.0

    # Risk factor - compute distances to obstacles
    obstacle_dists = []
    for state in path:
        pos = np.array([state[0], state[1]])
        if obstacles:
            min_dist = min(np.linalg.norm(pos - np.array(obs)) for obs in obstacles)
        else:
            min_dist = float('inf')
        obstacle_dists.append(max(min_dist, 0.01))  # Avoid division by zero

    try:
        risk = riskFactor(obstacle_dists)
    except Exception:
        risk = 0.0

    return {
        'path_length': length,
        'curvature_change': cc,
        'risk_factor': risk,
        'valid': True,
    }


def compute_min_clearance(
    path: List[np.ndarray],
    obstacles: List[Tuple[float, float]]
) -> float:
    """
    Compute minimum clearance (distance) to any obstacle along the path.

    Args:
        path: List of states
        obstacles: List of obstacle positions

    Returns:
        Minimum distance to any obstacle
    """
    if not path or not obstacles:
        return float('inf')

    min_clearance = float('inf')
    for state in path:
        pos = np.array([state[0], state[1]])
        for obs in obstacles:
            dist = np.linalg.norm(pos - np.array(obs))
            if dist < min_clearance:
                min_clearance = dist

    return min_clearance


def print_metrics(metrics: Dict[str, Any]) -> None:
    """Print metrics in a formatted way."""
    print("\n=== Path Metrics ===")
    print(f"  Path Length:      {metrics.get('path_length', 0):.2f} m")
    print(f"  Curvature Change: {metrics.get('curvature_change', 0):.4f}")
    print(f"  Risk Factor:      {metrics.get('risk_factor', 0):.4f}")
    print("====================\n")


if __name__ == "__main__":
    # Example usage
    from rrt import create_lane_change_rrt

    # Create and run RRT
    rrt = create_lane_change_rrt(use_trailer=True)
    result = rrt.plan(max_iters=3000)

    if result['success']:
        # Define obstacles (same as in test_rrt_lane_change.py)
        obstacles = [
            (15.0, 5.25),
            (30.0, 1.75),
            (45.0, 5.25),
            (60.0, 1.75),
        ]

        # Evaluate path
        metrics = evaluate_path(result['path'], obstacles)
        print_metrics(metrics)

        # Min clearance
        clearance = compute_min_clearance(result['path'], obstacles)
        print(f"Minimum clearance: {clearance:.2f} m")
    else:
        print("No path found!")
