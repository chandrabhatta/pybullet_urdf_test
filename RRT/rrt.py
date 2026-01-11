"""
Kinodynamic RRT planner for car and car+trailer lane change.
"""

import numpy as np
import time
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

try:
    from .vehicle_model import BicycleModel, CarTrailerModel
    from .collision import CollisionChecker, create_default_checker
    from .metrics import pathLength
except ImportError:
    from vehicle_model import BicycleModel, CarTrailerModel
    from collision import CollisionChecker, create_default_checker
    from metrics import pathLength


# RRT Parameters
STEERING_ANGLES = [-0.5, -0.25, 0, 0.25, 0.5]  # Discrete steering samples
VELOCITY = 5.0  # Fixed forward velocity (m/s)
DT = 0.1  # Integration timestep (s)
SIM_TIME = 0.5  # Extension duration (s)
GOAL_SAMPLE_RATE = 0.1  # Probability of sampling goal
GOAL_THRESHOLD = 1.0  # Distance to consider goal reached


@dataclass
class Node:
    """RRT tree node."""
    state: np.ndarray
    parent: Optional['Node'] = None
    control: Optional[Tuple[float, float]] = None
    trajectory: Optional[List[np.ndarray]] = None

    def __hash__(self):
        return id(self)


class RRT:
    """
    Kinodynamic RRT planner.

    Uses discrete control sampling to extend the tree.
    Works with both car-only and car+trailer models.
    """

    def __init__(self,
                 start: np.ndarray,
                 goal: np.ndarray,
                 model: BicycleModel | CarTrailerModel,
                 collision_checker: Optional[CollisionChecker] = None,
                 bounds: Optional[Tuple[float, float, float, float]] = None):
        """
        Initialize RRT planner.

        Args:
            start: Initial state
            goal: Goal state
            model: Vehicle kinematic model
            collision_checker: Collision checker (uses default if None)
            bounds: Sampling bounds (x_min, x_max, y_min, y_max)
        """
        self.start = np.array(start, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.model = model
        self.collision_checker = collision_checker or create_default_checker()

        # Determine if using trailer model
        self.is_trailer = isinstance(model, CarTrailerModel)
        self.state_dim = model.state_dim()

        # Sampling bounds
        if bounds is None:
            self.bounds = (0, 100, 0, 7)  # Default: 100m road, 7m width
        else:
            self.bounds = bounds

        # Initialize tree with start node
        self.nodes: List[Node] = [Node(state=self.start.copy())]

        # Planning parameters
        self.steering_angles = STEERING_ANGLES
        self.velocity = VELOCITY
        self.dt = DT
        self.sim_time = SIM_TIME
        self.goal_sample_rate = GOAL_SAMPLE_RATE
        self.goal_threshold = GOAL_THRESHOLD

    def sample_random(self) -> np.ndarray:
        """
        Sample a random state in the configuration space.
        With some probability, sample the goal instead.
        """
        if np.random.random() < self.goal_sample_rate:
            return self.goal.copy()

        x_min, x_max, y_min, y_max = self.bounds

        if self.is_trailer:
            # Sample [x1, y1, theta0, theta1]
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            theta0 = np.random.uniform(-np.pi, np.pi)
            theta1 = np.random.uniform(-np.pi, np.pi)
            return np.array([x, y, theta0, theta1])
        else:
            # Sample [x, y, theta]
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            theta = np.random.uniform(-np.pi, np.pi)
            return np.array([x, y, theta])

    def distance(self, state1: np.ndarray, state2: np.ndarray) -> float:
        """
        Compute distance between two states.
        Uses weighted Euclidean distance.
        """
        # Position weight vs angle weight
        pos_weight = 1.0
        angle_weight = 0.5

        if self.is_trailer:
            # [x1, y1, theta0, theta1]
            dx = state1[0] - state2[0]
            dy = state1[1] - state2[1]
            dtheta0 = self._angle_diff(state1[2], state2[2])
            dtheta1 = self._angle_diff(state1[3], state2[3])
            return np.sqrt(pos_weight * (dx**2 + dy**2) +
                          angle_weight * (dtheta0**2 + dtheta1**2))
        else:
            # [x, y, theta]
            dx = state1[0] - state2[0]
            dy = state1[1] - state2[1]
            dtheta = self._angle_diff(state1[2], state2[2])
            return np.sqrt(pos_weight * (dx**2 + dy**2) +
                          angle_weight * dtheta**2)

    def _angle_diff(self, a1: float, a2: float) -> float:
        """Compute shortest angle difference."""
        diff = a1 - a2
        return np.arctan2(np.sin(diff), np.cos(diff))

    def nearest(self, target: np.ndarray) -> Node:
        """Find the nearest node in the tree to target state."""
        min_dist = float('inf')
        nearest_node = self.nodes[0]

        for node in self.nodes:
            d = self.distance(node.state, target)
            if d < min_dist:
                min_dist = d
                nearest_node = node

        return nearest_node

    def steer(self, from_node: Node, target: np.ndarray) -> Optional[Tuple[Node, float]]:
        """
        Try to extend from from_node towards target using discrete controls.

        Returns the best new node and its distance to target, or None if all
        extensions result in collision.
        """
        best_node = None
        best_dist = float('inf')

        for delta in self.steering_angles:
            control = (self.velocity, delta)

            # Simulate forward
            new_state, trajectory = self.model.simulate(
                from_node.state, control, self.dt, self.sim_time
            )

            # Check collision along trajectory
            if not self.collision_checker.check_trajectory(
                trajectory, self.model, self.is_trailer
            ):
                continue

            # Compute distance to target
            dist = self.distance(new_state, target)

            if dist < best_dist:
                best_dist = dist
                best_node = Node(
                    state=new_state,
                    parent=from_node,
                    control=control,
                    trajectory=trajectory
                )

        if best_node is None:
            return None

        return best_node, best_dist

    def extend(self, target: np.ndarray) -> Optional[Node]:
        """
        Extend the tree towards target.

        Returns the new node if extension successful, None otherwise.
        """
        nearest_node = self.nearest(target)
        result = self.steer(nearest_node, target)

        if result is None:
            return None

        new_node, _ = result
        self.nodes.append(new_node)

        return new_node

    def is_goal_reached(self, state: np.ndarray) -> bool:
        """Check if state is close enough to goal."""
        # Only check position for goal (not heading)
        pos_dist = np.sqrt((state[0] - self.goal[0])**2 +
                          (state[1] - self.goal[1])**2)
        return pos_dist < self.goal_threshold

    def extract_path(self, goal_node: Node) -> List[np.ndarray]:
        """
        Extract the path from start to goal_node.

        Returns list of states along the path.
        """
        path = []
        node = goal_node

        while node is not None:
            if node.trajectory is not None:
                # Add trajectory in reverse (will be reversed at end)
                path.extend(reversed(node.trajectory))
            else:
                path.append(node.state)
            node = node.parent

        path.reverse()
        return path

    def plan(self, max_iters: int = 1000, verbose: bool = True) -> Dict[str, Any]:
        """
        Run RRT planning.

        Args:
            max_iters: Maximum number of iterations
            verbose: Print progress

        Returns:
            Dict with keys: 'path', 'success', 'time', 'iterations', 'nodes', 'path_length'
        """
        start_time = time.time()

        for i in range(max_iters):
            # Sample random state
            target = self.sample_random()

            # Extend tree towards target
            new_node = self.extend(target)

            if new_node is not None:
                # Check if we reached the goal
                if self.is_goal_reached(new_node.state):
                    elapsed = time.time() - start_time
                    path = self.extract_path(new_node)

                    # Compute path length using metrics
                    positions = [[s[0], s[1]] for s in path]
                    path_len = pathLength(positions)

                    result = {
                        'path': path,
                        'success': True,
                        'time': elapsed,
                        'iterations': i + 1,
                        'nodes': len(self.nodes),
                        'path_length': path_len,
                    }

                    if verbose:
                        print(f"Goal reached at iteration {i}!")
                        print(f"Time: {elapsed:.3f}s, Tree size: {len(self.nodes)} nodes")
                        print(f"Path length: {path_len:.2f}m")

                    return result

            if verbose and (i + 1) % 100 == 0:
                print(f"Iteration {i+1}/{max_iters}, tree size: {len(self.nodes)}")

        elapsed = time.time() - start_time
        if verbose:
            print(f"Failed to find path after {max_iters} iterations")
            print(f"Time: {elapsed:.3f}s, Tree size: {len(self.nodes)} nodes")

        return {
            'path': None,
            'success': False,
            'time': elapsed,
            'iterations': max_iters,
            'nodes': len(self.nodes),
            'path_length': 0.0,
        }

    def run_benchmark(self, num_trials: int = 10, max_iters: int = 3000) -> Dict[str, Any]:
        """
        Run multiple trials and compute statistics.

        Args:
            num_trials: Number of trials to run
            max_iters: Maximum iterations per trial

        Returns:
            Dict with success_rate, time_mean, time_std, length_mean, length_std
        """
        results = []
        for trial in range(num_trials):
            # Reset tree for each trial
            self.nodes = [Node(state=self.start.copy())]
            result = self.plan(max_iters=max_iters, verbose=False)
            results.append(result)
            status = 'Success' if result['success'] else 'Failed'
            print(f"Trial {trial+1}/{num_trials}: {status}")

        successes = [r for r in results if r['success']]
        success_rate = len(successes) / num_trials * 100

        if successes:
            times = [r['time'] for r in successes]
            lengths = [r['path_length'] for r in successes]

            return {
                'success_rate': success_rate,
                'time_mean': np.mean(times),
                'time_std': np.std(times),
                'length_mean': np.mean(lengths),
                'length_std': np.std(lengths),
                'num_trials': num_trials,
                'num_successes': len(successes),
            }

        return {
            'success_rate': 0.0,
            'num_trials': num_trials,
            'num_successes': 0,
        }

    def get_tree_edges(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Get all edges in the tree for visualization.

        Returns list of (parent_state, child_state) tuples.
        """
        edges = []
        for node in self.nodes:
            if node.parent is not None:
                edges.append((node.parent.state, node.state))
        return edges


def create_lane_change_rrt(start_lane: int = 0,
                           goal_lane: int = 1,
                           start_x: float = 10.0,
                           goal_x: float = 50.0,
                           use_trailer: bool = False) -> RRT:
    """
    Create an RRT planner for a lane change scenario.

    Args:
        start_lane: Starting lane (0 = right, 1 = left)
        goal_lane: Target lane
        start_x: Starting X position
        goal_x: Goal X position
        use_trailer: Whether to use car+trailer model

    Returns:
        Configured RRT planner
    """
    # Lane centers
    lane_centers = [1.75, 5.25]

    start_y = lane_centers[start_lane]
    goal_y = lane_centers[goal_lane]

    if use_trailer:
        model = CarTrailerModel(L0=2.5, L1=3.0, max_steer=0.5)
        start = np.array([start_x, start_y, 0.0, 0.0])  # Both headings = 0
        goal = np.array([goal_x, goal_y, 0.0, 0.0])
    else:
        model = BicycleModel(wheelbase=2.5, max_steer=0.5)
        start = np.array([start_x, start_y, 0.0])
        goal = np.array([goal_x, goal_y, 0.0])

    bounds = (0, 100, 0, 7)  # Road bounds

    return RRT(start, goal, model, bounds=bounds)
