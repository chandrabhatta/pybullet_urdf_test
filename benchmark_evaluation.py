"""
PDM Performance Evaluation Benchmark

Compares MPC and RRT planners on randomized lane-change scenarios
for a car+trailer system.

Usage:
    python benchmark_evaluation.py
    python benchmark_evaluation.py --trials 50
"""

import numpy as np
import time
import argparse
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

# Import existing modules
from RRT.rrt import RRT
from RRT.vehicle_model import CarTrailerModel
from RRT.collision import CollisionChecker
from RRT.evaluate_path import evaluate_path, compute_min_clearance
from RRT.metrics import pathLength


# ============================================================
# CONFIGURATION
# ============================================================
ROAD_LENGTH = 100.0
LANE_WIDTH = 3.5
LANE_CENTERS = [1.75, 5.25]  # Right lane, Left lane
VELOCITY = 5.0  # m/s for time-to-goal calculation


# ============================================================
# SCENARIO GENERATOR
# ============================================================
@dataclass
class Scenario:
    """A randomized test scenario."""
    scenario_id: int
    start: np.ndarray
    goal: np.ndarray
    obstacles: List[Tuple[float, float]]  # [(x, y), ...]
    num_obstacles: int


def generate_scenario(scenario_id: int, seed: int = 42) -> Scenario:
    """
    Generate a randomized scenario with obstacles, start, and goal.

    Args:
        scenario_id: Unique ID for this scenario
        seed: Base random seed for reproducibility

    Returns:
        Scenario with random obstacles and start/goal in opposite lanes
    """
    rng = np.random.default_rng(seed + scenario_id)

    # Random number of obstacles (2-5)
    num_obs = rng.integers(2, 6)

    # Place obstacles with minimum spacing
    obstacles = []
    used_x = []

    for _ in range(num_obs):
        # Try to find valid position
        for attempt in range(50):
            x = rng.uniform(15.0, ROAD_LENGTH - 20.0)
            if all(abs(x - ux) >= 10.0 for ux in used_x):
                used_x.append(x)
                y = rng.choice(LANE_CENTERS)
                obstacles.append((x, y))
                break

    # Random start lane, goal in opposite lane
    start_lane = rng.integers(0, 2)
    goal_lane = 1 - start_lane

    start = np.array([5.0, LANE_CENTERS[start_lane], 0.0, 0.0])
    goal = np.array([ROAD_LENGTH - 10.0, LANE_CENTERS[goal_lane], 0.0, 0.0])

    return Scenario(
        scenario_id=scenario_id,
        start=start,
        goal=goal,
        obstacles=obstacles,
        num_obstacles=len(obstacles)
    )


# ============================================================
# MPC PLANNER (adapted from new_test_car_env.py)
# ============================================================
def mpc_step(state: np.ndarray, goal: np.ndarray,
             obstacles: List[Tuple[float, float]], dt: float = 0.1) -> Tuple[float, float]:
    """
    Single MPC optimization step using LaValle car+trailer kinematics.

    Returns:
        (velocity, steering_angle)
    """
    x, y, theta, phi = state
    goal_y = goal[1]

    # Vehicle parameters
    L1 = 2.5  # car wheelbase
    L2 = 3.0  # trailer length
    horizon = 15  # Increased horizon for better planning

    # Determine current and goal lanes
    current_lane = min(LANE_CENTERS, key=lambda ly: abs(y - ly))
    other_lane = LANE_CENTERS[1] if current_lane == LANE_CENTERS[0] else LANE_CENTERS[0]

    # Check for obstacles ahead in current lane AND target lane
    LOOKAHEAD_DIST = 20.0  # Increased lookahead
    min_dist_current = np.inf
    min_dist_goal = np.inf

    for ox, oy in obstacles:
        if ox > x:  # Only look ahead
            dist = ox - x
            # Check current lane
            if abs(oy - current_lane) < 1.5:
                min_dist_current = min(min_dist_current, dist)
            # Check goal lane
            if abs(oy - goal_y) < 1.5:
                min_dist_goal = min(min_dist_goal, dist)

    # Decide target lane: go to goal lane unless obstacle is too close
    if min_dist_goal < 15.0 and min_dist_current > min_dist_goal:
        # Obstacle in goal lane is closer, stay in current lane or go to other
        target_lane = current_lane if min_dist_current > 10.0 else other_lane
    else:
        target_lane = goal_y

    # Adaptive speed based on closest obstacle in path
    min_dist = min(min_dist_current, min_dist_goal)
    if min_dist < 5.0:
        target_speed = 1.0
    elif min_dist < 10.0:
        target_speed = 1.5
    elif min_dist < 15.0:
        target_speed = 2.0
    else:
        target_speed = 2.5

    # MPC optimization over steering candidates (very conservative)
    steer_candidates = np.linspace(-0.20, 0.20, 11)
    best_cost = np.inf
    best_steer = 0.0

    # Car dimensions for corner checking
    car_half_length = 1.5
    car_half_width = 0.8

    for steer in steer_candidates:
        px, py, ptheta, pphi = x, y, theta, phi
        cost = 0.0

        for _ in range(horizon):
            # LaValle car+trailer kinematics
            px += target_speed * np.cos(ptheta) * dt
            py += target_speed * np.sin(ptheta) * dt
            ptheta += (target_speed / L1) * np.tan(steer) * dt
            pphi += (-(target_speed / L2) * np.sin(pphi) +
                     (target_speed / L1) * np.tan(steer)) * dt

            # Cost function
            cost += 12.0 * (py - target_lane) ** 2    # lane tracking
            cost += 15.0 * (ptheta ** 2)              # heading stability (increased more)
            cost += 12.0 * (pphi ** 2)                # trailer stability
            cost += 3.0 * (steer ** 2)                # steering smoothness

            # Estimate front corner positions (approximate)
            front_y_left = py + car_half_length * np.sin(ptheta) + car_half_width * np.cos(ptheta)
            front_y_right = py + car_half_length * np.sin(ptheta) - car_half_width * np.cos(ptheta)

            # Road bounds penalty on corners (much stricter)
            if front_y_left > 6.5 or front_y_right < 0.5:
                cost += 2000.0
            if py < 1.2 or py > 5.8:  # Center bounds
                cost += 500.0

            # Extreme heading penalty
            if abs(ptheta) > 0.3:
                cost += 500.0 * (abs(ptheta) - 0.3)

            # Obstacle avoidance cost
            for ox, oy in obstacles:
                dist_to_obs = np.sqrt((px - ox)**2 + (py - oy)**2)
                if dist_to_obs < 6.0:
                    cost += 150.0 * (6.0 - dist_to_obs)

        if cost < best_cost:
            best_cost = cost
            best_steer = steer

    return target_speed, best_steer


def run_mpc(start: np.ndarray, goal: np.ndarray,
            obstacles: List[Tuple[float, float]],
            collision_checker: CollisionChecker,
            model: CarTrailerModel,
            max_steps: int = 600,
            timeout: float = 30.0,
            dt: float = 0.1) -> Dict[str, Any]:
    """
    Run MPC simulation from start towards goal.

    Returns:
        Dict with: path, success, time, iterations, path_length
    """
    start_time = time.time()

    state = start.copy()
    path = [state.copy()]

    goal_threshold = 3.0  # Distance to consider goal reached

    for step in range(max_steps):
        # Check timeout
        elapsed = time.time() - start_time
        if elapsed > timeout:
            break

        # Check if goal reached (position only)
        dist_to_goal = np.sqrt((state[0] - goal[0])**2 + (state[1] - goal[1])**2)
        if dist_to_goal < goal_threshold:
            positions = [[s[0], s[1]] for s in path]
            return {
                'path': path,
                'success': True,
                'time': elapsed,
                'iterations': step,
                'path_length': pathLength(positions),
            }

        # Get MPC action
        velocity, steer = mpc_step(state, goal, obstacles, dt)

        # Simulate one step (LaValle kinematics)
        L1 = 2.5
        L2 = 3.0
        x, y, theta, phi = state

        new_x = x + velocity * np.cos(theta) * dt
        new_y = y + velocity * np.sin(theta) * dt
        new_theta = theta + (velocity / L1) * np.tan(steer) * dt
        new_phi = phi + (-(velocity / L2) * np.sin(phi) +
                         (velocity / L1) * np.tan(steer)) * dt

        new_state = np.array([new_x, new_y, new_theta, new_phi])

        # Collision check
        if not collision_checker.check_car_trailer(new_state, model):
            # Collision - fail
            elapsed = time.time() - start_time
            return {
                'path': path,
                'success': False,
                'time': elapsed,
                'iterations': step,
                'path_length': 0.0,
            }

        state = new_state
        path.append(state.copy())

    # Timeout or max steps reached
    elapsed = time.time() - start_time
    return {
        'path': path,
        'success': False,
        'time': elapsed,
        'iterations': len(path),
        'path_length': 0.0,
    }


# ============================================================
# RRT PLANNER WRAPPER
# ============================================================
def run_rrt(start: np.ndarray, goal: np.ndarray,
            obstacles: List[Tuple[float, float]],
            collision_checker: CollisionChecker,
            model: CarTrailerModel,
            max_iters: int = 2000) -> Dict[str, Any]:
    """
    Run RRT planning using existing RRT class.

    Returns:
        Dict with: path, success, time, iterations, nodes, path_length
    """
    # Clear and add obstacles
    collision_checker.clear_obstacles()
    for ox, oy in obstacles:
        collision_checker.add_obstacle(ox, oy, length=3.0, width=1.6, yaw=0.0)

    # Create RRT planner
    bounds = (0, ROAD_LENGTH, 0, 7)
    rrt = RRT(
        start=start,
        goal=goal,
        model=model,
        collision_checker=collision_checker,
        bounds=bounds
    )

    # Run planning
    result = rrt.plan(max_iters=max_iters, verbose=False)

    return result


# ============================================================
# METRICS COLLECTION
# ============================================================
@dataclass
class TrialResult:
    """Results from a single trial."""
    scenario_id: int
    planner: str
    success: bool
    compute_time: float
    path_length: float
    curvature_change: float
    risk_factor: float
    min_clearance: float
    time_to_goal: float
    iterations: int
    nodes: int  # RRT only


def compute_trial_metrics(scenario: Scenario, planner_name: str,
                          result: Dict[str, Any]) -> TrialResult:
    """Compute all metrics for a single trial."""

    if not result['success'] or result['path'] is None:
        return TrialResult(
            scenario_id=scenario.scenario_id,
            planner=planner_name,
            success=False,
            compute_time=result.get('time', 0.0),
            path_length=0.0,
            curvature_change=0.0,
            risk_factor=0.0,
            min_clearance=0.0,
            time_to_goal=0.0,
            iterations=result.get('iterations', 0),
            nodes=result.get('nodes', 0)
        )

    path = result['path']

    # Use existing evaluate_path function
    eval_result = evaluate_path(path, scenario.obstacles)

    # Compute min clearance
    min_clear = compute_min_clearance(path, scenario.obstacles)

    # Time to goal (assuming constant velocity)
    path_len = result.get('path_length', eval_result.get('path_length', 0.0))
    time_to_goal = path_len / VELOCITY if path_len > 0 else 0.0

    return TrialResult(
        scenario_id=scenario.scenario_id,
        planner=planner_name,
        success=True,
        compute_time=result['time'],
        path_length=path_len,
        curvature_change=eval_result.get('curvature_change', 0.0),
        risk_factor=eval_result.get('risk_factor', 0.0),
        min_clearance=min_clear,
        time_to_goal=time_to_goal,
        iterations=result.get('iterations', 0),
        nodes=result.get('nodes', 0)
    )


# ============================================================
# BENCHMARK RUNNER
# ============================================================
def run_benchmark(num_trials: int = 20, seed: int = 42,
                  run_mpc_flag: bool = True, run_rrt_flag: bool = True):
    """
    Run the full benchmark comparing MPC and RRT.

    Args:
        num_trials: Number of scenarios to test
        seed: Random seed for reproducibility
        run_mpc_flag: Whether to run MPC
        run_rrt_flag: Whether to run RRT
    """
    print("=" * 60)
    print("  PDM PERFORMANCE EVALUATION BENCHMARK")
    print("=" * 60)
    print(f"  Trials: {num_trials}")
    print(f"  Seed: {seed}")
    print(f"  Planners: {'MPC ' if run_mpc_flag else ''}{'RRT' if run_rrt_flag else ''}")
    print("=" * 60)

    # Initialize model
    model = CarTrailerModel(L0=2.5, L1=3.0, max_steer=0.5)

    # Collect results
    mpc_results: List[TrialResult] = []
    rrt_results: List[TrialResult] = []

    for i in range(num_trials):
        # Generate scenario
        scenario = generate_scenario(i, seed)

        print(f"\n--- Scenario {i+1}/{num_trials} ---")
        print(f"  Obstacles: {scenario.num_obstacles}")
        print(f"  Start: ({scenario.start[0]:.1f}, {scenario.start[1]:.1f})")
        print(f"  Goal:  ({scenario.goal[0]:.1f}, {scenario.goal[1]:.1f})")

        # Create collision checker for this scenario
        collision_checker = CollisionChecker()
        for ox, oy in scenario.obstacles:
            collision_checker.add_obstacle(ox, oy, length=3.0, width=1.6, yaw=0.0)

        # Run MPC
        if run_mpc_flag:
            print("  Running MPC...", end=" ", flush=True)
            mpc_result = run_mpc(
                scenario.start, scenario.goal, scenario.obstacles,
                collision_checker, model
            )
            mpc_metrics = compute_trial_metrics(scenario, "MPC", mpc_result)
            mpc_results.append(mpc_metrics)
            status = "SUCCESS" if mpc_metrics.success else "FAILED"
            print(f"{status} ({mpc_metrics.compute_time:.2f}s)")

        # Run RRT (needs fresh collision checker)
        if run_rrt_flag:
            print("  Running RRT...", end=" ", flush=True)
            rrt_collision_checker = CollisionChecker()
            rrt_result = run_rrt(
                scenario.start, scenario.goal, scenario.obstacles,
                rrt_collision_checker, model
            )
            rrt_metrics = compute_trial_metrics(scenario, "RRT", rrt_result)
            rrt_results.append(rrt_metrics)
            status = "SUCCESS" if rrt_metrics.success else "FAILED"
            print(f"{status} ({rrt_metrics.compute_time:.2f}s)")

    # Print summary
    print_summary(mpc_results, rrt_results)

    return mpc_results, rrt_results


def print_summary(mpc_results: List[TrialResult], rrt_results: List[TrialResult]):
    """Print formatted summary statistics."""

    def get_stats(results: List[TrialResult]) -> Dict[str, Any]:
        if not results:
            return {}

        successes = [r for r in results if r.success]
        success_rate = len(successes) / len(results) * 100

        stats = {
            'total': len(results),
            'successes': len(successes),
            'success_rate': success_rate,
        }

        if successes:
            stats['time_mean'] = np.mean([r.compute_time for r in successes])
            stats['time_std'] = np.std([r.compute_time for r in successes])
            stats['length_mean'] = np.mean([r.path_length for r in successes])
            stats['length_std'] = np.std([r.path_length for r in successes])
            stats['curvature_mean'] = np.mean([r.curvature_change for r in successes])
            stats['curvature_std'] = np.std([r.curvature_change for r in successes])
            stats['risk_mean'] = np.mean([r.risk_factor for r in successes])
            stats['risk_std'] = np.std([r.risk_factor for r in successes])
            stats['clearance_mean'] = np.mean([r.min_clearance for r in successes])
            stats['clearance_std'] = np.std([r.min_clearance for r in successes])
            stats['ttg_mean'] = np.mean([r.time_to_goal for r in successes])
            stats['ttg_std'] = np.std([r.time_to_goal for r in successes])
            stats['iter_mean'] = np.mean([r.iterations for r in successes])
            stats['iter_std'] = np.std([r.iterations for r in successes])

        return stats

    mpc_stats = get_stats(mpc_results)
    rrt_stats = get_stats(rrt_results)

    print("\n")
    print("=" * 70)
    print("  BENCHMARK RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<25}{'MPC':>20}{'RRT':>20}")
    print("-" * 70)

    def fmt_stat(stats, key_mean, key_std, fmt=".2f"):
        if not stats or key_mean not in stats:
            return "N/A"
        mean = stats[key_mean]
        std = stats.get(key_std, 0)
        return f"{mean:{fmt}} +/- {std:{fmt}}"

    # Success rate
    mpc_sr = f"{mpc_stats.get('success_rate', 0):.1f}%" if mpc_stats else "N/A"
    rrt_sr = f"{rrt_stats.get('success_rate', 0):.1f}%" if rrt_stats else "N/A"
    print(f"{'Success Rate':<25}{mpc_sr:>20}{rrt_sr:>20}")

    # Compute time
    print(f"{'Compute Time (s)':<25}{fmt_stat(mpc_stats, 'time_mean', 'time_std'):>20}"
          f"{fmt_stat(rrt_stats, 'time_mean', 'time_std'):>20}")

    # Path length
    print(f"{'Path Length (m)':<25}{fmt_stat(mpc_stats, 'length_mean', 'length_std'):>20}"
          f"{fmt_stat(rrt_stats, 'length_mean', 'length_std'):>20}")

    # Curvature change
    print(f"{'Curvature Change':<25}{fmt_stat(mpc_stats, 'curvature_mean', 'curvature_std', '.4f'):>20}"
          f"{fmt_stat(rrt_stats, 'curvature_mean', 'curvature_std', '.4f'):>20}")

    # Risk factor
    print(f"{'Risk Factor':<25}{fmt_stat(mpc_stats, 'risk_mean', 'risk_std', '.4f'):>20}"
          f"{fmt_stat(rrt_stats, 'risk_mean', 'risk_std', '.4f'):>20}")

    # Min clearance
    print(f"{'Min Clearance (m)':<25}{fmt_stat(mpc_stats, 'clearance_mean', 'clearance_std'):>20}"
          f"{fmt_stat(rrt_stats, 'clearance_mean', 'clearance_std'):>20}")

    # Time to goal
    print(f"{'Time to Goal (s)':<25}{fmt_stat(mpc_stats, 'ttg_mean', 'ttg_std'):>20}"
          f"{fmt_stat(rrt_stats, 'ttg_mean', 'ttg_std'):>20}")

    # Iterations
    print(f"{'Iterations':<25}{fmt_stat(mpc_stats, 'iter_mean', 'iter_std', '.0f'):>20}"
          f"{fmt_stat(rrt_stats, 'iter_mean', 'iter_std', '.0f'):>20}")

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDM Performance Benchmark")
    parser.add_argument('--trials', type=int, default=20,
                        help='Number of trials (default: 20)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--mpc-only', action='store_true',
                        help='Only run MPC benchmark')
    parser.add_argument('--rrt-only', action='store_true',
                        help='Only run RRT benchmark')

    args = parser.parse_args()

    run_mpc_flag = not args.rrt_only
    run_rrt_flag = not args.mpc_only

    run_benchmark(
        num_trials=args.trials,
        seed=args.seed,
        run_mpc_flag=run_mpc_flag,
        run_rrt_flag=run_rrt_flag
    )
