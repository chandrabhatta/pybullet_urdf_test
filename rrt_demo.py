"""
RRT Demo: Visualize kinodynamic RRT for lane-change planning.
Run this script to see the RRT tree growing and find a lane-change path.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection

# Add mpc to path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mpc.rrt import RRT, create_lane_change_rrt
from mpc.vehicle_model import BicycleModel, CarTrailerModel


def plot_vehicle(ax, state, model, color='blue', alpha=0.8, is_trailer=False):
    """Plot vehicle rectangle at given state."""
    if is_trailer:
        # Plot both tractor and trailer
        car_corners = model.get_car_corners(state, 3.0, 1.6)
        trailer_corners = model.get_trailer_corners(state, 2.0, 1.4)

        car_poly = patches.Polygon(car_corners, closed=True,
                                   facecolor=color, edgecolor='black',
                                   alpha=alpha, linewidth=1)
        trailer_poly = patches.Polygon(trailer_corners, closed=True,
                                       facecolor='green', edgecolor='black',
                                       alpha=alpha, linewidth=1)
        ax.add_patch(car_poly)
        ax.add_patch(trailer_poly)
    else:
        corners = model.get_car_corners(state, 3.0, 1.6)
        poly = patches.Polygon(corners, closed=True,
                              facecolor=color, edgecolor='black',
                              alpha=alpha, linewidth=1)
        ax.add_patch(poly)


def plot_road(ax, x_range=(0, 100)):
    """Plot the 2-lane Dutch highway."""
    x_min, x_max = x_range

    # Road surface (gray)
    road = patches.Rectangle((x_min, 0), x_max - x_min, 7.0,
                             facecolor='#404040', edgecolor='none')
    ax.add_patch(road)

    # Edge lines (white)
    ax.axhline(y=0, color='white', linewidth=2, linestyle='-')
    ax.axhline(y=7.0, color='white', linewidth=2, linestyle='-')

    # Center dashed line
    for x in np.arange(x_min, x_max, 12):
        ax.plot([x, x + 3], [3.5, 3.5], 'w-', linewidth=1.5)

    # Lane labels
    ax.text(x_min + 2, 1.75, 'Lane 0', color='white', fontsize=8,
            verticalalignment='center')
    ax.text(x_min + 2, 5.25, 'Lane 1', color='white', fontsize=8,
            verticalalignment='center')


def plot_tree(ax, rrt, color='cyan', alpha=0.3):
    """Plot all edges in the RRT tree."""
    edges = rrt.get_tree_edges()

    if not edges:
        return

    lines = []
    for parent_state, child_state in edges:
        lines.append([(parent_state[0], parent_state[1]),
                     (child_state[0], child_state[1])])

    lc = LineCollection(lines, colors=color, alpha=alpha, linewidths=0.5)
    ax.add_collection(lc)


def plot_path(ax, path, color='yellow', linewidth=2):
    """Plot the solution path."""
    if not path:
        return

    x = [s[0] for s in path]
    y = [s[1] for s in path]
    ax.plot(x, y, color=color, linewidth=linewidth, label='Path', zorder=10)


def run_demo(use_trailer=False, max_iters=1000, animate=False):
    """
    Run RRT demo.

    Args:
        use_trailer: Use car+trailer model instead of car-only
        max_iters: Maximum RRT iterations
        animate: Show tree growing in real-time (slower)
    """
    print("\n" + "="*60)
    print("  RRT Lane-Change Planning Demo")
    print("="*60)
    print(f"\nModel: {'Car + Trailer' if use_trailer else 'Car only'}")
    print(f"Max iterations: {max_iters}")
    print("\nPlanning lane change from Lane 0 to Lane 1...")
    print("-"*60)

    # Create RRT planner
    rrt = create_lane_change_rrt(
        start_lane=0,
        goal_lane=1,
        start_x=10.0,
        goal_x=50.0,
        use_trailer=use_trailer
    )

    # Set up figure
    fig, ax = plt.subplots(figsize=(14, 4))

    if animate:
        # Animated version - show tree growing
        plt.ion()

        for i in range(max_iters):
            target = rrt.sample_random()
            new_node = rrt.extend(target)

            if new_node is not None and rrt.is_goal_reached(new_node.state):
                print(f"\nGoal reached at iteration {i}!")
                path = rrt.extract_path(new_node)
                break

            if (i + 1) % 50 == 0:
                ax.clear()
                plot_road(ax, x_range=(0, 70))
                plot_tree(ax, rrt)

                # Plot start and goal
                ax.plot(rrt.start[0], rrt.start[1], 'go', markersize=10,
                       label='Start', zorder=20)
                ax.plot(rrt.goal[0], rrt.goal[1], 'r*', markersize=15,
                       label='Goal', zorder=20)

                ax.set_xlim(0, 70)
                ax.set_ylim(-1, 8)
                ax.set_aspect('equal')
                ax.set_title(f'RRT Iteration {i+1} - Tree size: {len(rrt.nodes)}')
                ax.legend(loc='upper right')

                plt.pause(0.01)
        else:
            path = None

        plt.ioff()
    else:
        # Non-animated - just run planning
        path = rrt.plan(max_iters=max_iters, verbose=True)

    # Final visualization
    ax.clear()
    plot_road(ax, x_range=(0, 70))
    plot_tree(ax, rrt, alpha=0.4)

    # Plot start and goal
    ax.plot(rrt.start[0], rrt.start[1], 'go', markersize=12,
           label='Start', zorder=20)
    ax.plot(rrt.goal[0], rrt.goal[1], 'r*', markersize=15,
           label='Goal', zorder=20)

    if path:
        plot_path(ax, path)

        # Plot vehicle at a few positions along path
        step = max(1, len(path) // 5)
        for i in range(0, len(path), step):
            alpha = 0.3 + 0.5 * (i / len(path))
            plot_vehicle(ax, path[i], rrt.model, color='blue',
                        alpha=alpha, is_trailer=use_trailer)

        # Final position
        plot_vehicle(ax, path[-1], rrt.model, color='blue',
                    alpha=1.0, is_trailer=use_trailer)

        print(f"\nPath found with {len(path)} waypoints")
    else:
        print("\nNo path found!")

    ax.set_xlim(0, 70)
    ax.set_ylim(-1, 8)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'RRT Lane Change - {"Car+Trailer" if use_trailer else "Car"} '
                f'(Tree: {len(rrt.nodes)} nodes)')
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

    return path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='RRT Lane-Change Demo')
    parser.add_argument('--trailer', action='store_true',
                       help='Use car+trailer model')
    parser.add_argument('--animate', action='store_true',
                       help='Show animated tree growth')
    parser.add_argument('--iters', type=int, default=1000,
                       help='Max RRT iterations')

    args = parser.parse_args()

    path = run_demo(
        use_trailer=args.trailer,
        max_iters=args.iters,
        animate=args.animate
    )
