# RRT Baseline for Lane-Change Planning

Kinodynamic RRT (Rapidly-exploring Random Trees) implementation for car and car+trailer lane-change maneuvers. This serves as a baseline for comparison with MPC approaches.

## Files

| File | Description |
|------|-------------|
| `vehicle_model.py` | Kinematic models (Bicycle + Car+Trailer) |
| `collision.py` | Lane boundary collision checking |
| `rrt.py` | Core RRT planner implementation |
| `rrt_demo.py` | Matplotlib visualization demo |

## Quick Start

### Run the Demo

```bash
# Navigate to RRT folder
cd RRT

# Car-only demo
python rrt_demo.py

# Car+trailer demo
python rrt_demo.py --trailer

# Animated tree growth
python rrt_demo.py --animate

# More iterations
python rrt_demo.py --iters 2000
```

### Use in Your Code

```python
from RRT import create_lane_change_rrt

# Create planner for lane change
rrt = create_lane_change_rrt(
    start_lane=0,      # Right lane
    goal_lane=1,       # Left lane
    start_x=10.0,      # Starting X position (m)
    goal_x=50.0,       # Goal X position (m)
    use_trailer=True   # Use car+trailer model
)

# Run planning
path = rrt.plan(max_iters=1000, verbose=True)

# Path is a list of states
# Car-only: [x, y, theta]
# Car+trailer: [x1, y1, theta0, theta1]
for state in path:
    print(state)
```

## Vehicle Models

### Car-Only (Bicycle Model)

```
State: [x, y, theta]
  - x, y: rear axle position (m)
  - theta: heading angle (rad)

Control: [v, delta]
  - v: forward velocity (m/s)
  - delta: steering angle (rad)

Kinematics:
  dx/dt = v * cos(theta)
  dy/dt = v * sin(theta)
  dtheta/dt = v * tan(delta) / L
```

### Car+Trailer

```
State: [x1, y1, theta0, theta1]
  - x1, y1: trailer rear axle position (m)
  - theta0: tractor heading (rad)
  - theta1: trailer heading (rad)

Control: [v, delta]
  - v: forward velocity (m/s)
  - delta: steering angle (rad)

Kinematics:
  beta = theta0 - theta1  (hitch angle)
  dtheta0/dt = v * tan(delta) / L0
  dtheta1/dt = v * sin(beta) / L1
  dx1/dt = v * cos(beta) * cos(theta1)
  dy1/dt = v * cos(beta) * sin(theta1)
```

## Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `L0` | 2.5 m | Tractor wheelbase |
| `L1` | 3.0 m | Trailer length (hitch to rear axle) |
| `max_steer` | 0.5 rad | Maximum steering angle (~29 deg) |
| `velocity` | 5.0 m/s | Fixed forward velocity |
| `dt` | 0.1 s | Integration timestep |
| `sim_time` | 0.5 s | Extension duration per step |
| `steering_angles` | [-0.5, -0.25, 0, 0.25, 0.5] | Discrete control samples |

## Lane Configuration

```
Y = 7.0   ═══════════════════ Outer edge
Y = 5.25    Lane 1 (Left/Fast)
Y = 3.5   - - - - - - - - - - Center line
Y = 1.75    Lane 0 (Right/Slow)
Y = 0     ═══════════════════ Inner edge

Road width: 7.0 m (2 lanes x 3.5 m)
```

## Algorithm Overview

1. **Sample** random state (or goal with 10% probability)
2. **Nearest** - find closest node in tree
3. **Steer** - try 5 discrete steering angles, simulate 0.5s forward
4. **Collision check** - verify trajectory stays in lane bounds
5. **Extend** - add best node to tree
6. **Goal check** - if within 1m of goal, extract path

## Dependencies

- `numpy`
- `matplotlib` (for visualization)

## Example Output

```
============================================================
  RRT Lane-Change Planning Demo
============================================================

Model: Car + Trailer
Max iterations: 1000

Planning lane change from Lane 0 to Lane 1...
------------------------------------------------------------
Goal reached at iteration 163!
Tree size: 254 nodes

Path found with 97 waypoints
```
