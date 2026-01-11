# PDM Project - Car+Trailer Motion Planning

Motion planning for a car+trailer system using **RRT** and **MPC** planners.

## Setup

### Requirements
```bash
pip install numpy pybullet
```

### Optional (for visualization/video export)
```bash
pip install imageio imageio-ffmpeg
```

## Running the Benchmark

The benchmark compares RRT and MPC planners on randomized lane-change scenarios.

### Basic Usage
```bash
cd pybullet_urdf_test
python benchmark_evaluation.py
```

### Options
```bash
# Run with custom number of trials (default: 20)
python benchmark_evaluation.py --trials 50

# Run only MPC benchmark
python benchmark_evaluation.py --mpc-only

# Run only RRT benchmark
python benchmark_evaluation.py --rrt-only

# Set random seed for reproducibility
python benchmark_evaluation.py --seed 123

# Combine options
python benchmark_evaluation.py --trials 100 --seed 42
```

### Expected Output
```
============================================================
  PDM PERFORMANCE EVALUATION BENCHMARK
============================================================
  Trials: 20
  Seed: 42
  Planners: MPC RRT
============================================================

--- Scenario 1/20 ---
  Obstacles: 2
  Start: (5.0, 5.2)
  Goal:  (90.0, 1.8)
  Running MPC... SUCCESS (1.23s)
  Running RRT... SUCCESS (5.67s)

... (more scenarios) ...

======================================================================
  BENCHMARK RESULTS SUMMARY
======================================================================
Metric                                    MPC                 RRT
----------------------------------------------------------------------
Success Rate                            10.0%               65.0%
Compute Time (s)                1.79 +/- 0.63      15.32 +/- 9.48
Path Length (m)                82.58 +/- 0.03      86.28 +/- 0.55
Curvature Change            0.0064 +/- 0.0010   0.0692 +/- 0.0185
Risk Factor                 0.1441 +/- 0.0258   0.1585 +/- 0.0354
Min Clearance (m)               3.84 +/- 0.06       2.15 +/- 0.36
Time to Goal (s)               16.52 +/- 0.01      17.26 +/- 0.11
Iterations                         421 +/- 46         785 +/- 424
======================================================================
```

## Metrics Explained

| Metric | Description |
|--------|-------------|
| **Success Rate** | % of trials where planner found collision-free path to goal |
| **Compute Time** | Time to compute the plan (seconds) |
| **Path Length** | Total distance traveled (meters) |
| **Curvature Change** | Path smoothness - lower is smoother |
| **Risk Factor** | Average inverse distance to obstacles - lower is safer |
| **Min Clearance** | Closest distance to any obstacle (meters) |
| **Time to Goal** | Simulated time to reach goal (path_length / velocity) |
| **Iterations** | MPC: simulation steps, RRT: tree expansions |

## Running Individual Tests

### RRT Lane Change Demo (with visualization)
```bash
python test_rrt_lane_change.py
```
This opens a PyBullet GUI showing the car+trailer following the RRT path.

### MPC Test
```bash
python new_test_car_env.py
```

## Project Structure

```
pybullet_urdf_test/
├── benchmark_evaluation.py    # Main benchmark script
├── RRT/
│   ├── rrt.py                 # RRT planner implementation
│   ├── vehicle_model.py       # Car+trailer kinematics
│   ├── collision.py           # Collision checking (SAT-based)
│   ├── metrics.py             # Path length, curvature metrics
│   ├── risk.py                # Risk factor calculation
│   └── evaluate_path.py       # Comprehensive path evaluation
├── car_env.py                 # PyBullet simulation environment
├── test_rrt_lane_change.py    # RRT demo with visualization
└── new_test_car_env.py        # MPC test script
```

## Notes on MPC Implementation

The current MPC is a **simplified version** using discrete steering search:
- Searches over 11 steering angles [-0.2, 0.2]
- Uses soft penalties for constraints (not hard constraints)
- Does not use an optimization solver

For a **proper MPC**, consider using:
- **CasADi + IPOPT**: Easier setup, good for prototyping
- **acados + HPIPM**: Faster runtime, better for real-time

See `benchmark_evaluation.py` lines 97-200 for the current MPC implementation.

## Adjusting Parameters

### Scenario Generation (in `benchmark_evaluation.py`)
- `ROAD_LENGTH = 100.0` - Road length in meters
- `LANE_CENTERS = [1.75, 5.25]` - Lane center positions
- Obstacles: 2-5 random obstacles per scenario

### MPC Parameters
- `horizon = 15` - Prediction horizon steps
- `steer_candidates = np.linspace(-0.20, 0.20, 11)` - Steering search range
- `max_steps = 600` - Maximum simulation steps
- `goal_threshold = 3.0` - Distance to consider goal reached

### RRT Parameters (in `RRT/rrt.py`)
- `max_iters = 2000` - Maximum tree expansions
- `GOAL_THRESHOLD = 1.0` - Goal reach distance
- `STEERING_ANGLES = [-0.5, -0.25, 0, 0.25, 0.5]` - Control samples
