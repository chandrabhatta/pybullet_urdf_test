This repository contains a Model Predictive Control (MPC) framework for a car + trailer system simulated in PyBullet, covering:

Highway driving

Obstacle avoidance

Lane changes

Parking with trailer alignment

---

Repository Structure
.
├── car/                     # Car models and MPC-related logic
├── mpc/                     # MPC controller implementations
├── simulation/              # Simulation utilities and helpers
├── urdf/                    # URDF files for car and trailer
├── modified_env_car/        # Modified environment variants
│
├── car_env.py               # Base car environment
├── car_env_with_car.py      # Environment with explicit car model
├── dynamic_env.py           # Main dynamic environment (cleaned, recommended)
│
├── test_car_env.py          # MPC testing on highway scenarios (including parking)
├── test_frenet_mpc.py       # Frenet-frame MPC experiments
├── test_car_park/           # Parking-specific experiments
├── test_env.py              # Environment sanity tests
├── test_state_reader.py     # State extraction and debugging
│
├── inspect_urdf_joints.py   # URDF joint inspection utility
├── run_urdf.py              # Standalone URDF runner
│
├── robotics_project.tar.gz  # Archived project snapshot
│
├── pyproject.toml           # Project configuration (Poetry)
├── poetry.lock              # Dependency lock file
├── .gitignore
└── README.md
