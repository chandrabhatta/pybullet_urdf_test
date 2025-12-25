import time
import os

from simulation.pybullet_env import PyBulletEnv
from simulation.state_reader import StateReader


# Absolute path to URDF
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.join(ROOT_DIR, "urdf", "simple_car.urdf")

# Create simulation
env = PyBulletEnv(urdf_path=URDF_PATH, gui=True)

# Create state reader
reader = StateReader(robot_id=env.robot_id)

print("Starting state reader test...")
print("Format: [x, y, theta_car, theta_trailer] | [s, phi]")

while True:
    x = reader.get_state()
    u = reader.get_input()

    print(f"State: {x} | Input: {u}")

    env.step()
