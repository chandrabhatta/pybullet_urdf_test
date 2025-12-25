import os
from simulation.pybullet_env import PyBulletEnv

if __name__ == "__main__":

    # Absolute path to project root
    project_root = os.path.dirname(os.path.abspath(__file__))

    # Absolute path to URDF
    urdf_path = os.path.join(project_root, "urdf", "simple_car.urdf")

    print("Using URDF:", urdf_path)

    env = PyBulletEnv(urdf_path, gui=True)

    try:
        while True:
            env.step()
    except KeyboardInterrupt:
        env.close()