import pybullet as p
import pybullet_data
import time
import os


class PyBulletEnv:
    def __init__(self, urdf_path, gui=True):
        """
        Initializes the PyBullet simulation environment.

        Parameters
        ----------
        urdf_path : str
            Absolute or relative path to the robot URDF.
        gui : bool
            Whether to launch PyBullet with GUI.
        """

        # Connect to PyBullet
        if gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        # Set search path for built-in assets
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Physics parameters
        p.setGravity(0, 0, -9.81)

        # Load ground plane
        p.loadURDF("plane.urdf")

        # Resolve URDF path
        self.urdf_path = os.path.abspath(urdf_path)

        # Load robot
        self.robot_id = p.loadURDF(
            self.urdf_path,
            basePosition=[0, 0, 0.1],
            useFixedBase=False
        )

        # Simulation timestep
        self.time_step = 1.0 / 240.0

    def step(self):
        """Advance the simulation by one step."""
        p.stepSimulation()
        time.sleep(self.time_step)

    def close(self):
        """Disconnect from PyBullet."""
        p.disconnect()
