import pybullet as p
import numpy as np


class StateReader:
    """
    Extracts kinematic states and inputs for a car–trailer system
    from a PyBullet simulation.
    """

    def __init__(self, robot_id,
                 steering_joint_name="steer_fl",
                 trailer_link_name="trailer_load"):
        """
        Parameters
        ----------
        robot_id : int
            PyBullet body unique ID.
        steering_joint_name : str
            Name of steering joint in URDF.
        trailer_link_name : str
            Name of trailer body/link for yaw extraction.
        """

        self.robot_id = robot_id

        # Map joint names to indices
        self.joint_name_to_index = {}
        for i in range(p.getNumJoints(robot_id)):
            info = p.getJointInfo(robot_id, i)
            self.joint_name_to_index[info[1].decode()] = i

        self.steering_joint = self.joint_name_to_index[steering_joint_name]

        # Find trailer link index
        self.trailer_link_index = None
        for i in range(p.getNumJoints(robot_id)):
            if p.getJointInfo(robot_id, i)[12].decode() == trailer_link_name:
                self.trailer_link_index = i
                break

        if self.trailer_link_index is None:
            raise ValueError("Trailer link not found in URDF")

    # --------------------------------------------------
    # Utility
    # --------------------------------------------------

    @staticmethod
    def _yaw_from_quaternion(q):
        """
        Extract yaw from quaternion.
        """
        _, _, yaw = p.getEulerFromQuaternion(q)
        return yaw

    # --------------------------------------------------
    # State extraction
    # --------------------------------------------------

    def get_state(self):
        """
        Returns the state vector:
            x = [x, y, theta_car, theta_trailer]^T
        """

        # Car base pose
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        x, y = pos[0], pos[1]
        theta_car = self._yaw_from_quaternion(orn)

        # Trailer pose
        trailer_state = p.getLinkState(self.robot_id,
                                       self.trailer_link_index,
                                       computeForwardKinematics=True)
        trailer_orn = trailer_state[1]
        theta_trailer = self._yaw_from_quaternion(trailer_orn)

        return np.array([x, y, theta_car, theta_trailer])

    # --------------------------------------------------
    # Input extraction
    # --------------------------------------------------

    def get_input(self):
        """
        Returns the input vector:
            u = [s, phi]^T
        where
            s   = longitudinal speed
            phi = steering angle
        """

        # Base velocity
        lin_vel, _ = p.getBaseVelocity(self.robot_id)

        # Project velocity onto car heading
        _, orn = p.getBasePositionAndOrientation(self.robot_id)
        theta = self._yaw_from_quaternion(orn)

        s = lin_vel[0] * np.cos(theta) + lin_vel[1] * np.sin(theta)

        # Steering angle
        phi = p.getJointState(self.robot_id, self.steering_joint)[0]

        return np.array([s, phi])
