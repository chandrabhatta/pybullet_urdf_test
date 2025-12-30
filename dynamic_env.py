import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces


class SimpleCarEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        ego_urdf="car/urdf/simple_car.urdf",
        obstacle_urdf="car/urdf/obstacle_car.urdf",
        dt=0.05,
        gui=True,
    ):
        super().__init__()

        self.dt = dt
        self.ego_urdf = ego_urdf
        self.obstacle_urdf = obstacle_urdf

        # =====================
        # ACTION / OBSERVATION
        # =====================
        self.action_space = spaces.Box(
            low=np.array([-2.0, -0.6], dtype=np.float32),
            high=np.array([5.0, 0.6], dtype=np.float32),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=np.array([-1000.0, -1000.0, -np.pi], dtype=np.float32),
            high=np.array([1000.0, 1000.0, np.pi], dtype=np.float32),
        )

        # =====================
        # PYBULLET SETUP
        # =====================
        self.physics_id = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.dt)
        p.loadURDF("plane.urdf")

        # =====================
        # LOAD EGO CAR
        # =====================
        self.car_id = p.loadURDF(
            self.ego_urdf,
            basePosition=[0, 0, 0.2],
            useFixedBase=False,
        )

        self.steer_joints = []
        for ji in range(p.getNumJoints(self.car_id)):
            name = p.getJointInfo(self.car_id, ji)[1].decode("utf-8")
            if "steer" in name:
                self.steer_joints.append(ji)

        # =====================
        # ROAD / LANES
        # =====================
        self.lane_width = 3.5
        self.num_lanes = 2
        self.road_length = 200.0

        self.lane_centers = [
            -self.lane_width / 2,
            +self.lane_width / 2,
        ]

        # =====================
        # OBSTACLES
        # =====================
        self.obstacle_ids = []
        self.obstacle_states = []

        self._create_road_and_lanes()
        self._create_obstacles()

        self.reset()

    # ==========================================================
    # ROAD + LANE MARKINGS
    # ==========================================================
    def _create_road_and_lanes(self):
        road_width = self.num_lanes * self.lane_width

        road_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[self.road_length / 2, road_width / 2, 0.01],
            rgbaColor=[0.15, 0.15, 0.15, 1],
        )

        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=road_visual,
            basePosition=[self.road_length / 2, 0, 0.01],
        )

        for i in range(self.num_lanes + 1):
            y = -road_width / 2 + i * self.lane_width
            color = [1, 1, 1, 1] if i in [0, self.num_lanes] else [1, 1, 0, 1]

            marking = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[self.road_length / 2, 0.05, 0.02],
                rgbaColor=color,
            )

            p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=marking,
                basePosition=[self.road_length / 2, y, 0.02],
            )

    # ==========================================================
    # SPAWN MOVING OBSTACLE CARS
    # ==========================================================
    def _create_obstacles(self):
        self._spawn_obstacle_car(10.0, self.lane_centers[1], speed=0.6)
        self._spawn_obstacle_car(25.0, self.lane_centers[0], speed=0.5)
        self._spawn_obstacle_car(35.0, self.lane_centers[1], speed=0.4)
        self._spawn_obstacle_car(45.0, self.lane_centers[0], speed=0.6)

    def _spawn_obstacle_car(self, x, y, speed=0.5, yaw=0.0):
        orn = p.getQuaternionFromEuler([0, 0, yaw])

        obs_id = p.loadURDF(
            self.obstacle_urdf,
            basePosition=[x, y, 0.2],
            baseOrientation=orn,
            useFixedBase=True,
        )

        self.obstacle_ids.append(obs_id)
        self.obstacle_states.append(
            {
                "id": obs_id,
                "x": x,
                "y": y,
                "yaw": yaw,
                "v": speed,
            }
        )

    # ==========================================================
    # RESET
    # ==========================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.state = np.array(
            [0.0, self.lane_centers[1], 0.0],
            dtype=np.float32,
        )

        orn = p.getQuaternionFromEuler([0, 0, 0])
        p.resetBasePositionAndOrientation(
            self.car_id,
            [self.state[0], self.state[1], 0.2],
            orn,
        )

        for j in self.steer_joints:
            p.resetJointState(self.car_id, j, 0.0)

        return self.state.copy(), {}

    # ==========================================================
    # MOVING TRAFFIC
    # ==========================================================
    def _update_obstacles(self):
        for obs in self.obstacle_states:
            obs["x"] += obs["v"] * self.dt * np.cos(obs["yaw"])
            obs["y"] += obs["v"] * self.dt * np.sin(obs["yaw"])

            orn = p.getQuaternionFromEuler([0, 0, obs["yaw"]])
            p.resetBasePositionAndOrientation(
                obs["id"],
                [obs["x"], obs["y"], 0.2],
                orn,
            )

    # ==========================================================
    # STEP
    # ==========================================================
    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        v, steer = float(action[0]), float(action[1])

        x, y, yaw = self.state
        L = 2.5

        x += v * np.cos(yaw) * self.dt
        y += v * np.sin(yaw) * self.dt
        yaw += (v * np.tan(steer) / L) * self.dt

        self.state = np.array([x, y, yaw], dtype=np.float32)

        orn = p.getQuaternionFromEuler([0, 0, yaw])
        p.resetBasePositionAndOrientation(
            self.car_id,
            [x, y, 0.2],
            orn,
        )

        for j in self.steer_joints:
            p.setJointMotorControl2(
                self.car_id,
                j,
                p.POSITION_CONTROL,
                targetPosition=steer,
                force=50,
            )

        p.stepSimulation()

        # Move obstacle cars
        self._update_obstacles()

        # Collision check
        collision = False
        for c in p.getContactPoints(bodyA=self.car_id):
            if c[2] in self.obstacle_ids:
                collision = True
                break

        reward = -10.0 if collision else 0.0
        terminated = collision
        truncated = False

        return self.state.copy(), reward, terminated, truncated, {}

    def close(self):
        p.disconnect()
