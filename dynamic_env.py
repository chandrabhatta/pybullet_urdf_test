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

        # =====================
        # DYNAMICS (TRAILER SAFE)
        # =====================
        for j in range(p.getNumJoints(self.car_id)):
            p.changeDynamics(
                self.car_id,
                j,
                lateralFriction=2.0,
                rollingFriction=0.02,
                spinningFriction=0.02,
                linearDamping=0.04,
                angularDamping=0.04,
            )

        self.steer_joints = []
        for ji in range(p.getNumJoints(self.car_id)):
            name = p.getJointInfo(self.car_id, ji)[1].decode("utf-8")
            if "steer" in name:
                self.steer_joints.append(ji)

        # =====================
        # ROAD PARAMETERS
        # =====================
        self.lane_width = 3.5
        self.num_lanes = 2
        self.road_length = 200.0
        self.lane_centers = [-self.lane_width / 2, +self.lane_width / 2]

        # =====================
        # SCENE CREATION
        # =====================
        self._create_road_and_lanes()
        self._create_grass()
        self._create_shoulders()
        self._create_trees()
        self._create_side_road()
        self._create_side_parking()



        # =====================
        # OBSTACLES
        # =====================
        self.obstacle_ids = []
        self.obstacle_states = []
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
    # GRASS VERGES
    # ==========================================================
    def _create_grass(self):
        grass_width = 20.0
        road_width = self.num_lanes * self.lane_width

        grass_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[self.road_length / 2, grass_width / 2, 0.01],
            rgbaColor=[0.1, 0.5, 0.1, 1],
        )

        for side in [-1, 1]:
            p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=grass_visual,
                basePosition=[
                    self.road_length / 2,
                    side * (road_width / 2 + grass_width / 2),
                    0.005,
                ],
            )

    # ==========================================================
    # ROAD SHOULDERS
    # ==========================================================
    def _create_shoulders(self):
        road_width = self.num_lanes * self.lane_width

        shoulder_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[self.road_length / 2, 0.3, 0.01],
            rgbaColor=[0.6, 0.6, 0.6, 1],
        )

        for side in [-1, 1]:
            p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=shoulder_visual,
                basePosition=[
                    self.road_length / 2,
                    side * (road_width / 2 + 0.3),
                    0.02,
                ],
            )

    # ==========================================================
    # TREES
    # ==========================================================
    def _spawn_tree(self, x, y):
        trunk = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.15,
            length=2.0,
            rgbaColor=[0.45, 0.25, 0.1, 1],
        )

        leaves = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.8,
            rgbaColor=[0.1, 0.6, 0.1, 1],
        )

        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=trunk,
            basePosition=[x, y, 1.0],
        )

        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=leaves,
            basePosition=[x, y, 2.3],
        )

    def _create_trees(self):
        tree_spacing = 12.0
        road_width = self.num_lanes * self.lane_width

        for x in np.arange(5, self.road_length, tree_spacing):
            self._spawn_tree(x, road_width / 2 + 6.0)
            self._spawn_tree(x, -(road_width / 2 + 6.0))

        # ==========================================================
    # SIDE ROAD (LEFT TURN)
    # ==========================================================
    def _create_side_road(self):
        self.side_road_x = 90.0
        self.side_road_length = 40.0
        self.side_road_width = 4.0

        road_width = self.num_lanes * self.lane_width
        side_center_y = road_width / 2 + self.side_road_length / 2

        # Side road asphalt
        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[
                self.side_road_width / 2,
                self.side_road_length / 2,
                0.02,
            ],
            rgbaColor=[0.15, 0.15, 0.15, 1],
        )

        col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[
                self.side_road_width / 2,
                self.side_road_length / 2,
                0.02,
            ],
        )

        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=vis,
            baseCollisionShapeIndex=col,
            basePosition=[
                self.side_road_x,
                side_center_y,
                0.02,
            ],
        )

        # Center line
        line_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.05, self.side_road_length / 2, 0.01],
            rgbaColor=[1, 1, 1, 1],
        )

        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=line_vis,
            basePosition=[
                self.side_road_x,
                side_center_y,
                0.04,
            ],
        )
    # ==========================================================
    # PARKING AT END OF SIDE ROAD
    # ==========================================================
    def _create_side_parking(self):
        parking_length = 10.0
        parking_width = 4.5

        road_width = self.num_lanes * self.lane_width

        parking_x = self.side_road_x
        parking_y = road_width / 2 + self.side_road_length + parking_length / 2

        # Parking asphalt
        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[parking_width / 2, parking_length / 2, 0.03],
            rgbaColor=[0.2, 0.2, 0.2, 1],
        )

        col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[parking_width / 2, parking_length / 2, 0.03],
        )

        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=vis,
            baseCollisionShapeIndex=col,
            basePosition=[parking_x, parking_y, 0.03],
        )

        # Parking lines
        line_thick = 0.05
        line_h = 0.02

        def line(dx, dy):
            return p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[dx, dy, line_h],
                rgbaColor=[1, 1, 1, 1],
            )

        for side in [-1, 1]:
            p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=line(parking_width / 2, line_thick),
                basePosition=[
                    parking_x,
                    parking_y + side * parking_length / 2,
                    0.07,
                ],
            )

        for side in [-1, 1]:
            p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=line(line_thick, parking_length / 2),
                basePosition=[
                    parking_x + side * parking_width / 2,
                    parking_y,
                    0.07,
                ],
            )



    # ==========================================================
    # OBSTACLES
    # ==========================================================
    def _create_obstacles(self):
        self._spawn_obstacle_car(10.0, self.lane_centers[1], speed=0.6)
        self._spawn_obstacle_car(25.0, self.lane_centers[0], speed=0.5)
        self._spawn_obstacle_car(40.0, self.lane_centers[1], speed=0.4)
        self._spawn_obstacle_car(55.0, self.lane_centers[0], speed=0.6)

    def _spawn_obstacle_car(self, x, y, speed=0.5, yaw=0.0):
        orn = p.getQuaternionFromEuler([0, 0, yaw])
        obs_id = p.loadURDF(
            self.obstacle_urdf,
            basePosition=[x, y, 0.2],
            baseOrientation=orn,
            useFixedBase=True,
        )
        self.obstacle_ids.append(obs_id)
        self.obstacle_states.append({"id": obs_id, "x": x, "y": y, "yaw": yaw, "v": speed})

    # ==========================================================
    # RESET / STEP
    # ==========================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([0.0, self.lane_centers[1], 0.0], dtype=np.float32)
        orn = p.getQuaternionFromEuler([0, 0, 0])
        p.resetBasePositionAndOrientation(self.car_id, [*self.state[:2], 0.2], orn)
        for j in self.steer_joints:
            p.resetJointState(self.car_id, j, 0.0)
        return self.state.copy(), {}

    def step(self, action):
        v, steer = np.clip(action, self.action_space.low, self.action_space.high)
        x, y, yaw = self.state
        L = 2.5

        x += v * np.cos(yaw) * self.dt
        y += v * np.sin(yaw) * self.dt
        yaw += (v * np.tan(steer) / L) * self.dt
        self.state = np.array([x, y, yaw], dtype=np.float32)

        orn = p.getQuaternionFromEuler([0, 0, yaw])
        p.resetBasePositionAndOrientation(self.car_id, [x, y, 0.2], orn)

        for j in self.steer_joints:
            p.setJointMotorControl2(
                self.car_id, j, p.POSITION_CONTROL, targetPosition=steer, force=50
            )

        p.stepSimulation()
        self._update_obstacles()

        collision = any(c[2] in self.obstacle_ids for c in p.getContactPoints(self.car_id))
        reward = -10.0 if collision else 0.0

        return self.state.copy(), reward, collision, False, {}

    def _update_obstacles(self):
        for obs in self.obstacle_states:
            obs["x"] += obs["v"] * self.dt
            orn = p.getQuaternionFromEuler([0, 0, obs["yaw"]])
            p.resetBasePositionAndOrientation(obs["id"], [obs["x"], obs["y"], 0.2], orn)

    def close(self):
        p.disconnect()
