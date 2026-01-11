"""
Test RRT Lane-Change with Car+Trailer in PyBullet.

This script demonstrates:
1. 2-lane road environment with static obstacle cars
2. Car+trailer ego vehicle
3. RRT path planning for lane change
4. Path execution visualization in PyBullet
5. Video recording and export to MP4
"""

import os
import sys
import time
import numpy as np
import pybullet as p
import pybullet_data
import imageio

# Add RRT folder to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'RRT'))

from RRT.rrt import RRT
from RRT.vehicle_model import CarTrailerModel
from RRT.collision import CollisionChecker


# ============================================================
# Configuration
# ============================================================
LANE_WIDTH = 3.5
NUM_LANES = 2
ROAD_LENGTH = 100.0

# Lane centers (y-coordinates)
LANE_0_Y = 1.75   # Right lane (start)
LANE_1_Y = 5.25   # Left lane (goal)

# Video settings
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
VIDEO_FPS = 30

# Obstacle positions: (x, y) - static cars
OBSTACLES = [
    (15.0, LANE_1_Y),   # Lane 1
    (30.0, LANE_0_Y),   # Lane 0
    (45.0, LANE_1_Y),   # Lane 1
    (60.0, LANE_0_Y),   # Lane 0
]

# Start and goal states for car+trailer: [x_trailer, y_trailer, theta_car, theta_trailer]
START_STATE = np.array([5.0, LANE_0_Y, 0.0, 0.0])
GOAL_STATE = np.array([80.0, LANE_1_Y, 0.0, 0.0])


class VideoRecorder:
    """Records PyBullet frames and exports to MP4."""

    def __init__(self, output_path, width=VIDEO_WIDTH, height=VIDEO_HEIGHT, fps=VIDEO_FPS):
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.frames = []
        self.view_matrix = None
        self.proj_matrix = None

    def setup_camera(self, target_pos, distance=35, yaw=0, pitch=-50):
        """Setup camera view matrix and projection matrix."""
        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=target_pos,
            distance=distance,
            yaw=yaw,
            pitch=pitch,
            roll=0,
            upAxisIndex=2
        )
        self.proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=self.width / self.height,
            nearVal=0.1,
            farVal=100.0
        )

    def capture_frame(self):
        """Capture current frame from PyBullet."""
        if self.view_matrix is None:
            self.setup_camera([40, 3.5, 0])

        _, _, rgb_img, _, _ = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.proj_matrix,
            renderer=p.ER_TINY_RENDERER
        )
        # Convert RGBA to RGB
        rgb_array = np.array(rgb_img, dtype=np.uint8).reshape(self.height, self.width, 4)
        rgb_array = rgb_array[:, :, :3]  # Remove alpha channel
        self.frames.append(rgb_array)

    def save(self):
        """Save frames to MP4 file."""
        if not self.frames:
            print("No frames to save!")
            return

        print(f"\nSaving video to {self.output_path}...")
        print(f"  Frames: {len(self.frames)}, FPS: {self.fps}")
        imageio.mimwrite(self.output_path, self.frames, fps=self.fps)
        print(f"  Video saved successfully!")


def create_road(road_length=ROAD_LENGTH):
    """Create 2-lane road with markings."""
    road_width = NUM_LANES * LANE_WIDTH

    # Road surface (dark gray)
    road_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[road_length / 2, road_width / 2, 0.01],
        rgbaColor=[0.15, 0.15, 0.15, 1],
    )
    p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=road_visual,
        basePosition=[road_length / 2, road_width / 2, 0.01],
    )

    # Lane markings
    for i in range(NUM_LANES + 1):
        y = i * LANE_WIDTH
        color = [1, 1, 1, 1] if i in [0, NUM_LANES] else [1, 1, 0, 1]

        marking = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[road_length / 2, 0.05, 0.02],
            rgbaColor=color,
        )
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=marking,
            basePosition=[road_length / 2, y, 0.02],
        )


def create_grass(road_length=ROAD_LENGTH):
    """Create grass verges on both sides of the road."""
    grass_width = 20.0
    road_width = NUM_LANES * LANE_WIDTH

    grass_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[road_length / 2, grass_width / 2, 0.01],
        rgbaColor=[0.1, 0.5, 0.1, 1],
    )

    # Grass on both sides
    for side in [0, 1]:
        y_pos = road_width + grass_width / 2 if side == 1 else -grass_width / 2
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=grass_visual,
            basePosition=[road_length / 2, y_pos, 0.005],
        )


def create_shoulders(road_length=ROAD_LENGTH):
    """Create road shoulders."""
    road_width = NUM_LANES * LANE_WIDTH

    shoulder_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[road_length / 2, 0.3, 0.01],
        rgbaColor=[0.6, 0.6, 0.6, 1],
    )

    # Shoulders on both sides
    for side in [0, 1]:
        y_pos = road_width + 0.3 if side == 1 else -0.3
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=shoulder_visual,
            basePosition=[road_length / 2, y_pos, 0.02],
        )


def spawn_tree(x, y):
    """Spawn a single tree at (x, y)."""
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


def create_trees(road_length=ROAD_LENGTH):
    """Create trees along the roadside."""
    tree_spacing = 12.0
    road_width = NUM_LANES * LANE_WIDTH

    for x in np.arange(5, road_length, tree_spacing):
        spawn_tree(x, road_width + 6.0)  # Right side
        spawn_tree(x, -6.0)               # Left side


def spawn_obstacle_cars(urdf_path):
    """Spawn static obstacle cars."""
    obstacle_ids = []
    for x, y in OBSTACLES:
        orn = p.getQuaternionFromEuler([0, 0, 0])
        obs_id = p.loadURDF(
            urdf_path,
            basePosition=[x, y, 0.2],
            baseOrientation=orn,
            useFixedBase=True,
        )
        obstacle_ids.append(obs_id)
        print(f"  Spawned obstacle at ({x}, {y})")
    return obstacle_ids


def draw_path(path, color=[1, 1, 0]):
    """Draw the planned path as debug lines in PyBullet."""
    for i in range(len(path) - 1):
        # For car+trailer state: [x1, y1, theta0, theta1]
        # x1, y1 is trailer position
        p1 = [path[i][0], path[i][1], 0.5]
        p2 = [path[i + 1][0], path[i + 1][1], 0.5]
        p.addUserDebugLine(p1, p2, lineColorRGB=color, lineWidth=2)


def get_car_position_from_trailer(state, model):
    """
    Get car (tractor) position from trailer state.
    State: [x_trailer, y_trailer, theta_car, theta_trailer]
    Uses the model's get_tractor_position method.
    """
    tractor_state = model.get_tractor_position(state)
    return tractor_state[0], tractor_state[1], tractor_state[2]


def execute_path(car_id, path, model, dt=0.05, video_recorder=None):
    """Execute the planned path in PyBullet."""
    print(f"\nExecuting path with {len(path)} waypoints...")

    # Find the trailer hinge joint index
    hinge_joint = None
    for i in range(p.getNumJoints(car_id)):
        joint_info = p.getJointInfo(car_id, i)
        joint_name = joint_info[1].decode('utf-8')
        if 'trailer_hinge' in joint_name:
            hinge_joint = i
            print(f"  Found trailer hinge joint at index {i}")
            break

    if hinge_joint is None:
        print("  WARNING: No trailer hinge joint found!")

    for i, state in enumerate(path):
        # Get car position from trailer state
        car_x, car_y, theta_car = get_car_position_from_trailer(state, model)
        theta_trailer = state[3]

        # Hitch angle (relative angle between car and trailer)
        # Note: The angle should be theta1 - theta0 (trailer relative to car)
        hitch_angle = theta_trailer - theta_car

        # Set car base position and orientation
        orn = p.getQuaternionFromEuler([0, 0, theta_car])
        p.resetBasePositionAndOrientation(car_id, [car_x, car_y, 0.2], orn)

        # Set trailer hinge joint angle
        if hinge_joint is not None:
            p.resetJointState(car_id, hinge_joint, hitch_angle)

        # Step simulation multiple times for smoother visualization
        for _ in range(4):
            p.stepSimulation()

        # Capture frame for video (every frame)
        if video_recorder is not None:
            video_recorder.capture_frame()
        else:
            # If not recording, add delay for live viewing
            time.sleep(dt)

        if (i + 1) % 50 == 0:
            print(f"  Waypoint {i + 1}/{len(path)} - pos: ({car_x:.1f}, {car_y:.1f})")

    print("Path execution complete!")


def main():
    print("=" * 60)
    print("  RRT Lane-Change Test with Car+Trailer")
    print("=" * 60)

    # ============================================================
    # Setup PyBullet
    # ============================================================
    print("\n[1] Setting up PyBullet environment...")
    physics_id = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1 / 240)

    # Load ground plane
    p.loadURDF("plane.urdf")

    # Create road and environment
    create_road()
    create_grass()
    create_shoulders()
    create_trees()

    # Set camera view (top-down)
    p.resetDebugVisualizerCamera(
        cameraDistance=50,
        cameraYaw=0,
        cameraPitch=-89.9,  # Top-down view
        cameraTargetPosition=[50, 3.5, 0]
    )

    # ============================================================
    # Spawn obstacles
    # ============================================================
    print("\n[2] Spawning obstacle cars...")
    project_root = os.path.dirname(os.path.abspath(__file__))
    obstacle_urdf = os.path.join(project_root, "urdf", "obstacle_car.urdf")
    obstacle_ids = spawn_obstacle_cars(obstacle_urdf)

    # ============================================================
    # Load ego car+trailer
    # ============================================================
    print("\n[3] Loading ego car+trailer...")
    ego_urdf = os.path.join(project_root, "urdf", "simple_car.urdf")

    # Initial position (in lane 0)
    start_car_x = START_STATE[0] + 3.0  # Offset for trailer length
    start_car_y = START_STATE[1]
    orn = p.getQuaternionFromEuler([0, 0, 0])

    car_id = p.loadURDF(
        ego_urdf,
        basePosition=[start_car_x, start_car_y, 0.2],
        baseOrientation=orn,
        useFixedBase=False,
    )
    print(f"  Loaded car+trailer at ({start_car_x}, {start_car_y})")

    # ============================================================
    # Setup RRT Planner
    # ============================================================
    print("\n[4] Setting up RRT planner...")

    # Create car+trailer model
    model = CarTrailerModel(L0=2.5, L1=3.0, max_steer=0.5)

    # Create collision checker with obstacles
    collision_checker = CollisionChecker(
        y_min=0.0,
        y_max=7.0,
        x_min=-5.0,
        x_max=ROAD_LENGTH + 10,
        safety_margin=0.3
    )

    # Add obstacle cars to collision checker
    for x, y in OBSTACLES:
        collision_checker.add_obstacle(x, y, length=3.0, width=1.6, yaw=0.0)
        print(f"  Added obstacle to collision checker at ({x}, {y})")

    # ============================================================
    # Plan path with RRT
    # ============================================================
    print("\n[5] Planning lane-change path with RRT...")
    print(f"  Start: {START_STATE}")
    print(f"  Goal:  {GOAL_STATE}")

    # Create RRT planner
    bounds = (0, ROAD_LENGTH, 0, 7)  # (x_min, x_max, y_min, y_max)
    rrt = RRT(
        start=START_STATE,
        goal=GOAL_STATE,
        model=model,
        collision_checker=collision_checker,
        bounds=bounds
    )

    # Plan
    result = rrt.plan(max_iters=3000, verbose=True)

    # Handle new dictionary return format
    if isinstance(result, dict):
        path = result.get('path')
        if not result.get('success', False) or path is None:
            print("\n[ERROR] No path found! Try increasing max_iters or adjusting obstacles.")
            print("Keeping simulation open for inspection...")
            try:
                while True:
                    p.stepSimulation()
                    time.sleep(0.01)
            except KeyboardInterrupt:
                p.disconnect()
                return
    else:
        # Backwards compatibility with old return format
        path = result
        if path is None:
            print("\n[ERROR] No path found! Try increasing max_iters or adjusting obstacles.")
            print("Keeping simulation open for inspection...")
            try:
                while True:
                    p.stepSimulation()
                    time.sleep(0.01)
            except KeyboardInterrupt:
                p.disconnect()
                return

    print(f"\n[SUCCESS] Path found with {len(path)} waypoints!")

    # ============================================================
    # Visualize and execute path (SMOOTH - no video recording)
    # ============================================================
    print("\n[6] Drawing planned path...")
    draw_path(path, color=[1, 1, 0])  # Yellow path

    # Draw start and goal markers
    p.addUserDebugText("START", [START_STATE[0], START_STATE[1], 1], textColorRGB=[0, 1, 0])
    p.addUserDebugText("GOAL", [GOAL_STATE[0], GOAL_STATE[1], 1], textColorRGB=[1, 0, 0])

    print("\n[7] Executing path (smooth animation)...")
    time.sleep(1)  # Pause before starting
    execute_path(car_id, path, model, dt=0.02, video_recorder=None)

    # ============================================================
    # Video Recording (separate pass)
    # ============================================================
    print("\n[8] Recording video (replaying path)...")
    video_path = os.path.join(project_root, "rrt_lane_change.mp4")
    video_recorder = VideoRecorder(video_path)
    video_recorder.setup_camera([50, 3.5, 0], distance=50, yaw=0, pitch=-89.9)  # Top-down view

    # Reset car to start position
    start_car_x, start_car_y, start_theta = get_car_position_from_trailer(START_STATE, model)
    orn = p.getQuaternionFromEuler([0, 0, start_theta])
    p.resetBasePositionAndOrientation(car_id, [start_car_x, start_car_y, 0.2], orn)

    # Capture initial scene
    for _ in range(VIDEO_FPS):
        p.stepSimulation()
        video_recorder.capture_frame()

    # Replay path for recording (faster, no display delay)
    print("  Recording path...")
    execute_path(car_id, path, model, dt=0, video_recorder=video_recorder)

    # Capture final scene
    for _ in range(VIDEO_FPS):
        p.stepSimulation()
        video_recorder.capture_frame()

    # Save video
    print("\n[9] Saving video...")
    video_recorder.save()
    print(f"  Video saved to: {video_path}")

    # ============================================================
    # Keep simulation running
    # ============================================================
    print("\n[10] Simulation complete! Window staying open.")
    print("    Press Ctrl+C to exit.")

    try:
        while True:
            p.stepSimulation()
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nExiting...")
        p.disconnect()


if __name__ == "__main__":
    main()
