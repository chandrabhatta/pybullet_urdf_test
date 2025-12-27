import pybullet as p
import pybullet_data
import time
import os
import math

print("Starting Dutch Highway Simulation...")

# Connect to PyBullet with GUI
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# Project root
project_root = os.path.dirname(os.path.abspath(__file__))

# Load Dutch highway URDF
highway_urdf_path = os.path.join(project_root, "urdf", "dutch_highway.urdf")
print(f"Loading highway: {highway_urdf_path}")

highway_id = p.loadURDF(
    highway_urdf_path,
    basePosition=[0, 0, 0],
    baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
    useFixedBase=True
)
print("Highway loaded!")

# Dutch highway configuration - 2 lanes
NUM_LANES = 2
LANE_WIDTH = 3.5
LANE_CENTERS = [1.75, 5.25]  # Lane 0: right (slow), Lane 1: left (fast)

print(f"Lane centers: {LANE_CENTERS}")

# Load the car
urdf_path = os.path.join(project_root, "urdf", "simple_car.urdf")
INITIAL_LANE = 0  # Start in right lane

car_id = p.loadURDF(
    urdf_path,
    basePosition=[10, LANE_CENTERS[INITIAL_LANE], 0.4],
    baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
    useFixedBase=False
)
print("Car loaded!")

# Set up sliders
velocity_slider = p.addUserDebugParameter("Velocity (m/s)", 0, 25, 8)
lane_slider = p.addUserDebugParameter("Target Lane (0=Right, 1=Left)", 0, 1, 0)

# Set camera to good viewing angle
p.resetDebugVisualizerCamera(
    cameraDistance=20,
    cameraYaw=0,
    cameraPitch=-45,
    cameraTargetPosition=[20, 4, 0]
)

print("\n" + "="*55)
print("   DUTCH HIGHWAY SIMULATION - 2 Lane MPC Demo")
print("="*55)
print("\nLane Layout:")
print("  Lane 0 (Right/Slow): Y = 1.75m")
print("  Lane 1 (Left/Fast):  Y = 5.25m")
print("\nControls:")
print("  - Velocity: Adjust car speed (0-25 m/s)")
print("  - Target Lane: 0 = Right lane, 1 = Left lane")
print("\nClose window to stop simulation")
print("="*55 + "\n")


def compute_steering(pos, vel, yaw, target_lane):
    """
    Improved PD controller with heading correction for lane switching.
    """
    target_y = LANE_CENTERS[target_lane]
    y_error = target_y - pos[1]
    vy = vel[1]

    # Desired heading angle to reach target lane
    lookahead = 5.0  # meters
    approach_angle = math.atan2(y_error, lookahead)

    # Heading error
    heading_error = approach_angle - yaw
    # Normalize to [-pi, pi]
    heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))

    # PD controller gains
    Kp_lateral = 0.4
    Kd_lateral = 0.8
    Kp_heading = 1.0

    steer = Kp_lateral * y_error - Kd_lateral * vy + Kp_heading * heading_error
    return max(-0.5, min(0.5, steer))


# Main simulation loop
dt = 1.0 / 240.0
frame = 0

while p.isConnected():
    try:
        # Get slider values
        target_vel = p.readUserDebugParameter(velocity_slider)
        target_lane_raw = p.readUserDebugParameter(lane_slider)
        target_lane = int(round(target_lane_raw))
        target_lane = max(0, min(NUM_LANES - 1, target_lane))

        # Get car state
        pos, orn = p.getBasePositionAndOrientation(car_id)
        vel, ang_vel = p.getBaseVelocity(car_id)
        euler = p.getEulerFromQuaternion(orn)
        yaw = euler[2]

        # Compute steering using improved controller
        steer = compute_steering(pos, vel, yaw, target_lane)

        # Apply steering to front wheels (joints 0 and 1)
        p.setJointMotorControl2(car_id, 0, p.POSITION_CONTROL, targetPosition=steer, force=100)
        p.setJointMotorControl2(car_id, 1, p.POSITION_CONTROL, targetPosition=steer, force=100)

        # Move the car forward and handle lane changes
        vx = target_vel * math.cos(yaw)

        # Calculate lateral velocity for lane changing
        target_y = LANE_CENTERS[target_lane]
        y_error = target_y - pos[1]

        # Proportional lateral velocity for lane change
        vy_lane = 2.0 * y_error  # Gain for responsive lane change
        vy_lane = max(-3.0, min(3.0, vy_lane))  # Clamp lateral speed to 3 m/s

        p.resetBaseVelocity(car_id, linearVelocity=[vx, vy_lane, vel[2]])

        # Update camera to follow car
        p.resetDebugVisualizerCamera(
            cameraDistance=20,
            cameraYaw=0,
            cameraPitch=-45,
            cameraTargetPosition=[pos[0] + 15, 4, 0]
        )

        # Print status every 2 seconds
        frame += 1
        if frame % 480 == 0:
            speed = math.sqrt(vel[0]**2 + vel[1]**2)
            lane_name = "Right" if target_lane == 0 else "Left"
            current_lane = 0 if pos[1] < 3.5 else 1
            current_lane_name = "Right" if current_lane == 0 else "Left"
            print(f"X: {pos[0]:.1f}m | Y: {pos[1]:.2f}m | Speed: {speed:.1f} m/s | "
                  f"Current: {current_lane_name} | Target: {lane_name}")

        p.stepSimulation()
        time.sleep(dt)

    except Exception as e:
        print(f"Error: {e}")
        break

print("Simulation ended")
p.disconnect()
