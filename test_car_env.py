from dynamic_env import SimpleCarEnv
import numpy as np
import time
import pybullet as p


# ==========================================================
# MPC CONTROLLER 
# ==========================================================
def mpc_control(state, obstacle_positions, dt):
    x, y, yaw = state

    L = 2.5
    L_t = 3.0
    horizon = 12

    lane_width = 3.5
    lane_centers = [-lane_width / 2, lane_width / 2]

    current_lane = min(lane_centers, key=lambda ly: abs(y - ly))
    target_lane = current_lane

    LOOKAHEAD_DIST = 10
    min_dist = np.inf

    for ox, oy in obstacle_positions:
        if abs(oy - current_lane) < 0.7 and ox > x:
            min_dist = min(min_dist, ox - x)

    if min_dist < LOOKAHEAD_DIST:
        target_lane = lane_centers[0] if current_lane == lane_centers[1] else lane_centers[1]

    if min_dist < 3.0:
        target_speed = 0.8
    elif min_dist < 5.0:
        target_speed = 1.2
    else:
        target_speed = 3

    steer_candidates = np.linspace(-0.3, 0.3, 9)

    best_cost = np.inf
    best_steer = 0.0

    for steer in steer_candidates:
        px, py, pyaw = x, y, yaw
        yaw_trailer = yaw
        py_trailer = y - L_t * np.sin(yaw_trailer)

        cost = 0.0

        for _ in range(horizon):
            px += target_speed * np.cos(pyaw) * dt
            py += target_speed * np.sin(pyaw) * dt
            pyaw += (target_speed * np.tan(steer) / L) * dt

            yaw_trailer += (
                target_speed / L_t
                * np.sin(pyaw - yaw_trailer)
                * dt
            )
            py_trailer = py - L_t * np.sin(yaw_trailer)

            lane_error = target_lane - py
            yaw_ref = np.clip(0.6 * lane_error, -0.3, 0.3)

            cost += 20.0 * (py - target_lane) ** 2
            cost += 40.0 * (py_trailer - target_lane) ** 2
            cost += 8.0 * (pyaw - yaw_ref) ** 2
            cost += 2.0 * steer ** 2

            if abs(py_trailer) > lane_width:
                cost += 600.0

        if cost < best_cost:
            best_cost = cost
            best_steer = steer

    return np.array([target_speed, best_steer], dtype=np.float32)


# ==========================================================
# MAIN LOOP
# ==========================================================
def main():
    env = SimpleCarEnv(gui=True)
    obs, info = env.reset()

    car_id = env.car_id

    # ======================================================
    #  CRITICAL PHYSICS FIXES (ONE-TIME SETUP)
    # ======================================================

    for j in range(p.getNumJoints(car_id)):
        joint_name = p.getJointInfo(car_id, j)[1].decode()

        # Global friction tuning
        p.changeDynamics(
            car_id,
            j,
            lateralFriction=2.0,
            rollingFriction=0.02,
            spinningFriction=0.02
        )

        # Trailer wheels → FREE ROLLING
        if "trailer_wheel" in joint_name:
            p.setJointMotorControl2(
                car_id,
                j,
                p.VELOCITY_CONTROL,
                targetVelocity=0,
                force=0
            )

        # Trailer hinge → FREE + DAMPED
        if "trailer_hinge" in joint_name:
            p.setJointMotorControl2(
                car_id,
                j,
                p.VELOCITY_CONTROL,
                targetVelocity=0,
                force=0
            )

    print("✅ Physics initialized: wheels roll, trailer articulates")

    try:
        for step in range(2000):

            obstacle_positions = []
            for obs_id in env.obstacle_ids:
                pos, _ = p.getBasePositionAndOrientation(obs_id)
                obstacle_positions.append((pos[0], pos[1]))

            action = mpc_control(obs, obstacle_positions, env.dt)
            obs, reward, terminated, truncated, info = env.step(action)

            if step % 25 == 0:
                print(f"step {step}, obs={obs}, action={action}")

            car_pos, _ = p.getBasePositionAndOrientation(car_id)
            p.resetDebugVisualizerCamera(
                cameraDistance=8.0,
                cameraYaw=270.0,
                cameraPitch=-20.0,
                cameraTargetPosition=car_pos
            )

            time.sleep(env.dt)

            if terminated or truncated:
                print("Episode ended")
                break

    except KeyboardInterrupt:
        print("Stopped by user")

    finally:
        env.close()


if __name__ == "__main__":
    main()
