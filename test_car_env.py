from dynamic_env import SimpleCarEnv
import numpy as np
import time
import pybullet as p


# ==========================================================
# MPC CONTROLLER (HIGHWAY + LEFT TURN)
# ==========================================================
def mpc_control(state, obstacle_positions, dt):
    x, y, yaw = state

    # =============================
    # VEHICLE PARAMETERS
    # =============================
    L = 2.5
    L_t = 3.0
    horizon = 12

    # =============================
    # ROAD GEOMETRY
    # =============================
    lane_width = 3.5
    lane_centers = [-lane_width / 2, lane_width / 2]

    LEFT_TURN_X = 87.0
    SIDE_ROAD_LANE_Y = 8.0
    TURN_YAW = np.pi / 2

    TURN_ACTIVE = x > LEFT_TURN_X

    # =============================
    # CURRENT LANE (HIGHWAY)
    # =============================
    current_lane = min(lane_centers, key=lambda ly: abs(y - ly))
    target_lane = current_lane

    # =============================
    # OBSTACLE LOOKAHEAD (HIGHWAY ONLY)
    # =============================
    LOOKAHEAD_DIST = 10.0
    min_dist = np.inf

    if not TURN_ACTIVE:
        for ox, oy in obstacle_positions:
            if abs(oy - current_lane) < 0.7 and ox > x:
                min_dist = min(min_dist, ox - x)

        if min_dist < LOOKAHEAD_DIST:
            target_lane = (
                lane_centers[0]
                if current_lane == lane_centers[1]
                else lane_centers[1]
            )

    # =============================
    # SPEED PROFILE
    # =============================
    if TURN_ACTIVE:
        target_speed = 1.0
    else:
        if min_dist < 3.0:
            target_speed = 0.8
        elif min_dist < 5.0:
            target_speed = 1.2
        else:
            target_speed = 3.0

    # =============================
    # MPC SEARCH
    # =============================
    steer_candidates = np.linspace(-0.35, 0.35, 11)

    best_cost = np.inf
    best_steer = 0.0

    for steer in steer_candidates:
        px, py, pyaw = x, y, yaw
        yaw_trailer = yaw
        py_trailer = py - L_t * np.sin(yaw_trailer)

        cost = 0.0

        for _ in range(horizon):
            # ===== rollout vehicle =====
            px += target_speed * np.cos(pyaw) * dt
            py += target_speed * np.sin(pyaw) * dt
            pyaw += (target_speed * np.tan(steer) / L) * dt

            # ===== rollout trailer =====
            yaw_trailer += (
                target_speed / L_t
                * np.sin(pyaw - yaw_trailer)
                * dt
            )
            py_trailer = py - L_t * np.sin(yaw_trailer)

            # =============================
            # COST FUNCTION
            # =============================
            if px > LEFT_TURN_X:
                # ===== LEFT TURN MODE =====
                yaw_ref = TURN_YAW

                cost += 25.0 * (py - SIDE_ROAD_LANE_Y) ** 2
                cost += 50.0 * (py_trailer - SIDE_ROAD_LANE_Y) ** 2
                cost += 12.0 * (pyaw - yaw_ref) ** 2
                cost += 2.0 * steer ** 2

            else:
                # ===== HIGHWAY MODE =====
                lane_error = target_lane - py
                yaw_ref = np.clip(0.6 * lane_error, -0.3, 0.3)

                cost += 20.0 * (py - target_lane) ** 2
                cost += 40.0 * (py_trailer - target_lane) ** 2
                cost += 8.0 * (pyaw - yaw_ref) ** 2
                cost += 2.0 * steer ** 2

            # ===== hard bounds =====
            if abs(py_trailer) > 2.5 * lane_width:
                cost += 800.0

        if cost < best_cost:
            best_cost = cost
            best_steer = steer

    return np.array([target_speed, best_steer], dtype=np.float32)


# ==========================================================
# MAIN LOOP
# ==========================================================
def main():
    env = SimpleCarEnv(gui=True)
    obs, _ = env.reset()

    car_id = env.car_id

    # ======================================================
    # PHYSICS INITIALIZATION
    # ======================================================
    for j in range(p.getNumJoints(car_id)):
        name = p.getJointInfo(car_id, j)[1].decode()

        p.changeDynamics(
            car_id,
            j,
            lateralFriction=2.0,
            rollingFriction=0.02,
            spinningFriction=0.02
        )

        if "trailer_wheel" in name or "trailer_hinge" in name:
            p.setJointMotorControl2(
                car_id,
                j,
                p.VELOCITY_CONTROL,
                targetVelocity=0,
                force=0
            )

    print("✅ Physics initialized")

    try:
        for step in range(2000):
            obstacle_positions = []
            for obs_id in env.obstacle_ids:
                pos, _ = p.getBasePositionAndOrientation(obs_id)
                obstacle_positions.append((pos[0], pos[1]))

            action = mpc_control(obs, obstacle_positions, env.dt)
            obs, reward, terminated, truncated, _ = env.step(action)

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

    finally:
        env.close()


if __name__ == "__main__":
    main()
