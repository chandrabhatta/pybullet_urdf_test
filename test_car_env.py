from dynamic_env import SimpleCarEnv
import numpy as np
import time
import pybullet as p


# ==========================================================
# HIGHWAY / LEFT-TURN MPC
# ==========================================================
def mpc_control(state, obstacle_positions, dt, left_turn=False):
    x, y, yaw = state

    L = 2.5
    L_t = 3.0
    horizon = 12

    lane_width = 3.5
    lane_centers = [-lane_width / 2, lane_width / 2]

    current_lane = min(lane_centers, key=lambda ly: abs(y - ly))

    # ======================================================
    # TARGET SELECTION
    # ======================================================
    if left_turn:
        target_lane = current_lane + 1.5
        target_speed = 1.2
    else:
        target_lane = current_lane

        LOOKAHEAD_DIST = 10.0
        min_dist = np.inf

        for ox, oy in obstacle_positions:
            if abs(oy - current_lane) < 0.7 and ox > x:
                min_dist = min(min_dist, ox - x)

        if min_dist < LOOKAHEAD_DIST:
            target_lane = (
                lane_centers[0]
                if current_lane == lane_centers[1]
                else lane_centers[1]
            )

        if min_dist < 3.0:
            target_speed = 0.8
        elif min_dist < 5.0:
            target_speed = 1.2
        else:
            target_speed = 3.0

    steer_candidates = np.linspace(-0.45, 0.45, 13)

    best_cost = np.inf
    best_steer = 0.0

    # ======================================================
    # MPC ROLLOUT
    # ======================================================
    for steer in steer_candidates:
        px, py, pyaw = x, y, yaw
        yaw_trailer = yaw
        py_trailer = y - L_t * np.sin(yaw_trailer)

        cost = 0.0

        for _ in range(horizon):
            px += target_speed * np.cos(pyaw) * dt
            py += target_speed * np.sin(pyaw) * dt
            pyaw += target_speed * np.tan(steer) / L * dt

            yaw_trailer += (
                target_speed / L_t * np.sin(pyaw - yaw_trailer) * dt
            )
            py_trailer = py - L_t * np.sin(yaw_trailer)

            lane_error = target_lane - py

            if left_turn:
                yaw_ref = np.clip(0.6 * lane_error, -0.5, 0.5)
            else:
                yaw_ref = np.clip(0.6 * lane_error, -0.3, 0.3)

            cost += 20.0 * lane_error**2
            cost += 40.0 * (py_trailer - target_lane)**2
            cost += 10.0 * (pyaw - yaw_ref)**2
            cost += 2.0 * steer**2

        if cost < best_cost:
            best_cost = cost
            best_steer = steer

    return np.array([target_speed, best_steer], dtype=np.float32)


# ==========================================================
# PARKING CONTROLLER
# ==========================================================
def parking_control(state):
    x, y, yaw = state

    if abs(yaw) > 0.4:
        steer = np.clip(-1.4 * yaw, -0.45, 0.45)
        return np.array([0.6, steer], dtype=np.float32)

    parking_x = 100.0
    parking_length = 15.0
    parking_y = 6.0
    final_yaw = 0.0

    entry_x = parking_x - 3.0
    entry_y = parking_y - 1.5

    depth_reached = x > parking_x + 1.5

    if not depth_reached:
        dx = entry_x - x
        dy = entry_y - y

        yaw_ref = np.arctan2(dy, dx)
        yaw_err = np.arctan2(
            np.sin(yaw_ref - yaw),
            np.cos(yaw_ref - yaw)
        )

        steer = np.clip(1.4 * yaw_err, -0.45, 0.45)
        v = 0.9

    else:
        yaw_err = np.arctan2(
            np.sin(final_yaw - yaw),
            np.cos(final_yaw - yaw)
        )

        steer = np.clip(2.0 * yaw_err, -0.35, 0.35)

        if abs(yaw_err) > 0.2:
            v = 0.2
        else:
            v = 0.4

        if (
            abs(yaw_err) < 0.05
            and abs(y - parking_y) < 0.25
            and x > parking_x + parking_length / 2 - 0.5
        ):
            return np.array([0.0, 0.0], dtype=np.float32)

    return np.array([v, steer], dtype=np.float32)


# ==========================================================
# MAIN LOOP (FSM)
# ==========================================================
def main():
    env = SimpleCarEnv(gui=True)
    obs, _ = env.reset()
    car_id = env.car_id

    mode = "HIGHWAY"

    LEFT_TURN_START_X = 88.0
    PARKING_TRIGGER_X = 92.5

    for j in range(p.getNumJoints(car_id)):
        p.changeDynamics(
            car_id,
            j,
            lateralFriction=2.0,
            rollingFriction=0.02,
            spinningFriction=0.02
        )

    print("Highway → Left Turn → Parking FSM running")

    try:
        for step in range(3000):

            obstacle_positions = []
            for obs_id in env.obstacle_ids:
                pos, _ = p.getBasePositionAndOrientation(obs_id)
                obstacle_positions.append((pos[0], pos[1]))

            if mode == "HIGHWAY" and obs[0] > LEFT_TURN_START_X:
                mode = "LEFT_TURN"
                print("↩️ LEFT TURN")

            elif mode == "LEFT_TURN" and obs[0] > PARKING_TRIGGER_X:
                mode = "PARKING"
                print("PARKING")

            if mode == "HIGHWAY":
                action = mpc_control(obs, obstacle_positions, env.dt)

            elif mode == "LEFT_TURN":
                action = mpc_control(obs, obstacle_positions, env.dt, left_turn=True)

            elif mode == "PARKING":
                x, y, yaw = obs
                if abs(y - 6.0) < 0.25 and abs(yaw) < 0.06:
                    print("✅ PARKED")
                    mode = "PARKED"
                    env.parked = True
                    action = np.array([0.0, 0.0], dtype=np.float32)
                else:
                    action = parking_control(obs)

            elif mode == "PARKED":
                action = np.array([0.0, 0.0], dtype=np.float32)

            else:
                action = np.array([0.0, 0.0], dtype=np.float32)

            obs, _, terminated, truncated, _ = env.step(action)

            if mode == "PARKING":
                x, y, yaw = obs
                if abs(y - 6.0) < 0.3 and abs(yaw) < 0.08:
                    print("✅ PARKED")
                    mode = "PARKED"
                    env.parked = True
                    p.resetBaseVelocity(car_id, [0, 0, 0], [0, 0, 0])

            if mode == "PARKED":
                p.resetBaseVelocity(car_id, [0, 0, 0], [0, 0, 0])
                for j in range(p.getNumJoints(car_id)):
                    p.setJointMotorControl2(
                        car_id,
                        j,
                        p.VELOCITY_CONTROL,
                        targetVelocity=0,
                        force=0
                    )

            car_pos, _ = p.getBasePositionAndOrientation(car_id)
            p.resetDebugVisualizerCamera(
                cameraDistance=8.0,
                cameraYaw=270,
                cameraPitch=-20,
                cameraTargetPosition=car_pos
            )

            time.sleep(env.dt)

            if terminated or truncated:
                break

    finally:
        env.close()


if __name__ == "__main__":
    main()
