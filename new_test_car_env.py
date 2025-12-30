from car_env import SimpleCarTrailerEnv
import numpy as np
import time
import pybullet as p


# ==========================================================
# MPC CONTROLLER (CAR + SINGLE TRAILER)
# ==========================================================
def mpc_control(state, obstacle_positions, dt):
    # CHANGED: state dimension
    x, y, theta, phi = state

    # =============================
    # VEHICLE PARAMETERS (LaValle)
    # =============================
    L1 = 2.5   # car wheelbase
    L2 = 3.0   # trailer length

    # =============================
    # MPC PARAMETERS
    # =============================
    horizon = 10

    lane_width = 3.5
    lane_centers = [-lane_width / 2, lane_width / 2]

    # =============================
    # CURRENT + TARGET LANE
    # =============================
    current_lane = min(lane_centers, key=lambda ly: abs(y - ly))
    target_lane = current_lane

    # =============================
    # OBSTACLE LOOKAHEAD
    # =============================
    LOOKAHEAD_DIST = 6.0
    min_dist = np.inf

    for ox, oy in obstacle_positions:
        same_lane = abs(oy - current_lane) < 0.6
        if same_lane and ox > x:
            min_dist = min(min_dist, ox - x)

    obstacle_ahead = min_dist < LOOKAHEAD_DIST

    # =============================
    # LANE CHANGE LOGIC
    # =============================
    if obstacle_ahead:
        target_lane = (
            lane_centers[0]
            if current_lane == lane_centers[1]
            else lane_centers[1]
        )

    # =============================
    # SPEED CONTROL
    # =============================
    if min_dist < 2.5:
        target_speed = 0.7
    elif min_dist < 4.0:
        target_speed = 1.0
    elif min_dist < 6.0:
        target_speed = 1.4
    else:
        target_speed = 1.8

    # =============================
    # MPC OPTIMIZATION
    # =============================
    steer_candidates = np.linspace(-0.3, 0.3, 7)

    best_cost = np.inf
    best_steer = 0.0

    for steer in steer_candidates:
        px, py, ptheta, pphi = x, y, theta, phi
        cost = 0.0

        for _ in range(horizon):
            # ===== LaValle car + trailer prediction =====
            px += target_speed * np.cos(ptheta) * dt
            py += target_speed * np.sin(ptheta) * dt

            ptheta += (target_speed / L1) * np.tan(steer) * dt
            pphi += (
                -(target_speed / L2) * np.sin(pphi)
                + (target_speed / L1) * np.tan(steer)
            ) * dt

            # =============================
            # COST FUNCTION
            # =============================
            cost += 15.0 * (py - target_lane) ** 2    # lane keeping
            cost += 2.0 * (ptheta ** 2)               # heading stability
            cost += 5.0 * (pphi ** 2)                 # trailer stability
            cost += 1.0 * (steer ** 2)                # steering smoothness

            # HARD ROAD BOUNDS
            if abs(py) > lane_width:
                cost += 200.0

        if cost < best_cost:
            best_cost = cost
            best_steer = steer

    return np.array([target_speed, best_steer], dtype=np.float32)


# ==========================================================
# MAIN LOOP
# ==========================================================
def main():
    env = SimpleCarTrailerEnv(gui=True)

    obs, info = env.reset()
    print("Initial obs:", obs)

    try:
        for step in range(1000):

            # Get obstacle positions
            obstacle_positions = []
            for obs_id in env.obstacle_ids:
                pos, _ = p.getBasePositionAndOrientation(obs_id)
                obstacle_positions.append((pos[0], pos[1]))

            # MPC chooses action
            action = mpc_control(obs, obstacle_positions, env.dt)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if step % 25 == 0:
                print(f"step {step}, obs={obs}, action={action}")

            # =========================
            # CAMERA: FOLLOW FROM BEHIND
            # =========================
            car_pos, _ = p.getBasePositionAndOrientation(env.car_id)

            p.resetDebugVisualizerCamera(
                cameraDistance=8.0,
                cameraYaw=270.0,
                cameraPitch=-20.0,
                cameraTargetPosition=car_pos
            )

            time.sleep(env.dt)

            if done:
                print("Episode ended due to collision")
                break

    except KeyboardInterrupt:
        print("Stopped by user")

    finally:
        env.close()


if __name__ == "__main__":
    main()
