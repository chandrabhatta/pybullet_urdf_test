from dynamic_env import SimpleCarEnv
import numpy as np
import time
import pybullet as p


# ==========================================================
# FRENET MPC CONTROLLER (FIXED + STABLE)
# ==========================================================
def mpc_control(state, obstacle_positions, dt):
    x, y, yaw = state

    # -----------------------------
    # Frenet (correct for your env)
    # -----------------------------
    s = x
    d = y
    psi = yaw

    L = 2.5
    horizon = 20

    lane_width = 3.5
    lane_centers = [-lane_width / 2, lane_width / 2]

    current_lane = min(lane_centers, key=lambda lc: abs(d - lc))
    target_lane = current_lane

    # -----------------------------
    # OBSTACLE DETECTION
    # -----------------------------
   # -----------------------------

    LOOKAHEAD = 12.0
    VEHICLE_HALF_WIDTH = 0.9   # car + safety margin

    min_s_dist = np.inf
    obstacle_ahead = False

    for ox, oy in obstacle_positions:
        obs_s = ox
        obs_d = oy

        longitudinal_ok = 0.5 < (obs_s - s) < LOOKAHEAD
        lateral_overlap = abs(obs_d - d) < VEHICLE_HALF_WIDTH

        if longitudinal_ok and lateral_overlap:
            dist = obs_s - s
            if dist < min_s_dist:
                min_s_dist = dist
                obstacle_ahead = True


    # -----------------------------
    # LANE CHANGE DECISION
    # -----------------------------
    if obstacle_ahead:
        target_lane = (
            lane_centers[0]
            if current_lane == lane_centers[1]
            else lane_centers[1]
        )

    # -----------------------------
    # SMOOTH LANE CHANGE PROFILE
    # -----------------------------
    s0 = s
    LC_LENGTH = 14.0

    def d_reference(ps):
        if not obstacle_ahead:
            return current_lane

        ds = np.clip(ps - s0, 0.0, LC_LENGTH)
        alpha = ds / LC_LENGTH

        return (
            current_lane
            + 0.5 * (target_lane - current_lane)
            * (1 - np.cos(np.pi * alpha))
        )

    # -----------------------------
    # SPEED POLICY
    # -----------------------------
    target_speed = 8
    if obstacle_ahead and min_s_dist < 4.0:
        target_speed = 1.2
    if abs(d - target_lane) > 0.5:
        target_speed = min(target_speed, 1.0)

    # -----------------------------
    # MPC OPTIMIZATION
    # -----------------------------
    steer_candidates = np.linspace(-0.35, 0.35, 11)
    best_cost = np.inf
    best_steer = 0.0

    for steer in steer_candidates:
        ps, pd, ppsi = s, d, psi
        cost = 0.0

        for _ in range(horizon):
            ps += target_speed * np.cos(ppsi) * dt
            pd += target_speed * np.sin(ppsi) * dt
            ppsi += (target_speed * np.tan(steer) / L) * dt

            d_ref = d_reference(ps)

            cost += 40.0 * (pd - d_ref) ** 2
            cost += 3.0 * (ppsi ** 2)
            cost += 1.5 * (steer ** 2)

            if abs(pd) > lane_width:
                cost += 1000.0

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
    print("Initial obs:", obs)

    try:
        for step in range(1500):

            # Get obstacle positions
            obstacle_positions = []
            for obs_id in env.obstacle_ids:
                pos, _ = p.getBasePositionAndOrientation(obs_id)
                obstacle_positions.append((pos[0], pos[1]))

            # MPC action
            action = mpc_control(obs, obstacle_positions, env.dt)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if step % 25 == 0:
                print(f"step {step}, obs={obs}, action={action}")

            # Camera follow
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
