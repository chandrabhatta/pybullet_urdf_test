from car_env_with_car import SimpleCarEnv
import numpy as np
import time
import pybullet as p


# ==========================================================
# MPC CONTROLLER (STABLE + LANE-BOUND + SAFE)
# ==========================================================
def mpc_control(state, obstacle_positions, dt):
    x, y, yaw = state

    # Frenet state
    s = x
    d = y
    psi = yaw

    L = 2.5
    horizon = 18

    lane_width = 3.5
    lane_centers = [-lane_width / 2, lane_width / 2]

    current_lane = min(lane_centers, key=lambda ly: abs(d - ly))
    target_lane = current_lane

    # =============================
    # OBSTACLE DETECTION
    # =============================
    LOOKAHEAD = 10.0
    min_s_dist = np.inf

    for ox, oy in obstacle_positions:
        if abs(oy - current_lane) < 0.6 and ox > s:
            min_s_dist = min(min_s_dist, ox - s)

    obstacle_ahead = min_s_dist < LOOKAHEAD

    # =============================
    # LANE CHANGE PLAN
    # =============================
    lane_change_active = obstacle_ahead

    if lane_change_active:
        target_lane = (
            lane_centers[0]
            if current_lane == lane_centers[1]
            else lane_centers[1]
        )

    # =============================
    # LANE CHANGE TRAJECTORY
    # =============================
    s0 = s
    LC_LENGTH = 12.0   # <<< THIS IS CRITICAL

    def d_reference(ps):
        if not lane_change_active:
            return current_lane

        ds = np.clip(ps - s0, 0.0, LC_LENGTH)
        alpha = ds / LC_LENGTH

        return (
            current_lane
            + 0.5 * (target_lane - current_lane)
            * (1 - np.cos(np.pi * alpha))
        )

    # =============================
    # SPEED POLICY
    # =============================
    target_speed = 1.8
    if min_s_dist < 6.0:
        target_speed = 1.2
    if abs(d - target_lane) > 0.6:
        target_speed = min(target_speed, 1.0)

    # =============================
    # MPC OPTIMIZATION
    # =============================
    steer_candidates = np.linspace(-0.35, 0.35, 9)
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

            cost += 30.0 * (pd - d_ref) ** 2
            cost += 4.0 * (ppsi ** 2)
            cost += 2.0 * (steer ** 2)

            if abs(pd) > lane_width:
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
