Enviornments and Test files guide:

Environments -

Multiple simulation environments were developed and used over the course of this project:

car_env.py – The initial static environment used for early MPC development and validation.

car_env_with_car.py – A static environment extended with cars as obstacles for testing obstacle avoidance and lane changes.

dynamic_env.py – The final dynamic environment with moving vehicles and a designated parking space; this is the primary environment used for the final experiments.

---

Test Files -

Several test scripts are included to reflect different stages and experiments in the MPC development:

test_car_env.py – Initial MPC implementation featuring obstacle avoidance, lane changes, and early parking logic.

test_car_park – Final MPC implementation focused on robust parking behavior.

test_frenet_mpc.py – Experimental MPC implementation using a Frenet-frame formulation.

