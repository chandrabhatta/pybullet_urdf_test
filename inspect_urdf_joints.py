import os
import pybullet as p
import pybullet_data
import time

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

p.loadURDF("plane.urdf")

# Absolute path to URDF
script_dir = os.path.dirname(os.path.abspath(__file__))
urdf_path = os.path.join(script_dir, "simple_car.urdf")

print("Loading URDF from:", urdf_path)

robot_id = p.loadURDF(
    urdf_path,
    basePosition=[0, 0, 0.1],
    useFixedBase=False
)

for i in range(p.getNumJoints(robot_id)):
    print(i, p.getJointInfo(robot_id, i)[1].decode())

while True:
    p.stepSimulation()
    time.sleep(1.0 / 240.0)

