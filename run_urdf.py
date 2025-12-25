import pybullet as p
import pybullet_data
import time

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

p.loadURDF("plane.urdf")

p.loadURDF(
    "simple_car.urdf",
    basePosition=[0, 0, 0.1]
)

while True:
    p.stepSimulation()
    time.sleep(1.0 / 240.0)
