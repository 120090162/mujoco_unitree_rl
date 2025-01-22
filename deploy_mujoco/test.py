import time
import mujoco
import mujoco.viewer
from threading import Thread
import threading


locker = threading.Lock()

mj_model = mujoco.MjModel.from_xml_path(
    "/home/joshua/WORK/test_genesis/mujoco_unitree_rl/resources/unitree_a1/scene.xml"
)
mj_data = mujoco.MjData(mj_model)

viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

mj_model.opt.timestep = 0.005
num_motor_ = mj_model.nu
dim_motor_sensor_ = 3 * num_motor_

time.sleep(0.2)


def SimulationThread():
    global mj_data, mj_model

    while viewer.is_running():
        step_start = time.perf_counter()

        locker.acquire()

        mujoco.mj_step(mj_model, mj_data)

        locker.release()

        time_until_next_step = mj_model.opt.timestep - (
            time.perf_counter() - step_start
        )
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)


def PhysicsViewerThread():
    while viewer.is_running():
        locker.acquire()
        viewer.sync()
        locker.release()
        time.sleep(0.02)


if __name__ == "__main__":
    viewer_thread = Thread(target=PhysicsViewerThread)
    sim_thread = Thread(target=SimulationThread)

    viewer_thread.start()
    sim_thread.start()
