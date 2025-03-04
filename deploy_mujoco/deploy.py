import mujoco.glfw
import torch
from rl_sdk import *
from observation_buffer import *
import utils

import mujoco.viewer
import mujoco
import time
from threading import Thread
import threading

CSV_LOGGER = True
CONTACT_LOGGER = False
MOTOR_SENSOR_NUM = 3


class RL_Sim(RL):
    def __init__(self, robot_name="a1", config_file="config.yaml"):
        super().__init__()

        # member variables for RL_Sim
        # self.cmd_vel = [0, 0, 0]

        # read params from yaml
        self.robot_name = robot_name
        self.ReadYaml(config_file)
        for i in range(len(self.params.observations)):
            if self.params.observations[i] == "ang_vel":
                self.params.observations[i] = "ang_vel_world"
                # self.params.observations[i] = "ang_vel_body"

        # history
        if len(self.params.observations_history) != 0:
            self.history_obs_buf = ObservationBuffer(
                1, self.params.num_observations, len(self.params.observations_history)
            )

        self.cmd_vel = self.params.cmd_vel

        # init
        torch.set_grad_enabled(False)

        self.InitObservations()
        self.InitOutputs()
        self.InitControl()
        self.running_state = STATE.STATE_WAITING
        # self.running_state = STATE.STATE_RL_RUNNING

        # Load robot model
        self.locker = threading.Lock()
        self.m = mujoco.MjModel.from_xml_path(self.params.xml_path)
        self.d = mujoco.MjData(self.m)
        self.m.opt.timestep = self.params.dt
        self.num_motor = self.m.nu
        self.dim_motor_sensor = MOTOR_SENSOR_NUM * self.num_motor
        self.viewer = mujoco.viewer.launch_passive(
            self.m, self.d, key_callback=self.MujocoKeyCallback
        )
        self.motor_strength = [1.0] * 12

        # Check sensor
        # for i in range(self.dim_motor_sensor, self.m.nsensor):
        #     name = mujoco.mj_id2name(self.m, mujoco._enums.mjtObj.mjOBJ_SENSOR, i)
        #     if name == "imu_quat":
        #         self.have_imu_ = True
        #     if name == "frame_pos":
        #         self.have_frame_sensor_ = True

        # model
        self.model = torch.jit.load(self.params.policy_path)
        # 初始化轨迹绘制器
        self.trajectory_drawer0 = utils.TrajectoryDrawer(
            max_segments=1000, min_distance=0.02
        )
        self.trajectory_drawer1 = utils.TrajectoryDrawer(
            max_segments=1000, min_distance=0.02
        )
        self.trajectory_drawer2 = utils.TrajectoryDrawer(
            max_segments=1000, min_distance=0.02
        )
        self.trajectory_drawer3 = utils.TrajectoryDrawer(
            max_segments=1000, min_distance=0.02
        )
        FL_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "FL_touch")
        FR_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "FR_touch")
        RL_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "RL_touch")
        RR_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "RR_touch")
        self.id = [FL_id, FR_id, RL_id, RR_id]
        # others
        if CSV_LOGGER:
            self.CSVInit()
        if CONTACT_LOGGER:
            self.CONTACTInit()

        print(LOGGER.INFO + "RL_Sim start")

        # thread loops
        self.viewer_thread = Thread(target=self.PhysicsViewerThread)
        self.sim_thread = Thread(target=self.SimulationThread)
        self.viewer_thread.start()
        self.sim_thread.start()

    def __del__(self):
        print("\r\n" + LOGGER.INFO + "RL_Sim exit")

    def GetState(self):
        if self.params.framework == "isaacgym":
            self.robot_state.imu.quaternion[3] = self.d.sensordata[
                self.dim_motor_sensor + 0
            ]
            self.robot_state.imu.quaternion[0] = self.d.sensordata[
                self.dim_motor_sensor + 1
            ]
            self.robot_state.imu.quaternion[1] = self.d.sensordata[
                self.dim_motor_sensor + 2
            ]
            self.robot_state.imu.quaternion[2] = self.d.sensordata[
                self.dim_motor_sensor + 3
            ]
        elif self.params.framework == "isaacsim" or self.params.framework == "isaaclab":
            self.robot_state.imu.quaternion[0] = self.d.sensordata[
                self.dim_motor_sensor + 0
            ]
            self.robot_state.imu.quaternion[1] = self.d.sensordata[
                self.dim_motor_sensor + 1
            ]
            self.robot_state.imu.quaternion[2] = self.d.sensordata[
                self.dim_motor_sensor + 2
            ]
            self.robot_state.imu.quaternion[3] = self.d.sensordata[
                self.dim_motor_sensor + 3
            ]

        self.robot_state.imu.gyroscope[0] = self.d.sensordata[self.dim_motor_sensor + 4]
        self.robot_state.imu.gyroscope[1] = self.d.sensordata[self.dim_motor_sensor + 5]
        self.robot_state.imu.gyroscope[2] = self.d.sensordata[self.dim_motor_sensor + 6]

        self.robot_state.imu.accelerometer[0] = self.d.sensordata[
            self.dim_motor_sensor + 7
        ]
        self.robot_state.imu.accelerometer[1] = self.d.sensordata[
            self.dim_motor_sensor + 8
        ]
        self.robot_state.imu.accelerometer[2] = self.d.sensordata[
            self.dim_motor_sensor + 9
        ]
        for i in range(self.params.num_of_dofs):
            self.robot_state.motor_state.q[i] = self.d.sensordata[i]
            self.robot_state.motor_state.dq[i] = self.d.sensordata[i + self.num_motor]
            self.robot_state.motor_state.tau_est[i] = self.d.sensordata[
                i + 2 * self.num_motor
            ]
        self.robot_state.foot_force[0] = (
            int(self.d.sensordata[self.dim_motor_sensor + 16]) > 0
        )
        self.robot_state.foot_force[1] = (
            int(self.d.sensordata[self.dim_motor_sensor + 17]) > 0
        )
        self.robot_state.foot_force[2] = (
            int(self.d.sensordata[self.dim_motor_sensor + 18]) > 0
        )
        self.robot_state.foot_force[3] = (
            int(self.d.sensordata[self.dim_motor_sensor + 19]) > 0
        )

    def StateController(self, state, command):  # FSM
        # waiting
        if self.running_state == STATE.STATE_WAITING:
            for i in range(self.params.num_of_dofs):
                command.motor_command.q[i] = state.motor_state.q[i]
            if self.control.control_state == STATE.STATE_POS_GETUP:
                self.control.control_state = STATE.STATE_WAITING
                self.getup_percent = 0.0
                for i in range(self.params.num_of_dofs):
                    self.now_state.motor_state.q[i] = state.motor_state.q[i]
                    self.start_state.motor_state.q[i] = self.now_state.motor_state.q[i]
                self.running_state = STATE.STATE_POS_GETUP
                print("\r\n" + LOGGER.INFO + "Switching to STATE_POS_GETUP")

        # stand up (position control)
        elif self.running_state == STATE.STATE_POS_GETUP:
            if self.getup_percent < 1.0:
                self.getup_percent += 1 / 500.0
                self.getup_percent = min(self.getup_percent, 1.0)
                for i in range(self.params.num_of_dofs):
                    command.motor_command.q[i] = (
                        1 - self.getup_percent
                    ) * self.now_state.motor_state.q[
                        i
                    ] + self.getup_percent * self.params.default_dof_pos[
                        0
                    ][
                        i
                    ].item()
                    command.motor_command.dq[i] = 0
                    command.motor_command.kp[i] = self.params.fixed_kp[0][i].item()
                    command.motor_command.kd[i] = self.params.fixed_kd[0][i].item()
                    command.motor_command.tau[i] = 0
                print(
                    "\r" + LOGGER.INFO + f"Getting up {self.getup_percent * 100.0:.1f}",
                    end="",
                    flush=True,
                )

            if self.control.control_state == STATE.STATE_RL_INIT:
                self.control.control_state = STATE.STATE_WAITING
                self.running_state = STATE.STATE_RL_INIT
                print("\r\n" + LOGGER.INFO + "Switching to STATE_RL_INIT")

            elif self.control.control_state == STATE.STATE_POS_GETDOWN:
                self.control.control_state = STATE.STATE_WAITING
                self.getdown_percent = 0.0
                for i in range(self.params.num_of_dofs):
                    self.now_state.motor_state.q[i] = state.motor_state.q[i]
                self.running_state = STATE.STATE_POS_GETDOWN
                print("\r\n" + LOGGER.INFO + "Switching to STATE_POS_GETDOWN")

        # init obs and start rl loop
        elif self.running_state == STATE.STATE_RL_INIT:
            if self.getup_percent == 1:
                self.InitObservations()
                self.InitOutputs()
                self.InitControl()
                self.running_state = STATE.STATE_RL_RUNNING
                print("\r\n" + LOGGER.INFO + "Switching to STATE_RL_RUNNING")

        # rl loop
        if self.running_state == STATE.STATE_RL_RUNNING:
            print(
                "\r"
                + LOGGER.INFO
                + f"RL Controller x: {self.control.x:.1f} y: {self.control.y:.1f} yaw: {self.control.yaw:.1f}",
                end="",
                flush=True,
            )
            for i in range(self.params.num_of_dofs):
                command.motor_command.q[i] = self.output_dof_pos[0][i].item()
                command.motor_command.dq[i] = 0
                command.motor_command.kp[i] = self.params.rl_kp[0][i].item()
                command.motor_command.kd[i] = self.params.rl_kd[0][i].item()
                command.motor_command.tau[i] = 0

            if self.control.control_state == STATE.STATE_POS_GETDOWN:
                self.control.control_state = STATE.STATE_WAITING
                self.getdown_percent = 0.0
                for i in range(self.params.num_of_dofs):
                    self.now_state.motor_state.q[i] = state.motor_state.q[i]
                self.running_state = STATE.STATE_POS_GETDOWN
                print("\r\n" + LOGGER.INFO + "Switching to STATE_POS_GETDOWN")

            elif self.control.control_state == STATE.STATE_POS_GETUP:
                self.control.control_state = STATE.STATE_WAITING
                self.getup_percent = 0.0
                for i in range(self.params.num_of_dofs):
                    self.now_state.motor_state.q[i] = state.motor_state.q[i]
                self.running_state = STATE.STATE_POS_GETUP
                print("\r\n" + LOGGER.INFO + "Switching to STATE_POS_GETUP")

        # get down (position control)
        elif self.running_state == STATE.STATE_POS_GETDOWN:
            if self.getdown_percent < 1.0:
                self.getdown_percent += 1 / 500.0
                self.getdown_percent = min(1.0, self.getdown_percent)
                for i in range(self.params.num_of_dofs):
                    command.motor_command.q[i] = (
                        1 - self.getdown_percent
                    ) * self.now_state.motor_state.q[
                        i
                    ] + self.getdown_percent * self.start_state.motor_state.q[
                        i
                    ]
                    command.motor_command.dq[i] = 0
                    command.motor_command.kp[i] = self.params.fixed_kp[0][i].item()
                    command.motor_command.kd[i] = self.params.fixed_kd[0][i].item()
                    command.motor_command.tau[i] = 0
                print(
                    "\r"
                    + LOGGER.INFO
                    + f"Getting down {self.getdown_percent * 100.0:.1f}",
                    end="",
                    flush=True,
                )

            if self.getdown_percent == 1:
                self.InitObservations()
                self.InitOutputs()
                self.InitControl()
                self.running_state = STATE.STATE_WAITING
                print("\r\n" + LOGGER.INFO + "Switching to STATE_WAITING")

    def SetCommand(self, command):
        for i in range(self.num_motor):
            self.d.ctrl[i] = command.motor_command.tau[i] + self.motor_strength[i] * (
                command.motor_command.kp[i]
                * (command.motor_command.q[i] - self.d.sensordata[i])
                + command.motor_command.kd[i]
                * (command.motor_command.dq[i] - self.d.sensordata[i + self.num_motor])
            )

    def RobotControl(self):
        if self.control.control_state == STATE.STATE_RESET_SIMULATION:
            mujoco.mj_resetData(self.m, self.d)
            self.control.control_state = STATE.STATE_WAITING
        if self.control.control_state == STATE.STATE_TOGGLE_SIMULATION:
            print("\r\n" + LOGGER.INFO + "Simulation Start")
            self.simulation_running = not self.simulation_running
            self.control.control_state = STATE.STATE_WAITING

        if self.simulation_running:
            self.GetState()
            self.StateController(self.robot_state, self.robot_command)
            self.SetCommand(self.robot_command)

    # def get_gravity_orientation(self, quaternion):
    #     qw = quaternion[0]
    #     qx = quaternion[1]
    #     qy = quaternion[2]
    #     qz = quaternion[3]

    #     gravity_orientation = np.zeros(3)

    #     gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    #     gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    #     gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    #     return gravity_orientation

    def RunModel(self):
        if self.running_state == STATE.STATE_RL_RUNNING and self.simulation_running:
            if (
                self.params.framework == "isaacgym"
                or self.params.framework == "isaacsim"
            ):
                self.obs.lin_vel = torch.tensor(
                    [
                        [
                            self.d.sensordata[self.dim_motor_sensor + 13],
                            self.d.sensordata[self.dim_motor_sensor + 14],
                            self.d.sensordata[self.dim_motor_sensor + 15],
                        ]
                    ],
                    dtype=torch.float,
                )
                # self.obs.lin_vel = torch.tensor(
                #     [
                #         [
                #             self.d.qvel[0],
                #             self.d.qvel[1],
                #             self.d.qvel[2],
                #         ]
                #     ],
                #     dtype=torch.float,
                # )
                # self.obs.gravity_vec = torch.tensor(
                #     self.get_gravity_orientation(self.d.qpos[3:7]), dtype=torch.float
                # ).unsqueeze(0)

                self.obs.ang_vel = torch.tensor(
                    self.robot_state.imu.gyroscope, dtype=torch.float
                ).unsqueeze(0)
                self.obs.commands = torch.tensor(
                    [[self.cmd_vel[0], self.cmd_vel[1], self.cmd_vel[2]]],
                    dtype=torch.float,
                )
                # self.obs.commands = torch.tensor(
                #     [[self.control.x, self.control.y, self.control.yaw]],
                #     dtype=torch.float,
                # )
                self.obs.base_quat = torch.tensor(
                    self.robot_state.imu.quaternion, dtype=torch.float
                ).unsqueeze(0)
                self.obs.dof_pos = (
                    torch.tensor(self.robot_state.motor_state.q, dtype=torch.float)
                    .narrow(0, 0, self.params.num_of_dofs)
                    .unsqueeze(0)
                )
                self.obs.dof_vel = (
                    torch.tensor(self.robot_state.motor_state.dq, dtype=torch.float)
                    .narrow(0, 0, self.params.num_of_dofs)
                    .unsqueeze(0)
                )
            elif self.params.framework == "isaaclab":
                self.obs.lin_vel = torch.tensor(
                    [
                        [
                            self.d.sensordata[self.dim_motor_sensor + 13],
                            self.d.sensordata[self.dim_motor_sensor + 14],
                            self.d.sensordata[self.dim_motor_sensor + 15],
                        ]
                    ],
                    dtype=torch.float,
                )
                self.obs.ang_vel = torch.tensor(
                    self.robot_state.imu.gyroscope, dtype=torch.float
                ).unsqueeze(0)
                self.obs.commands = torch.tensor(
                    [[self.cmd_vel[0], self.cmd_vel[1], self.cmd_vel[2]]],
                    dtype=torch.float,
                )
                self.obs.dof_pos = (
                    torch.tensor(self.robot_state.motor_state.q, dtype=torch.float)
                    .narrow(0, 0, self.params.num_of_dofs)
                    .unsqueeze(0)
                )
                self.obs.dof_vel = (
                    torch.tensor(self.robot_state.motor_state.dq, dtype=torch.float)
                    .narrow(0, 0, self.params.num_of_dofs)
                    .unsqueeze(0)
                )
                self.obs.foot_force = torch.tensor(
                    self.robot_state.foot_force, dtype=torch.float
                ).unsqueeze(0)
                # self.obs.motor_strength = torch.tensor(
                #     self.motor_strength, dtype=torch.float
                # ).unsqueeze(0)
                # self.obs.rigid_object_properties = torch.tensor(
                #     [
                #         self.d.sensordata[self.dim_motor_sensor + 10],
                #         self.d.sensordata[self.dim_motor_sensor + 11],
                #         self.d.sensordata[self.dim_motor_sensor + 12],
                #         12.0,
                #         0.8,
                #         0.02,
                #         0.01,
                #     ],
                #     dtype=torch.float,
                # ).unsqueeze(0)
                # self.obs.height_map = torch.tensor(
                #     [0] * 187, dtype=torch.float
                # ).unsqueeze(0)

            clamped_actions = self.Forward()

            for i in self.params.hip_scale_reduction_indices:
                clamped_actions[0][i] *= self.params.hip_scale_reduction

            self.obs.actions = clamped_actions

            origin_output_dof_tau = self.ComputeTorques(self.obs.actions)

            self.output_dof_tau = torch.clamp(
                origin_output_dof_tau,
                -(self.params.torque_limits),
                self.params.torque_limits,
            )
            self.output_dof_pos = self.ComputePosition(self.obs.actions)

            if CSV_LOGGER:
                self.CSVLogger(
                    self.d.time,
                    self.d.sensordata[self.dim_motor_sensor + 13],
                    self.cmd_vel[0],
                    self.d.sensordata[self.dim_motor_sensor + 6],
                    self.cmd_vel[2],
                )
                # tau_est = torch.zeros((1, self.params.num_of_dofs))
                # for i in range(self.params.num_of_dofs):
                #     tau_est[0, i] = self.d.sensordata[i + 2 * self.num_motor]
                # self.CSVLogger(
                #     self.output_dof_tau,
                #     tau_est,
                #     self.obs.dof_pos,
                #     self.output_dof_pos,
                #     self.obs.dof_vel,
                # )
            if CONTACT_LOGGER:
                self.CONTACTLogger(self.d.time, self.robot_state.foot_force)

    def Forward(self):
        torch.set_grad_enabled(False)
        clamped_obs = self.ComputeObservation()
        if len(self.params.observations_history) != 0:
            self.history_obs_buf.insert(clamped_obs)
            history_obs = self.history_obs_buf.get_obs_vec(
                self.params.observations_history
            )
            actions = self.model.forward(history_obs)
        else:
            actions = self.model.forward(clamped_obs)
        if (
            self.params.clip_actions_lower is not None
            and self.params.clip_actions_upper is not None
        ):
            return torch.clamp(
                actions, self.params.clip_actions_lower, self.params.clip_actions_upper
            )
        else:
            return actions

    def SimulationThread(self):
        counter = 0
        change = 0
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while self.viewer.is_running() and time.time() - start < self.params.total_time:
            step_start = time.perf_counter()
            simTime = self.d.time
            if simTime > 10:
                if change == 0:
                    change = 1
                    self.cmd_vel[0] -= 0.1
                    self.cmd_vel[2] += 0.1
            # if simTime > 10:
            #     if change == 1:
            #         self.cmd_vel[0] += 0.02
            #         self.cmd_vel[2] = 0
            if simTime > 14:
                # print("fault")
                self.motor_strength[2] = 0.6
                # self.motor_strength[5] = 0.2
                # self.motor_strength[8] = 0.2
                # self.motor_strength[11] = 0.2

            self.locker.acquire()
            # Apply state update here.
            self.RobotControl()
            # physics update
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(self.m, self.d)

            self.locker.release()

            counter += 1
            if counter % self.params.decimation == 0:
                # Apply control signal here.
                self.RunModel()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = self.m.opt.timestep - (
                time.perf_counter() - step_start
            )
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    def PhysicsViewerThread(self):
        while self.viewer.is_running():
            self.locker.acquire()
            self.viewer.user_scn.ngeom = 0
            self.trajectory_drawer0.add_point(self.d.site_xpos[self.id[0]].copy())
            self.trajectory_drawer0.draw_trajectory(
                self.viewer,
                color=[1, 0, 0, 1],  # 绿色轨迹
                width=0.002,
            )
            self.trajectory_drawer1.add_point(self.d.site_xpos[self.id[1]].copy())
            self.trajectory_drawer1.draw_trajectory(
                self.viewer,
                color=[0, 1, 0, 1],  # 绿色轨迹
                width=0.002,
            )
            self.trajectory_drawer2.add_point(self.d.site_xpos[self.id[2]].copy())
            self.trajectory_drawer2.draw_trajectory(
                self.viewer,
                color=[0, 0, 1, 1],  # 绿色轨迹
                width=0.002,
            )
            self.trajectory_drawer3.add_point(self.d.site_xpos[self.id[3]].copy())
            self.trajectory_drawer3.draw_trajectory(
                self.viewer,
                color=[0.5569, 0.8118, 0.7882, 1],  # 绿色轨迹
                width=0.002,
            )
            self.viewer.sync()
            self.locker.release()
            time.sleep(self.params.viewer_dt)

    def MujocoKeyCallback(self, key):
        glfw = mujoco.glfw.glfw
        if key == glfw.KEY_1:
            self.control.control_state = STATE.STATE_POS_GETUP
        elif key == glfw.KEY_P:
            self.control.control_state = STATE.STATE_RL_INIT
        elif key == glfw.KEY_4:
            self.control.control_state = STATE.STATE_POS_GETDOWN
        elif key == glfw.KEY_W:
            self.control.x += 0.1
        elif key == glfw.KEY_S:
            self.control.x -= 0.1
        elif key == glfw.KEY_A:
            self.control.yaw += 0.1
        elif key == glfw.KEY_D:
            self.control.yaw -= 0.1
        elif key == glfw.KEY_J:
            self.control.y += 0.1
        elif key == glfw.KEY_L:
            self.control.y -= 0.1
        elif key == glfw.KEY_R:
            self.control.control_state = STATE.STATE_RESET_SIMULATION
        elif key == glfw.KEY_ENTER:
            self.control.control_state = STATE.STATE_TOGGLE_SIMULATION
        elif key == glfw.KEY_SPACE:
            self.control.x = 0
            self.control.y = 0
            self.control.yaw = 0
