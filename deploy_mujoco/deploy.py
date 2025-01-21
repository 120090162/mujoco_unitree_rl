import torch
from rl_sdk import *
from observation_buffer import *

import mujoco.viewer
import mujoco
import time

CSV_LOGGER = False
MOTOR_SENSOR_NUM = 3


class RL_Sim(RL):
    def __init__(self, robot_name="a1", config_file="config.yaml"):
        super().__init__()

        # member variables for RL_Sim
        self.cmd_vel = [0, 0, 0]

        # read params from yaml
        self.robot_name = robot_name
        self.ReadYaml(config_file)
        for i in range(len(self.params.observations)):
            if self.params.observations[i] == "ang_vel":
                self.params.observations[i] = "ang_vel_world"

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

        # Load robot model
        self.m = mujoco.MjModel.from_xml_path(self.params.xml_path)
        self.d = mujoco.MjData(self.m)
        self.m.opt.timestep = self.params.dt
        self.num_motor = self.m.nu
        self.dim_motor_sensor = MOTOR_SENSOR_NUM * self.num_motor

        # Check sensor
        for i in range(self.dim_motor_sensor, self.m.nsensor):
            name = mujoco.mj_id2name(self.m, mujoco._enums.mjtObj.mjOBJ_SENSOR, i)
            if name == "imu_quat":
                self.have_imu_ = True
            if name == "frame_pos":
                self.have_frame_sensor_ = True

        # model
        self.model = torch.jit.load(self.params.policy_path)
        # others
        if CSV_LOGGER:
            self.CSVInit()

        print(LOGGER.INFO + "RL_Sim start")

    def __del__(self):
        print("\n" + LOGGER.INFO + "RL_Sim exit")

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
        elif self.params.framework == "isaacsim":
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

    def StateController(self, command):
        # rl loop
        print(
            "\r"
            + LOGGER.INFO
            + f"RL Controller x: {self.cmd_vel[0]:.1f} y: {self.cmd_vel[1]:.1f} yaw: {self.cmd_vel[2]:.1f}",
            end="",
            flush=True,
        )
        for i in range(self.params.num_of_dofs):
            command.motor_command.q[i] = self.output_dof_pos[0][i].item()
            command.motor_command.dq[i] = 0
            command.motor_command.kp[i] = self.params.rl_kp[0][i].item()
            command.motor_command.kd[i] = self.params.rl_kd[0][i].item()
            command.motor_command.tau[i] = 0

    def SetCommand(self, command):
        for i in range(self.num_motor):
            self.d.ctrl[i] = (
                command.motor_command.tau[i]
                + command.motor_command.kp[i]
                * (command.motor_command.q[i] - self.d.sensordata[i])
                + command.motor_command.kd[i]
                * (command.motor_command.dq[i] - self.d.sensordata[i + self.num_motor])
            )

    def RunModel(self):
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
            [[self.cmd_vel[0], self.cmd_vel[1], self.cmd_vel[2]]], dtype=torch.float
        )
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
            tau_est = torch.zeros((1, self.params.num_of_dofs))
            for i in range(self.params.num_of_dofs):
                tau_est[0, i] = self.d.sensordata[i + 2 * self.num_motor]
            self.CSVLogger(
                self.output_dof_tau,
                tau_est,
                self.obs.dof_pos,
                self.output_dof_pos,
                self.obs.dof_vel,
            )

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

    def main_loop(self):
        counter = 0

        with mujoco.viewer.launch_passive(self.m, self.d) as viewer:
            # Close the viewer automatically after simulation_duration wall-seconds.
            start = time.time()
            while viewer.is_running() and time.time() - start < self.params.total_time:
                step_start = time.time()
                # Apply state update here.
                self.GetState()
                self.StateController(self.robot_command)
                self.SetCommand(self.robot_command)

                # physics update
                # mj_step can be replaced with code that also evaluates
                # a policy and applies a control signal before stepping the physics.
                mujoco.mj_step(self.m, self.d)

                counter += 1
                if counter % self.params.decimation == 0:
                    # Apply control signal here.
                    self.RunModel()

                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = self.m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
