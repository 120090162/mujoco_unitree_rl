from datetime import datetime
import torch
import yaml
import csv

LEGGED_GYM_ROOT_DIR = "/home/joshua/WORK/test_genesis/mujoco_unitree_rl"


class LOGGER:
    INFO = "\033[0;37m[INFO]\033[0m "
    WARNING = "\033[0;33m[WARNING]\033[0m "
    ERROR = "\033[0;31m[ERROR]\033[0m "
    DEBUG = "\033[0;32m[DEBUG]\033[0m "


class RobotCommand:
    def __init__(self):
        self.motor_command = self.MotorCommand()

    class MotorCommand:
        def __init__(self):
            self.q = [0.0] * 32
            self.dq = [0.0] * 32
            self.tau = [0.0] * 32
            self.kp = [0.0] * 32
            self.kd = [0.0] * 32


class RobotState:
    def __init__(self):
        self.imu = self.IMU()
        self.motor_state = self.MotorState()

    class IMU:
        def __init__(self):
            self.quaternion = [1.0, 0.0, 0.0, 0.0]  # w, x, y, z
            self.gyroscope = [0.0, 0.0, 0.0]
            self.accelerometer = [0.0, 0.0, 0.0]

    class MotorState:
        def __init__(self):
            self.q = [0.0] * 32
            self.dq = [0.0] * 32
            self.ddq = [0.0] * 32
            self.tau_est = [0.0] * 32
            self.cur = [0.0] * 32


class ModelParams:
    def __init__(self):
        self.policy_path = None
        self.xml_path = None
        self.model_name = None
        self.framework = None
        self.total_time = None
        self.dt = None
        self.decimation = None
        self.num_observations = None
        self.observations = None
        self.observations_history = None
        self.damping = None
        self.stiffness = None
        self.action_scale = None
        self.hip_scale_reduction = None
        self.hip_scale_reduction_indices = None
        self.clip_actions_upper = None
        self.clip_actions_lower = None
        self.num_of_dofs = None
        self.lin_vel_scale = None
        self.ang_vel_scale = None
        self.dof_pos_scale = None
        self.dof_vel_scale = None
        self.clip_obs = None
        self.torque_limits = None
        self.rl_kd = None
        self.rl_kp = None
        self.fixed_kp = None
        self.fixed_kd = None
        self.commands_scale = None
        self.default_dof_pos = None
        self.joint_controller_names = None
        self.cmd_vel = None


class Observations:
    def __init__(self):
        self.lin_vel = None
        self.ang_vel = None
        self.gravity_vec = None
        self.commands = None
        self.base_quat = None
        self.dof_pos = None
        self.dof_vel = None
        self.actions = None


class RL:
    # Static variables
    start_state = RobotState()
    now_state = RobotState()
    getup_percent = 0.0
    getdown_percent = 0.0

    def __init__(self):
        ### public in cpp ###
        self.params = ModelParams()
        self.obs = Observations()

        self.robot_state = RobotState()
        self.robot_command = RobotCommand()

        # others
        self.robot_name = ""

        ### protected in cpp ###
        # rl module
        self.model = None
        self.walk_model = None
        self.stand_model = None

        # output buffer
        self.output_dof_tau = torch.zeros(1, 32)
        self.output_dof_pos = torch.zeros(1, 32)

    def ComputeObservation(self):
        obs_list = []
        for observation in self.params.observations:
            """
            The first argument of the QuatRotateInverse function is the quaternion representing the robot's orientation, and the second argument is in the world coordinate system. The function outputs the value of the second argument in the body coordinate system.
            In IsaacGym, the coordinate system for angular velocity is in the world coordinate system. During training, the angular velocity in the observation uses QuatRotateInverse to transform the coordinate system to the body coordinate system.
            In Gazebo, the coordinate system for angular velocity is also in the world coordinate system, so QuatRotateInverse is needed to transform the coordinate system to the body coordinate system.
            In some real robots like Unitree, if the coordinate system for the angular velocity is already in the body coordinate system, no transformation is necessary.
            Forgetting to perform the transformation or performing it multiple times may cause controller crashes when the rotation reaches 180 degrees.
            """
            if observation == "lin_vel":
                obs_list.append(self.obs.lin_vel * self.params.lin_vel_scale)
            elif observation == "ang_vel_body":
                obs_list.append(self.obs.ang_vel * self.params.ang_vel_scale)
            elif observation == "ang_vel_world":
                obs_list.append(
                    self.QuatRotateInverse(
                        self.obs.base_quat, self.obs.ang_vel, self.params.framework
                    )
                    * self.params.ang_vel_scale
                )
            elif observation == "gravity_vec":
                obs_list.append(
                    self.QuatRotateInverse(
                        self.obs.base_quat, self.obs.gravity_vec, self.params.framework
                    )
                )
            elif observation == "commands":
                obs_list.append(self.obs.commands * self.params.commands_scale)
            elif observation == "dof_pos":
                obs_list.append(
                    (self.obs.dof_pos - self.params.default_dof_pos)
                    * self.params.dof_pos_scale
                )
            elif observation == "dof_vel":
                obs_list.append(self.obs.dof_vel * self.params.dof_vel_scale)
            elif observation == "actions":
                obs_list.append(self.obs.actions)
        obs = torch.cat(obs_list, dim=-1)
        clamped_obs = torch.clamp(obs, -self.params.clip_obs, self.params.clip_obs)
        return clamped_obs

    def InitObservations(self):
        self.obs.lin_vel = torch.zeros(1, 3, dtype=torch.float)
        self.obs.ang_vel = torch.zeros(1, 3, dtype=torch.float)
        self.obs.gravity_vec = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float)
        self.obs.commands = torch.zeros(1, 3, dtype=torch.float)
        self.obs.base_quat = torch.zeros(1, 4, dtype=torch.float)
        self.obs.dof_pos = self.params.default_dof_pos
        self.obs.dof_vel = torch.zeros(1, self.params.num_of_dofs, dtype=torch.float)
        self.obs.actions = torch.zeros(1, self.params.num_of_dofs, dtype=torch.float)

    def InitOutputs(self):
        self.output_dof_tau = torch.zeros(1, self.params.num_of_dofs, dtype=torch.float)
        self.output_dof_pos = self.params.default_dof_pos

    def ComputeTorques(self, actions):
        actions_scaled = actions * self.params.action_scale
        output_dof_tau = (
            self.params.rl_kp
            * (actions_scaled + self.params.default_dof_pos - self.obs.dof_pos)
            - self.params.rl_kd * self.obs.dof_vel
        )
        return output_dof_tau

    def ComputePosition(self, actions):
        actions_scaled = actions * self.params.action_scale
        return actions_scaled + self.params.default_dof_pos

    def QuatRotateInverse(self, q, v, framework):
        if framework == "isaacsim":
            q_w = q[:, 0]
            q_vec = q[:, 1:4]
        elif framework == "isaacgym":
            q_w = q[:, 3]
            q_vec = q[:, 0:3]
        shape = q.shape
        a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
        b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
        c = (
            q_vec
            * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1)
            * 2.0
        )
        return a - b + c

    def ReadVectorFromYaml(self, values, framework, rows, cols):
        if framework == "isaacsim":
            transposed_values = [0] * cols * rows
            for r in range(rows):
                for c in range(cols):
                    transposed_values[c * rows + r] = values[r * cols + c]
            return transposed_values
        elif framework == "isaacgym":
            return values
        else:
            raise ValueError(f"Unsupported framework: {framework}")

    def ReadYaml(self, config_file):
        config_path = f"{LEGGED_GYM_ROOT_DIR}/deploy_mujoco/configs/{config_file}"
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
        except FileNotFoundError as e:
            print(LOGGER.ERROR + f"The file '{config_path}' does not exist")
            return
        self.params.policy_path = config["policy_path"].replace(
            "{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR
        )
        self.params.xml_path = config["xml_path"].replace(
            "{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR
        )
        self.params.model_name = config["model_name"]
        self.params.policy_path = self.params.policy_path.replace(
            "{model_name}", self.params.model_name
        )
        self.params.framework = config["framework"]
        rows = config["rows"]
        cols = config["cols"]
        self.params.total_time = config["simulation_duration"]
        self.params.dt = config["simulation_dt"]
        self.params.decimation = config["control_decimation"]
        self.params.num_observations = config["num_obs"]
        self.params.observations = config["observations"]
        self.params.observations_history = config["observations_history"]
        self.params.clip_obs = config["clip_obs"]
        self.params.action_scale = config["action_scale"]
        self.params.hip_scale_reduction = config["hip_scale_reduction"]
        self.params.hip_scale_reduction_indices = config["hip_scale_reduction_indices"]
        if (
            config["clip_actions_lower"] is None
            and config["clip_actions_upper"] is None
        ):
            self.params.clip_actions_upper = None
            self.params.clip_actions_lower = None
        else:
            self.params.clip_actions_upper = torch.tensor(
                self.ReadVectorFromYaml(
                    config["clip_actions_upper"], self.params.framework, rows, cols
                )
            ).view(1, -1)
            self.params.clip_actions_lower = torch.tensor(
                self.ReadVectorFromYaml(
                    config["clip_actions_lower"], self.params.framework, rows, cols
                )
            ).view(1, -1)
        self.params.num_of_dofs = config["num_actions"]
        self.params.lin_vel_scale = config["lin_vel_scale"]
        self.params.ang_vel_scale = config["ang_vel_scale"]
        self.params.dof_pos_scale = config["dof_pos_scale"]
        self.params.dof_vel_scale = config["dof_vel_scale"]
        self.params.commands_scale = torch.tensor(
            [
                self.params.lin_vel_scale,  # x
                self.params.lin_vel_scale,  # y
                self.params.ang_vel_scale,  # yaw
            ]
        )
        self.params.rl_kp = torch.tensor(
            self.ReadVectorFromYaml(config["rl_kp"], self.params.framework, rows, cols)
        ).view(1, -1)
        self.params.rl_kd = torch.tensor(
            self.ReadVectorFromYaml(config["rl_kd"], self.params.framework, rows, cols)
        ).view(1, -1)
        self.params.fixed_kp = torch.tensor(
            self.ReadVectorFromYaml(
                config["fixed_kp"], self.params.framework, rows, cols
            )
        ).view(1, -1)
        self.params.fixed_kd = torch.tensor(
            self.ReadVectorFromYaml(
                config["fixed_kd"], self.params.framework, rows, cols
            )
        ).view(1, -1)
        self.params.torque_limits = torch.tensor(
            self.ReadVectorFromYaml(
                config["torque_limits"], self.params.framework, rows, cols
            )
        ).view(1, -1)
        self.params.default_dof_pos = torch.tensor(
            self.ReadVectorFromYaml(
                config["default_dof_pos"], self.params.framework, rows, cols
            )
        ).view(1, -1)
        self.params.joint_controller_names = self.ReadVectorFromYaml(
            config["joint_controller_names"], self.params.framework, rows, cols
        )

        self.params.cmd_vel = config["cmd_init"]

    def CSVInit(self):
        self.csv_filename = self.params.policy_path
        self.csv_filename = self.csv_filename.replace("policy.pt", "csv")

        # Uncomment these lines if need timestamp for file name
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d%H%M%S")
        self.csv_filename += (
            f"/{self.params.framework}_{self.params.model_name}_{timestamp}"
        )

        self.csv_filename += ".csv"

        with open(self.csv_filename, "w", newline="") as file:
            writer = csv.writer(file)

            header = []
            header += [f"tau_cal_{i}" for i in range(12)]
            header += [f"tau_est_{i}" for i in range(12)]
            header += [f"joint_pos_{i}" for i in range(12)]
            header += [f"joint_pos_target_{i}" for i in range(12)]
            header += [f"joint_vel_{i}" for i in range(12)]

            writer.writerow(header)

    def CSVLogger(self, torque, tau_est, joint_pos, joint_pos_target, joint_vel):
        with open(self.csv_filename, "a", newline="") as file:
            writer = csv.writer(file)

            row = []
            row += [torque[0][i].item() for i in range(12)]
            row += [tau_est[0][i].item() for i in range(12)]
            row += [joint_pos[0][i].item() for i in range(12)]
            row += [joint_pos_target[0][i].item() for i in range(12)]
            row += [joint_vel[0][i].item() for i in range(12)]

            writer.writerow(row)
