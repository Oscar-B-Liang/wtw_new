import isaacgym

assert isaacgym
import matplotlib.pyplot as plt
import torch
from tqdm import trange

from go1_gym.envs import *
from go1_gym.envs.base.legged_robot_config import Cfg
from go1_gym.envs.go1.go1_config import config_go1
from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv


def run_env(render=False, headless=False):
    # prepare environment
    config_go1(Cfg)

    Cfg.commands.num_lin_vel_bins = 30
    Cfg.commands.num_ang_vel_bins = 30
    Cfg.curriculum_thresholds.tracking_ang_vel = 0.7
    Cfg.curriculum_thresholds.tracking_lin_vel = 0.8
    Cfg.curriculum_thresholds.tracking_contacts_shaped_vel = 0.90
    Cfg.curriculum_thresholds.tracking_contacts_shaped_force = 0.90

    Cfg.commands.distributional_commands = True

    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = True
    Cfg.control.control_type = "actuator_net"

    Cfg.domain_rand.randomize_rigids_after_start = False
    Cfg.env.priv_observe_motion = False
    Cfg.env.priv_observe_gravity_transformed_motion = False
    Cfg.domain_rand.randomize_friction_indep = False
    Cfg.env.priv_observe_friction_indep = False
    Cfg.domain_rand.randomize_friction = True
    Cfg.env.priv_observe_friction = True
    Cfg.domain_rand.friction_range = [0.1, 3.0]
    Cfg.domain_rand.randomize_restitution = True
    Cfg.env.priv_observe_restitution = True
    Cfg.domain_rand.restitution_range = [0.0, 0.4]
    Cfg.domain_rand.randomize_base_mass = True
    Cfg.env.priv_observe_base_mass = False
    Cfg.domain_rand.added_mass_range = [-1.0, 3.0]
    Cfg.domain_rand.randomize_gravity = True
    Cfg.domain_rand.gravity_range = [-1.0, 1.0]
    Cfg.domain_rand.gravity_rand_interval_s = 8.0
    Cfg.domain_rand.gravity_impulse_duration = 0.99
    Cfg.env.priv_observe_gravity = False
    Cfg.domain_rand.randomize_com_displacement = False
    Cfg.domain_rand.com_displacement_range = [-0.15, 0.15]
    Cfg.env.priv_observe_com_displacement = False
    Cfg.domain_rand.randomize_ground_friction = True
    Cfg.env.priv_observe_ground_friction = False
    Cfg.env.priv_observe_ground_friction_per_foot = False
    Cfg.domain_rand.ground_friction_range = [0.0, 0.0]
    Cfg.domain_rand.randomize_motor_strength = True
    Cfg.domain_rand.motor_strength_range = [0.9, 1.1]
    Cfg.env.priv_observe_motor_strength = False
    Cfg.domain_rand.randomize_motor_offset = True
    Cfg.domain_rand.motor_offset_range = [-0.02, 0.02]
    Cfg.env.priv_observe_motor_offset = False
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.env.priv_observe_Kp_factor = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.env.priv_observe_Kd_factor = False
    Cfg.env.priv_observe_body_velocity = False
    Cfg.env.priv_observe_body_height = False
    Cfg.env.priv_observe_desired_contact_states = False
    Cfg.env.priv_observe_contact_forces = False
    Cfg.env.priv_observe_foot_displacement = False
    Cfg.env.priv_observe_gravity_transformed_foot_displacement = False

    Cfg.env.num_privileged_obs = 2
    Cfg.env.num_observation_history = 30

    Cfg.domain_rand.rand_interval_s = 4
    Cfg.commands.num_commands = 15
    Cfg.env.observe_two_prev_actions = True
    Cfg.env.observe_yaw = False
    Cfg.env.num_observations = 70
    Cfg.env.num_scalar_observations = 70
    Cfg.env.observe_gait_commands = True
    Cfg.env.observe_timing_parameter = False
    Cfg.env.observe_clock_inputs = True

    Cfg.domain_rand.tile_height_range = [-0.0, 0.0]
    Cfg.domain_rand.tile_height_curriculum = False
    Cfg.domain_rand.tile_height_update_interval = 1000000
    Cfg.domain_rand.tile_height_curriculum_step = 0.01
    Cfg.terrain.border_size = 0.0
    Cfg.terrain.mesh_type = "trimesh"
    Cfg.terrain.num_cols = 30
    Cfg.terrain.num_rows = 30
    Cfg.terrain.terrain_width = 5.0
    Cfg.terrain.terrain_length = 5.0
    Cfg.terrain.x_init_range = 0.2
    Cfg.terrain.y_init_range = 0.2
    Cfg.terrain.teleport_thresh = 0.3
    Cfg.terrain.teleport_robots = False
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 4
    Cfg.terrain.horizontal_scale = 0.10
    Cfg.rewards.use_terminal_foot_height = False
    Cfg.rewards.use_terminal_body_height = True
    Cfg.rewards.terminal_body_height = 0.05
    Cfg.rewards.use_terminal_roll_pitch = True
    Cfg.rewards.terminal_body_ori = 1.6

    Cfg.commands.resampling_time = 10

    Cfg.rewards.base_height_target = 0.30
    Cfg.rewards.kappa_gait_probs = 0.07
    Cfg.rewards.gait_force_sigma = 100.
    Cfg.rewards.gait_vel_sigma = 10.

    Cfg.rewards.reward_container_name = "CoRLRewards"
    Cfg.rewards.only_positive_rewards = False
    Cfg.rewards.only_positive_rewards_ji22_style = True
    Cfg.rewards.sigma_rew_neg = 0.02

    Cfg.env.commands_mask = args.mask
    if Cfg.env.commands_mask == "del":
      Cfg.env.num_observations = 70 - 16
      Cfg.env.num_scalar_observations = 70 - 16
      Cfg.env.observe_clock_inputs = False
      Cfg.env.observe_gait_commands = False

    # Task Rewards.
    Cfg.reward_scales.tracking_lin_vel = 1.0
    Cfg.reward_scales.tracking_ang_vel = 0.5

    # Augmented Auxiliary Rewards.
    Cfg.reward_scales.tracking_contacts_shaped_force = 0.0
    Cfg.reward_scales.tracking_contacts_shaped_vel = 0.0
    Cfg.reward_scales.jump = 0.0
    Cfg.reward_scales.orientation_control = 0.0
    Cfg.reward_scales.raibert_heuristic = 0.0
    Cfg.reward_scales.feet_clearance_cmd_linear = 0.0

    # Fixed Auxiliary Rewards.
    Cfg.reward_scales.lin_vel_z = -0.02
    Cfg.reward_scales.ang_vel_xy = -0.001
    Cfg.reward_scales.feet_slip = -0.04
    Cfg.reward_scales.collision = -5.0
    Cfg.reward_scales.dof_pos_limits = -10.0
    Cfg.reward_scales.torques = -0.0001
    Cfg.reward_scales.dof_vel = -1e-4
    Cfg.reward_scales.dof_acc = -2.5e-7
    Cfg.reward_scales.action_smoothness_1 = -0.1
    Cfg.reward_scales.action_smoothness_2 = -0.1

    # Rewards used in legged gym, but unparticipated here.
    Cfg.reward_scales.feet_air_time = 0.0
    Cfg.reward_scales.action_rate = -0.01

    # Unparticipated Rewards.
    Cfg.reward_scales.dof_pos = -0.0
    Cfg.reward_scales.feet_impact_vel = -0.0
    Cfg.reward_scales.orientation = 0.0
    Cfg.reward_scales.feet_contact_forces = 0.0

    # Uncorresponded Rewards.
    Cfg.reward_scales.base_height = 0.0
    Cfg.reward_scales.estimation_bonus = 0.0
    Cfg.reward_scales.feet_clearance = -0.0
    Cfg.reward_scales.feet_clearance_cmd = -0.0
    Cfg.reward_scales.tracking_stance_width = -0.0
    Cfg.reward_scales.tracking_stance_length = -0.0
    Cfg.reward_scales.hop_symmetry = 0.0
    Cfg.reward_scales.feet_stumble = 0.0
    Cfg.reward_scales.stand_still = 0.0
    Cfg.reward_scales.tracking_lin_vel_lat = 0.0
    Cfg.reward_scales.tracking_lin_vel_long = 0.0
    Cfg.reward_scales.tracking_contacts = 0.0
    Cfg.reward_scales.tracking_contacts_shaped = 0.0
    Cfg.reward_scales.energy = 0.0
    Cfg.reward_scales.energy_expenditure = 0.0
    Cfg.reward_scales.survival = 0.0
    Cfg.reward_scales.base_motion = 0.0

    Cfg.commands.lin_vel_x = [-1.0, 1.0]
    Cfg.commands.lin_vel_y = [-0.6, 0.6]
    Cfg.commands.ang_vel_yaw = [-1.0, 1.0]

    Cfg.commands.body_height_cmd = [-0.25, 0.15]
    Cfg.commands.gait_frequency_cmd_range = [2.0, 4.0]
    Cfg.commands.gait_phase_cmd_range = [0.0, 1.0]
    Cfg.commands.gait_offset_cmd_range = [0.0, 1.0]
    Cfg.commands.gait_bound_cmd_range = [0.0, 1.0]
    Cfg.commands.gait_duration_cmd_range = [0.5, 0.5]
    Cfg.commands.footswing_height_range = [0.03, 0.35]
    Cfg.commands.body_pitch_range = [-0.4, 0.4]
    Cfg.commands.body_roll_range = [-0.0, 0.0]
    Cfg.commands.stance_width_range = [0.10, 0.45]
    Cfg.commands.stance_length_range = [0.35, 0.45]

    Cfg.commands.limit_vel_x = [-5.0, 5.0]
    Cfg.commands.limit_vel_y = [-0.6, 0.6]
    Cfg.commands.limit_vel_yaw = [-5.0, 5.0]

    Cfg.commands.limit_body_height = [-0.25, 0.15]
    Cfg.commands.limit_gait_frequency = [2.0, 4.0]
    Cfg.commands.limit_gait_phase = [0.0, 1.0]
    Cfg.commands.limit_gait_offset = [0.0, 1.0]
    Cfg.commands.limit_gait_bound = [0.0, 1.0]
    Cfg.commands.limit_gait_duration = [0.5, 0.5]
    Cfg.commands.limit_footswing_height = [0.03, 0.35]
    Cfg.commands.limit_body_pitch = [-0.4, 0.4]
    Cfg.commands.limit_body_roll = [-0.0, 0.0]
    Cfg.commands.limit_stance_width = [0.10, 0.45]
    Cfg.commands.limit_stance_length = [0.35, 0.45]

    Cfg.commands.num_bins_vel_x = 21
    Cfg.commands.num_bins_vel_y = 1
    Cfg.commands.num_bins_vel_yaw = 21
    Cfg.commands.num_bins_body_height = 1
    Cfg.commands.num_bins_gait_frequency = 1
    Cfg.commands.num_bins_gait_phase = 1
    Cfg.commands.num_bins_gait_offset = 1
    Cfg.commands.num_bins_gait_bound = 1
    Cfg.commands.num_bins_gait_duration = 1
    Cfg.commands.num_bins_footswing_height = 1
    Cfg.commands.num_bins_body_roll = 1
    Cfg.commands.num_bins_body_pitch = 1
    Cfg.commands.num_bins_stance_width = 1

    Cfg.normalization.friction_range = [0, 1]
    Cfg.normalization.ground_friction_range = [0, 1]
    Cfg.terrain.yaw_init_range = 3.14
    Cfg.normalization.clip_actions = 10.0

    Cfg.commands.exclusive_phase_offset = False
    Cfg.commands.pacing_offset = False
    Cfg.commands.binary_phases = True
    Cfg.commands.gaitwise_curricula = True

    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=headless, cfg=Cfg)
    env.reset()

    if render and headless:
        img = env.render(mode="rgb_array")
        plt.imshow(img)
        plt.show()
        print("Show the first frame and exit.")
        exit()

    for i in trange(1000, desc="Running"):
        actions = 0. * torch.ones(env.num_envs, env.num_actions, device=env.device)
        obs, rew, done, info = env.step(actions)

    print("Done")


if __name__ == '__main__':
    run_env(render=True, headless=True)
