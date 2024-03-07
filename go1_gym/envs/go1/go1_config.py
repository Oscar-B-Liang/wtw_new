from typing import Union

from params_proto import Meta

from go1_gym.envs.base.legged_robot_config import Cfg


def config_go1(Cnfg: Union[Cfg, Meta]):
    _ = Cnfg.init_state

    _.pos = [0.0, 0.0, 0.34]  # x,y,z [m]
    _.default_joint_angles = {  # = target angles [rad] when action = 0.0
        'FL_hip_joint': 0.1,  # [rad]
        'RL_hip_joint': 0.1,  # [rad]
        'FR_hip_joint': -0.1,  # [rad]
        'RR_hip_joint': -0.1,  # [rad]

        'FL_thigh_joint': 0.8,  # [rad]
        'RL_thigh_joint': 1.,  # [rad]
        'FR_thigh_joint': 0.8,  # [rad]
        'RR_thigh_joint': 1.,  # [rad]

        'FL_calf_joint': -1.5,  # [rad]
        'RL_calf_joint': -1.5,  # [rad]
        'FR_calf_joint': -1.5,  # [rad]
        'RR_calf_joint': -1.5  # [rad]
    }

    _ = Cnfg.control
    _.control_type = "actuator_net"
    _.stiffness = {'joint': 20.}  # [N*m/rad]
    _.damping = {'joint': 0.5}  # [N*m*s/rad]
    # action scale: target angle = actionScale * action + defaultAngle
    _.action_scale = 0.25
    _.hip_scale_reduction = 0.5
    # decimation: Number of control action updates @ sim DT per policy DT
    _.decimation = 4

    _ = Cnfg.asset
    _.file = '{MINI_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1.urdf'
    _.foot_name = "foot"
    _.penalize_contacts_on = ["thigh", "calf"]
    _.terminate_after_contacts_on = ["base", "thigh", "calf", "hip"]
    _.self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
    _.flip_visual_attachments = False
    _.fix_base_link = False

    _ = Cnfg.rewards
    _.soft_dof_pos_limit = 0.9
    _.base_height_target = 0.3
    _.use_terminal_foot_height = False
    _.use_terminal_body_height = True
    _.terminal_body_height = 0.05
    _.use_terminal_roll_pitch = True
    _.terminal_body_ori = 1.6
    _.alpha_normalize = False
    _.alpha_check_speeds = [-2.5, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5]
    _.alpha_check_values = [0.7, 0.9, 0.9, 1.1, 1.3, 1.3, 1.1, 0.9, 0.9, 0.7]
    _.alpha_check_scales = [0.72, 0.81, 1.00, 1.18, 1.45, 1.45, 1.18, 1.00, 0.81, 0.72]

    _.kappa_gait_probs = 0.07
    _.gait_force_sigma = 100.
    _.gait_vel_sigma = 10.

    _.reward_container_name = "CoRLRewards"
    _.only_positive_rewards = False
    _.only_positive_rewards_ji22_style = True
    _.sigma_rew_neg = 0.02
    
    _ = Cnfg.reward_scales
    _.torques = -0.0001
    _.action_rate = -0.01
    _.dof_pos_limits = -10.0
    _.orientation = -5.
    _.base_height = -30.

    _ = Cnfg.terrain
    _.mesh_type = 'trimesh'
    _.measure_heights = False
    _.terrain_noise_magnitude = 0.0
    _.teleport_robots = True
    _.border_size = 0.0

    _.terrain_proportions = [0, 0, 0, 0, 0, 0, 0, 0, 1.0]
    _.curriculum = False
    _.num_cols = 30
    _.num_rows = 30
    _.terrain_width = 5.0
    _.terrain_length = 5.0
    _.x_init_range = 0.2
    _.y_init_range = 0.2
    _.teleport_thresh = 0.3
    _.center_robots = True
    _.center_span = 4
    _.horizontal_scale = 0.10
    
    _ = Cnfg.curriculum_thresholds
    _.tracking_ang_vel = 0.7
    _.tracking_lin_vel = 0.8
    _.tracking_contacts_shaped_vel = 0.90
    _.tracking_contacts_shaped_force = 0.90
    
    _ = Cnfg.env
    _.num_observations = 70
    _.observe_vel = False
    _.num_envs = 4000    
    _.priv_observe_motion = False
    _.priv_observe_gravity_transformed_motion = False
    _.priv_observe_friction_indep = False
    _.priv_observe_friction = True
    _.priv_observe_restitution = True
    _.priv_observe_base_mass = False
    _.priv_observe_gravity = False
    _.priv_observe_com_displacement = False
    _.priv_observe_ground_friction = False
    _.priv_observe_ground_friction_per_foot = False
    _.priv_observe_motor_strength = False
    _.priv_observe_motor_offset = False
    _.priv_observe_Kp_factor = False
    _.priv_observe_Kd_factor = False
    _.priv_observe_body_velocity = False
    _.priv_observe_body_height = False
    _.priv_observe_desired_contact_states = False
    _.priv_observe_contact_forces = False
    _.priv_observe_foot_displacement = False
    _.priv_observe_gravity_transformed_foot_displacement = False

    _.num_privileged_obs = 2
    _.num_observation_history = 30
    _.observe_two_prev_actions = True
    _.observe_yaw = False
    _.num_observations = 70
    _.num_scalar_observations = 70
    _.observe_gait_commands = True
    _.observe_timing_parameter = False
    _.observe_clock_inputs = True
    
    _.commands_mask = "del"
    
    _ = Cnfg.commands
    _.heading_command = False
    _.resampling_time = 10.0
    _.command_curriculum = True
    _.num_lin_vel_bins = 30
    _.num_ang_vel_bins = 30
    _.lin_vel_x = [-1.0, 1.0]
    _.lin_vel_y = [-0.6, 0.6]
    _.ang_vel_yaw = [-1.0, 1.0]
    _.distributional_commands = True
    _.num_commands = 15
    _.body_height_cmd = [-0.25, 0.15]
    _.gait_frequency_cmd_range = [2.0, 4.0]
    _.gait_phase_cmd_range = [0.0, 1.0]
    _.gait_offset_cmd_range = [0.0, 1.0]
    _.gait_bound_cmd_range = [0.0, 1.0]
    _.gait_duration_cmd_range = [0.5, 0.5]
    _.footswing_height_range = [0.03, 0.35]
    _.body_pitch_range = [-0.4, 0.4]
    _.body_roll_range = [-0.0, 0.0]
    _.stance_width_range = [0.10, 0.45]
    _.stance_length_range = [0.35, 0.45]

    _.limit_vel_x = [-5.0, 5.0]
    _.limit_vel_y = [-0.6, 0.6]
    _.limit_vel_yaw = [-5.0, 5.0]

    _.limit_body_height = [-0.25, 0.15]
    _.limit_gait_frequency = [2.0, 4.0]
    _.limit_gait_phase = [0.0, 1.0]
    _.limit_gait_offset = [0.0, 1.0]
    _.limit_gait_bound = [0.0, 1.0]
    _.limit_gait_duration = [0.5, 0.5]
    _.limit_footswing_height = [0.03, 0.35]
    _.limit_body_pitch = [-0.4, 0.4]
    _.limit_body_roll = [-0.0, 0.0]
    _.limit_stance_width = [0.10, 0.45]
    _.limit_stance_length = [0.35, 0.45]

    _.num_bins_vel_x = 21
    _.num_bins_vel_y = 1
    _.num_bins_vel_yaw = 21
    _.num_bins_body_height = 1
    _.num_bins_gait_frequency = 1
    _.num_bins_gait_phase = 1
    _.num_bins_gait_offset = 1
    _.num_bins_gait_bound = 1
    _.num_bins_gait_duration = 1
    _.num_bins_footswing_height = 1
    _.num_bins_body_roll = 1
    _.num_bins_body_pitch = 1
    _.num_bins_stance_width = 1
    _.exclusive_phase_offset = False
    _.pacing_offset = False
    _.binary_phases = True
    _.gaitwise_curricula = True
    
    _ = Cnfg.domain_rand
    _.randomize_base_mass = True
    _.push_robots = False
    _.max_push_vel_xy = 0.5
    _.randomize_friction = True
    _.randomize_restitution = True
    _.restitution = 0.5  # default terrain restitution
    _.randomize_motor_strength = True
    _.Kp_factor_range = [0.8, 1.3]
    _.Kd_factor_range = [0.5, 1.5]
    _.randomize_rigids_after_start = False
    _.randomize_friction_indep = False
    _.friction_range = [0.1, 3.0]
    _.restitution_range = [0.0, 0.4]
    _.added_mass_range = [-1.0, 3.0]
    _.randomize_gravity = True
    _.gravity_range = [-1.0, 1.0]
    _.gravity_rand_interval_s = 8.0
    _.gravity_impulse_duration = 0.99
    _.randomize_com_displacement = False
    _.com_displacement_range = [-0.15, 0.15]
    _.randomize_ground_friction = True
    _.ground_friction_range = [0.0, 0.0]
    _.motor_strength_range = [0.9, 1.1]
    _.randomize_motor_offset = True
    _.motor_offset_range = [-0.02, 0.02]
    _.randomize_Kp_factor = False
    _.randomize_Kd_factor = False
    _.rand_interval_s = 4
    _.tile_height_range = [-0.0, 0.0]
    _.tile_height_curriculum = False
    _.tile_height_update_interval = 1000000
    _.tile_height_curriculum_step = 0.01
    
    _ = Cnfg.normalization
    _.friction_range = [0, 1]
    _.ground_friction_range = [0, 1]
    _.clip_actions = 10.0
    
    # Process delete commands
    if Cnfg.env.commands_mask == "del":
        Cnfg.env.num_observations = 70 - 16
        Cnfg.env.num_scalar_observations = 70 - 16
        Cnfg.env.observe_clock_inputs = False
        Cnfg.env.observe_gait_commands = False
      