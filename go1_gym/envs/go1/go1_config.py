from go1_gym.envs.base.legged_robot_config import Cfg


class Go1Config(Cfg):

    class init_state(Cfg.init_state):
        pos = [0.0, 0.0, 0.34]
        default_joint_angles = {
            'FL_hip_joint': 0.1,
            'RL_hip_joint': 0.1,
            'FR_hip_joint': -0.1,
            'RR_hip_joint': -0.1,

            'FL_thigh_joint': 0.8,
            'RL_thigh_joint': 1.,
            'FR_thigh_joint': 0.8,
            'RR_thigh_joint': 1.,

            'FL_calf_joint': -1.5,
            'RL_calf_joint': -1.5,
            'FR_calf_joint': -1.5,
            'RR_calf_joint': -1.5
        }
    
    class control(Cfg.control):
        control_type = "actuator_net"
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        hip_scale_reduction = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
    
    class asset(Cfg.asset):
        file = '{MINI_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1.urdf'
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base", "thigh", "calf", "hip"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        fix_base_link = False
    
    class rewards(Cfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.3
        use_terminal_foot_height = False
        use_terminal_body_height = True
        terminal_body_height = 0.05
        use_terminal_roll_pitch = True
        terminal_body_ori = 1.6

        kappa_gait_probs = 0.07
        gait_force_sigma = 100.
        gait_vel_sigma = 10.

        reward_container_name = "CoRLRewards"
        only_positive_rewards = False
        only_positive_rewards_ji22_style = True
        sigma_rew_neg = 0.02

        alpha_normalize = False
        alpha_check_speeds = [-2.5, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5]
        alpha_check_values = [0.7, 0.9, 0.9, 1.1, 1.3, 1.3, 1.1, 0.9, 0.9, 0.7]
        alpha_check_scales = [0.72, 0.81, 1.00, 1.18, 1.45, 1.45, 1.18, 1.00, 0.81, 0.72]

        class scales(Cfg.rewards.scales):
            torques = -0.0001
            action_rate = -0.01
            dof_pos_limits = -10.0
            orientation = -5.0
            base_height = -30.0
    
    class terrain(Cfg.terrain):
        mesh_type = 'trimesh'
        measure_heights = False
        terrain_noise_magnitude = 0.0
        teleport_robots = True
        border_size = 0.0

        terrain_proportions = [0, 0, 0, 0, 0, 0, 0, 0, 1.0]
        curriculum = False
        num_cols = 30
        num_rows = 30
        terrain_width = 5.0
        terrain_length = 5.0
        x_init_range = 0.2
        y_init_range = 0.2
        teleport_thresh = 0.3
        center_robots = True
        center_span = 4
        horizontal_scale = 0.10
    
    class curriculum_thresholds(Cfg.curriculum_thresholds):
        tracking_ang_vel = 0.7
        tracking_lin_vel = 0.8
        tracking_contacts_shaped_vel = 0.90
        tracking_contacts_shaped_force = 0.90
    
    class env(Cfg.env):
        num_observations = 70
        observe_vel = False
        num_envs = 4000    
        priv_observe_motion = False
        priv_observe_gravity_transformed_motion = False
        priv_observe_friction_indep = False
        priv_observe_friction = True
        priv_observe_restitution = True
        priv_observe_base_mass = False
        priv_observe_gravity = False
        priv_observe_com_displacement = False
        priv_observe_ground_friction = False
        priv_observe_ground_friction_per_foot = False
        priv_observe_motor_strength = False
        priv_observe_motor_offset = False
        priv_observe_Kp_factor = False
        priv_observe_Kd_factor = False
        priv_observe_body_velocity = False
        priv_observe_body_height = False
        priv_observe_desired_contact_states = False
        priv_observe_contact_forces = False
        priv_observe_foot_displacement = False
        priv_observe_gravity_transformed_foot_displacement = False

        num_privileged_obs = 2
        num_observation_history = 30
        observe_two_prev_actions = True
        observe_yaw = False
        num_observations = 70
        num_scalar_observations = 70
        observe_gait_commands = True
        observe_timing_parameter = False
        observe_clock_inputs = True
    
    class commands(Cfg.commands):
        heading_command = False
        resampling_time = 10.0
        command_curriculum = True
        num_lin_vel_bins = 30
        num_ang_vel_bins = 30
        lin_vel_x = [-1.0, 1.0]
        lin_vel_y = [-0.6, 0.6]
        ang_vel_yaw = [-1.0, 1.0]
        distributional_commands = True
        num_commands = 15
        body_height_cmd = [-0.25, 0.15]
        gait_frequency_cmd_range = [2.0, 4.0]
        gait_phase_cmd_range = [0.0, 1.0]
        gait_offset_cmd_range = [0.0, 1.0]
        gait_bound_cmd_range = [0.0, 1.0]
        gait_duration_cmd_range = [0.5, 0.5]
        footswing_height_range = [0.03, 0.35]
        body_pitch_range = [-0.4, 0.4]
        body_roll_range = [-0.0, 0.0]
        stance_width_range = [0.10, 0.45]
        stance_length_range = [0.35, 0.45]

        limit_vel_x = [-5.0, 5.0]
        limit_vel_y = [-0.6, 0.6]
        limit_vel_yaw = [-5.0, 5.0]

        limit_body_height = [-0.25, 0.15]
        limit_gait_frequency = [2.0, 4.0]
        limit_gait_phase = [0.0, 1.0]
        limit_gait_offset = [0.0, 1.0]
        limit_gait_bound = [0.0, 1.0]
        limit_gait_duration = [0.5, 0.5]
        limit_footswing_height = [0.03, 0.35]
        limit_body_pitch = [-0.4, 0.4]
        limit_body_roll = [-0.0, 0.0]
        limit_stance_width = [0.10, 0.45]
        limit_stance_length = [0.35, 0.45]

        num_bins_vel_x = 21
        num_bins_vel_y = 1
        num_bins_vel_yaw = 21
        num_bins_body_height = 1
        num_bins_gait_frequency = 1
        num_bins_gait_phase = 1
        num_bins_gait_offset = 1
        num_bins_gait_bound = 1
        num_bins_gait_duration = 1
        num_bins_footswing_height = 1
        num_bins_body_roll = 1
        num_bins_body_pitch = 1
        num_bins_stance_width = 1
        exclusive_phase_offset = False
        pacing_offset = False
        binary_phases = True
        gaitwise_curricula = True
    
    class domain_rand(Cfg.domain_rand):
        randomize_base_mass = True
        push_robots = False
        max_push_vel_xy = 0.5
        randomize_friction = True
        randomize_restitution = True
        restitution = 0.5  # default terrain restitution
        randomize_motor_strength = True
        Kp_factor_range = [0.8, 1.3]
        Kd_factor_range = [0.5, 1.5]
        randomize_rigids_after_start = False
        randomize_friction_indep = False
        friction_range = [0.1, 3.0]
        restitution_range = [0.0, 0.4]
        added_mass_range = [-1.0, 3.0]
        randomize_gravity = True
        gravity_range = [-1.0, 1.0]
        gravity_rand_interval_s = 8.0
        gravity_impulse_duration = 0.99
        randomize_com_displacement = False
        com_displacement_range = [-0.15, 0.15]
        randomize_ground_friction = True
        ground_friction_range = [0.0, 0.0]
        motor_strength_range = [0.9, 1.1]
        randomize_motor_offset = True
        motor_offset_range = [-0.02, 0.02]
        randomize_Kp_factor = False
        randomize_Kd_factor = False
        rand_interval_s = 4
        tile_height_range = [-0.0, 0.0]
        tile_height_curriculum = False
        tile_height_update_interval = 1000000
        tile_height_curriculum_step = 0.01
    
    class normalization(Cfg.normalization):
        friction_range = [0, 1]
        ground_friction_range = [0, 1]
        clip_actions = 10.0
      