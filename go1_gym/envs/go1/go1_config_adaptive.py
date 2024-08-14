from go1_gym.envs.go1.go1_config import Go1Config


class AdaptiveGo1Config(Go1Config):

    class rewards(Go1Config.rewards):

        alpha_normalize = False
        m_alpha = 0.3295
        b_alpha = 0.6791
        m_Z = 0.4317
        b_Z = 0.6377
        # alpha_check_speeds = [-2.5, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5]
        # alpha_check_values = [0.7, 0.9, 0.9, 1.1, 1.3, 1.3, 1.1, 0.9, 0.9, 0.7]
        # alpha_check_scales = [0.72, 0.81, 1.00, 1.18, 1.45, 1.45, 1.18, 1.00, 0.81, 0.72]

        energy_sigma = 300.0
        energy_sigma_lin = 1000.0
        energy_clip_lin = 0.2
        energy_sigma_ang = 500.0
        energy_clip_rot = 0.2

        class scales(Go1Config.rewards.scales):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5

            # Augmented Auxiliary Rewards.
            tracking_contacts_shaped_force = 0.0
            tracking_contacts_shaped_vel = 0.0
            jump = 0.0
            orientation_control = 0.0
            raibert_heuristic = 0.0
            feet_clearance_cmd_linear = 0.0

            # Fixed Auxiliary Rewards.
            lin_vel_z = -0.02
            ang_vel_xy = -0.001
            feet_slip = -0.04
            collision = -5.0
            dof_pos_limits = -10.0
            torques = -0.0001
            dof_vel = -1e-4
            dof_acc = -2.5e-7
            action_smoothness_1 = -0.1
            action_smoothness_2 = -0.1

            # Rewards used in legged gym, but unparticipated here.
            action_rate = -0.01

            # Unparticipated Rewards.
            dof_pos = 0.0
            feet_impact_vel = 0.0
            orientation = -5.0
            feet_contact_forces = 0.0

            # Energy rewards.
            energy = 0.0
            energy_dep = 0.0
            energy_new_actual = 0.8
            energy_new_cmd = 0.0
    
    class terrain:
        mesh_type = 'trimesh'  # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 0  # 25 # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0
        terrain_noise_magnitude = 0.0
        # rough terrain only:
        terrain_smoothness = 0.005
        measure_heights = True
        # 1mx1.6m rectangle (without center line)
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        min_init_terrain_level = 0
        max_init_terrain_level = 5  # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh only:
        slope_treshold = 0.75  # slopes above this threshold will be corrected to vertical surfaces
        difficulty_scale = 1.
        x_init_range = 0.2
        y_init_range = 0.2
        yaw_init_range = 0.
        x_init_offset = 0.
        y_init_offset = 0.
        teleport_robots = True
        teleport_thresh = 0.3
        max_platform_height = 0.2
        center_robots = False
        center_span = 5
