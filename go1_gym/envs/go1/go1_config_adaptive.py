from go1_gym.envs.go1.go1_config import Go1Config


class AdaptiveGo1Config(Go1Config):

    class rewards(Go1Config.rewards):

        alpha_normalize = True
        m_alpha = 0.3295
        b_alpha = 0.6791
        m_Z = 0.4317
        b_Z = 0.6377
        # alpha_check_speeds = [-2.5, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5]
        # alpha_check_values = [0.7, 0.9, 0.9, 1.1, 1.3, 1.3, 1.1, 0.9, 0.9, 0.7]
        # alpha_check_scales = [0.72, 0.81, 1.00, 1.18, 1.45, 1.45, 1.18, 1.00, 0.81, 0.72]

        energy_sigma = 300.0
        energy_legs_sigma = 100.0

        class scales(Go1Config.rewards.scales):
            tracking_lin_vel = 1.0
            tracking_lin_vel_dep = 0.0
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
            soft_dof_pos_limit = 0.9
            dof_pos_limits = -10.0
            torques = -0.0001
            dof_vel = -1e-4
            dof_acc = -2.5e-7
            action_smoothness_1 = -0.1
            action_smoothness_2 = -0.1

            # Rewards used in legged gym, but unparticipated here.
            feet_air_time = 0.0
            action_rate = -0.01

            # Unparticipated Rewards.
            dof_pos = 0.0
            feet_impact_vel = 0.0
            orientation = -5.0
            feet_contact_forces = 0.0

            # Energy rewards.
            energy = 0.0
            energy_dep = 1.0
            energy_legs = 0.0

            # Uncorresponded Rewards.
            base_height = 0.0
            estimation_bonus = 0.0
            feet_clearance = 0.0
            feet_clearance_cmd = 0.0
            tracking_stance_width = 0.0
            tracking_stance_length = 0.0
            hop_symmetry = 0.0
            feet_stumble = 0.0
            stand_still = 0.0
            tracking_lin_vel_lat = 0.0
            tracking_lin_vel_long = 0.0
            tracking_contacts = 0.0
            tracking_contacts_shaped = 0.0
            energy_expenditure = 0.0
            survival = 0.0
            base_motion = 0.0
