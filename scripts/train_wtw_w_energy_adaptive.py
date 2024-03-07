def train_go1(args):

    import isaacgym
    assert isaacgym
    import torch

    from go1_gym.envs.base.legged_robot_config import Cfg
    from go1_gym.envs.go1.go1_config import config_go1
    from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv

    from ml_logger import logger

    from go1_gym_learn.ppo_cse import Runner
    from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper
    from go1_gym_learn.ppo_cse.actor_critic import AC_Args
    from go1_gym_learn.ppo_cse.ppo import PPO_Args
    from go1_gym_learn.ppo_cse import RunnerArgs

    config_go1(Cfg)

    # Task Rewards.
    Cfg.reward_scales.tracking_lin_vel = 1.0
    Cfg.reward_scales.tracking_lin_vel_dep = 0.0
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
    Cfg.rewards.soft_dof_pos_limit = 0.9
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
    Cfg.reward_scales.dof_pos = 0.0
    Cfg.reward_scales.feet_impact_vel = 0.0
    Cfg.reward_scales.orientation = -5.0
    Cfg.reward_scales.feet_contact_forces = 0.0

    # Energy rewards.
    Cfg.reward_scales.energy = 0.0
    Cfg.reward_scales.energy_sigma = 300
    Cfg.reward_scales.energy_dep = 1.0
    Cfg.reward_scales.energy_legs = 0.0
    Cfg.reward_scales.energy_legs_sigma = 100

    # Uncorresponded Rewards.
    Cfg.reward_scales.base_height = 0.0
    Cfg.reward_scales.estimation_bonus = 0.0
    Cfg.reward_scales.feet_clearance = 0.0
    Cfg.reward_scales.feet_clearance_cmd = 0.0
    Cfg.reward_scales.tracking_stance_width = 0.0
    Cfg.reward_scales.tracking_stance_length = 0.0
    Cfg.reward_scales.hop_symmetry = 0.0
    Cfg.reward_scales.feet_stumble = 0.0
    Cfg.reward_scales.stand_still = 0.0
    Cfg.reward_scales.tracking_lin_vel_lat = 0.0
    Cfg.reward_scales.tracking_lin_vel_long = 0.0
    Cfg.reward_scales.tracking_contacts = 0.0
    Cfg.reward_scales.tracking_contacts_shaped = 0.0
    Cfg.reward_scales.energy_expenditure = 0.0
    Cfg.reward_scales.survival = 0.0
    Cfg.reward_scales.base_motion = 0.0

    Cfg.rewards.alpha_normalize = True
    Cfg.rewards.alpha_check_speeds = [-2.5, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5]
    Cfg.rewards.alpha_check_values = [0.7, 0.9, 0.9, 1.1, 1.3, 1.3, 1.1, 0.9, 0.9, 0.7]
    Cfg.rewards.alpha_check_scales = [0.72, 0.81, 1.00, 1.18, 1.45, 1.45, 1.18, 1.00, 0.81, 0.72]

    if args.train_speed is not None:
        Cfg.commands.lin_vel_x = [-min(args.train_speed + 0.1, 1.0), min(args.train_speed + 0.1, 1.0)]
        Cfg.commands.limit_vel_x = [-(args.train_speed + 0.1), args.train_speed + 0.1]

    env = VelocityTrackingEasyEnv(sim_device=f'cuda:{args.device}', headless=args.headless, cfg=Cfg)

    # log the experiment parameters
    logger.log_params(AC_Args=vars(AC_Args), PPO_Args=vars(PPO_Args), RunnerArgs=vars(RunnerArgs),
                      Cfg=vars(Cfg))

    env = HistoryWrapper(env)
    gpu_id = args.device
    runner = Runner(env, device=f"cuda:{gpu_id}")
    runner.learn(num_learning_iterations=2000, init_at_random_ep_len=True, eval_freq=100)


if __name__ == '__main__':
    from pathlib import Path
    from ml_logger import logger
    from go1_gym import MINI_GYM_ROOT_DIR

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--headless', action="store_true")
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--train_speed', default=None, type=float)
    args = parser.parse_args()

    stem = Path(__file__).stem
    if args.train_speed is None:
        logger.configure(
            logger.utcnow(f'{stem}/adaptive'),
            root=Path(f"{MINI_GYM_ROOT_DIR}/checkpoints").resolve()
        )
    else:
        logger.configure(
            logger.utcnow(f'{stem}/adaptive-speed-{args.train_speed:.1f}'),
            root=Path(f"{MINI_GYM_ROOT_DIR}/checkpoints").resolve()
        )
    logger.log_text("""
                charts: 
                - yKey: train/episode/rew_total/mean
                  xKey: iterations
                - yKey: train/episode/rew_tracking_lin_vel/mean
                  xKey: iterations
                - yKey: train/episode/rew_tracking_contacts_shaped_force/mean
                  xKey: iterations
                - yKey: train/episode/rew_action_smoothness_1/mean
                  xKey: iterations
                - yKey: train/episode/rew_action_smoothness_2/mean
                  xKey: iterations
                - yKey: train/episode/rew_tracking_contacts_shaped_vel/mean
                  xKey: iterations
                - yKey: train/episode/rew_orientation_control/mean
                  xKey: iterations
                - yKey: train/episode/rew_dof_pos/mean
                  xKey: iterations
                - yKey: train/episode/command_area_trot/mean
                  xKey: iterations
                - yKey: train/episode/max_terrain_height/mean
                  xKey: iterations
                - type: video
                  glob: "videos/*.mp4"
                - yKey: adaptation_loss/mean
                  xKey: iterations
                """, filename=".charts.yml", dedent=True)

    # to see the environment rendering, set headless=False
    train_go1(args=args)
