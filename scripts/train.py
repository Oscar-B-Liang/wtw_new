from argparse import ArgumentParser

import isaacgym
assert isaacgym
import torch
import numpy as np
import random

from go1_gym import MINI_GYM_ROOT_DIR
from go1_gym.envs.base.legged_robot_config import Cfg
from go1_gym.envs.go1.go1_config_adaptive import AdaptiveGo1Config
from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv

from ml_logger import logger

from go1_gym_learn.ppo_cse import Runner
from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper
from go1_gym_learn.ppo_cse.actor_critic import AC_Args
from go1_gym_learn.ppo_cse.ppo import PPO_Args
from go1_gym_learn.ppo_cse import RunnerArgs

from pathlib import Path


def train_go1(args):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    cfg = AdaptiveGo1Config()

    if args.train_speed is not None:
        cfg.commands.lin_vel_x = [-min(args.train_speed + 0.1, 1.0), min(args.train_speed + 0.1, 1.0)]
        cfg.commands.limit_vel_x = [-(args.train_speed + 0.1), args.train_speed + 0.1]

    env = VelocityTrackingEasyEnv(sim_device=f'cuda:{args.device}', headless=args.headless, cfg=cfg)

    # log the experiment parameters
    logger.log_params(
        AC_Args=vars(AC_Args),
        PPO_Args=vars(PPO_Args),
        RunnerArgs=vars(RunnerArgs),
        Cfg=vars(Cfg)
    )

    env = HistoryWrapper(env)
    gpu_id = args.device
    runner = Runner(env, device=f"cuda:{gpu_id}")
    runner.learn(num_learning_iterations=2000, init_at_random_ep_len=True, eval_freq=100)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--headless', action="store_true")
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--train_speed', default=None, type=float)
    parser.add_argument('--seed', default=0, type=int)
    args = parser.parse_args()

    stem = Path(__file__).stem
    if args.train_speed is None:
        logger.configure(
            logger.utcnow(f'{stem}/seed-{args.seed}'),
            root=Path(f"{MINI_GYM_ROOT_DIR}/checkpoints").resolve()
        )
    else:
        logger.configure(
            logger.utcnow(f'{stem}/seed-{args.seed}-speed-{args.train_speed:.1f}'),
            root=Path(f"{MINI_GYM_ROOT_DIR}/checkpoints").resolve()
        )
    logger.log_text(
        """
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
        """,
        filename=".charts.yml",
        dedent=True
    )

    # to see the environment rendering, set headless=False
    train_go1(args=args)
