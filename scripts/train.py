from argparse import ArgumentParser

import isaacgym
assert isaacgym
import torch
import numpy as np
import random

from go1_gym import MINI_GYM_ROOT_DIR
from go1_gym.envs.base.legged_robot_config import Cfg
from go1_gym.envs.go1.go1_config import Go1Config
from go1_gym.envs.go1.go1_config_adaptive import AdaptiveGo1Config
from go1_gym.envs.go1.go1_config_adaptive_terrain import AdaptiveGo1ConfigTerrain
from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv
from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper
from go1_gym.utils.helpers import class_to_dict

from ml_logger import logger

from go1_gym_learn.ppo_cse import Runner
from go1_gym_learn.ppo_cse.actor_critic import AC_Args
from go1_gym_learn.ppo_cse.ppo import PPO_Args
from go1_gym_learn.ppo_cse import RunnerArgs

from pathlib import Path
import yaml


def train_go1(args, logdir):
    
    cfg_mapping = {
        "original": Go1Config,
        "adaptive_en": AdaptiveGo1Config,
        "adaen_terrain": AdaptiveGo1ConfigTerrain
    }

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    cfg: Cfg = cfg_mapping[args.cfg]()
    cfg.rewards.scales.energy_new_actual = args.en_new_actual
    cfg.rewards.scales.energy_new_cmd = args.en_new_cmd
    env_dict = class_to_dict(cfg)
    with open(f"{logdir}/env_cfg.yaml", "w") as file:
        yaml.dump(env_dict, file)
    file.close()

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
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--cfg', default="adaptive_en", type=str)
    parser.add_argument('--en_new_actual', default=0.0, type=float)
    parser.add_argument('--en_new_cmd', default=0.0, type=float)
    args = parser.parse_args()

    stem = Path(__file__).stem
    if args.cfg == "adaen_terrain":
        logdir = f"{MINI_GYM_ROOT_DIR}/checkpoints/{stem}/terrain-seed-{args.seed}-ennewa-{args.en_new_actual:.1f}-ennewc-{args.en_new_cmd:.1f}"
        logger.configure(
            logger.utcnow(f'{stem}/terrain-seed-{args.seed}-ennewa-{args.en_new_actual:.1f}-ennewc-{args.en_new_cmd:.1f}'),
            root=Path(f"{MINI_GYM_ROOT_DIR}/checkpoints").resolve()
        )
    else:
        logdir = f"{MINI_GYM_ROOT_DIR}/checkpoints/{stem}/seed-{args.seed}-ennewa-{args.en_new_actual:.1f}-ennewc-{args.en_new_cmd:.1f}"
        logger.configure(
            logger.utcnow(f'{stem}/seed-{args.seed}-ennewa-{args.en_new_actual:.1f}-ennewc-{args.en_new_cmd:.1f}'),
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
    train_go1(args=args, logdir=logdir)
