from argparse import ArgumentParser
import isaacgym

assert isaacgym
import torch
import numpy as np

import pickle as pkl

from go1_gym import MINI_GYM_ROOT_DIR
from go1_gym.envs.go1.go1_config_adaptive import AdaptiveGo1Config
from go1_gym.envs.go1.go1_config_adaptive_terrain import AdaptiveGo1ConfigTerrain
from go1_gym.envs.base.legged_robot_config import Cfg
from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv
from go1_gym.utils.helpers import dict_to_env_cfg
from go1_gym.utils.logger import Logger

from tqdm import tqdm
import os
import io


class Local_Unpickler(pkl.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def load_policy(logdir):
    body = torch.jit.load(logdir + '/checkpoints/body_latest.jit', map_location="cpu")
    # import os
    adaptation_module = torch.jit.load(logdir + '/checkpoints/adaptation_module_latest.jit', map_location="cpu")

    def policy(obs, info={}):
        # i = 0
        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
        info['latent'] = latent
        return action

    return policy


def load_env(logdir, headless=False):
    print("Loading from directory ", logdir)

    with open(logdir + "/parameters.pkl", 'rb') as file:
        pkl_cfg = Local_Unpickler(file).load()
        cfg = AdaptiveGo1ConfigTerrain()
        cfg: Cfg = dict_to_env_cfg(pkl_cfg["Cfg"], cfg)

    # turn off DR for evaluation script
    cfg.domain_rand.push_robots = False
    cfg.domain_rand.randomize_friction = False
    cfg.domain_rand.randomize_gravity = False
    cfg.domain_rand.randomize_restitution = False
    cfg.domain_rand.randomize_motor_offset = False
    cfg.domain_rand.randomize_motor_strength = False
    cfg.domain_rand.randomize_friction_indep = False
    cfg.domain_rand.randomize_ground_friction = False
    cfg.domain_rand.randomize_base_mass = False
    cfg.domain_rand.randomize_Kd_factor = False
    cfg.domain_rand.randomize_Kp_factor = False
    cfg.domain_rand.randomize_joint_friction = False
    cfg.domain_rand.randomize_com_displacement = False

    # Disable terrain
    # cfg.terrainterrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
    # cfg.terrain.curriculum = False

    cfg.env.num_recording_envs = 1
    cfg.env.num_envs = 1

    cfg.terrain.num_rows = 5
    cfg.terrain.num_cols = 5
    cfg.terrain.border_size = 0
    cfg.terrain.center_robots = True
    cfg.terrain.center_span = 1
    cfg.terrain.teleport_robots = True

    cfg.domain_rand.lag_timesteps = 6
    cfg.domain_rand.randomize_lag_timesteps = True
    cfg.control.control_type = "actuator_net"

    cfg.viewer.cam_offset = [1.0, 2.0, 0.6]

    # Do camera recording in play:
    cfg.viewer.cam_env_ids = [0]
    temp_cap_dir = os.path.join(MINI_GYM_ROOT_DIR, logdir, "temp_cap_dir")
    os.makedirs(temp_cap_dir, exist_ok=True)

    from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper

    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=headless, cfg=cfg, enable_camera_sensor=True, temp_cap_dir=temp_cap_dir)
    env = HistoryWrapper(env)

    # load policy
    policy = load_policy(logdir)

    return env, policy


def play_go1(model_dir, lin_x_speed, yaw_speed, headless=True):

    model_dir = f"{MINI_GYM_ROOT_DIR}/{model_dir}"
    env, policy = load_env(model_dir, headless=headless)
    os.makedirs(os.path.join(model_dir, "analysis"), exist_ok=True)

    num_eval_steps = 500
    gaits = {
        "pronking": [0, 0, 0],
        "trotting": [0.5, 0, 0],
        "bounding": [0, 0.5, 0],
        "pacing": [0, 0, 0.5]
    }

    x_vel_cmd, y_vel_cmd, yaw_vel_cmd = lin_x_speed, 0.0, yaw_speed
    # body_height_cmd = 0.0
    # step_frequency_cmd = 3.0
    # gait = torch.tensor(gaits["trotting"])
    # footswing_height_cmd = 0.08
    # pitch_cmd = 0.0
    # roll_cmd = 0.0
    # stance_width_cmd = 0.25
    body_height_cmd = 0.0
    step_frequency_cmd = 0.0
    gait = torch.tensor(gaits["pronking"])
    footswing_height_cmd = 0.0
    pitch_cmd = 0.0
    roll_cmd = 0.0
    stance_width_cmd = 0.0

    measured_x_vels = np.zeros(num_eval_steps)
    joint_positions = np.zeros((num_eval_steps, 12))

    logger = Logger(env.env.dt, env.env.dof_names, env.feet_names, lin_x_speed, yaw_speed, model_dir, 200)
    obs = env.reset()

    env.env.start_video_recording()

    for i in tqdm(range(num_eval_steps)):
        with torch.no_grad():
            actions = policy(obs)
        env.commands[:, 0] = x_vel_cmd
        env.commands[:, 1] = y_vel_cmd
        env.commands[:, 2] = yaw_vel_cmd
        env.commands[:, 3] = body_height_cmd
        env.commands[:, 4] = step_frequency_cmd
        env.commands[:, 5:8] = gait
        env.commands[:, 8] = 0.0
        env.commands[:, 9] = footswing_height_cmd
        env.commands[:, 10] = pitch_cmd
        env.commands[:, 11] = roll_cmd
        env.commands[:, 12] = stance_width_cmd
        obs, rew, done, info = env.step(actions)

        if i >= 100 and i <= 400:
            log_dict = {
                'command_x': env.env.commands[:, 0].cpu().numpy(),
                'command_y': env.env.commands[:, 1].cpu().numpy(),
                'command_yaw': env.env.commands[:, 2].cpu().numpy(),
            }

            log_dict['base_pos_x'] = env.env.base_pos[:, 0].cpu().numpy()
            log_dict['base_pos_y'] = env.env.base_pos[:, 1].cpu().numpy()
            log_dict['base_pos_z'] = env.env.base_pos[:, 2].cpu().numpy()
            log_dict['base_vel_x'] = env.env.base_lin_vel[:, 0].cpu().numpy()
            log_dict['base_vel_y'] = env.env.base_lin_vel[:, 1].cpu().numpy()
            log_dict['base_vel_z'] = env.env.base_lin_vel[:, 2].cpu().numpy()
            log_dict['base_vel_roll'] = env.env.base_ang_vel[:, 0].cpu().numpy()
            log_dict['base_vel_pitch'] = env.env.base_ang_vel[:, 1].cpu().numpy()
            log_dict['base_vel_yaw'] = env.env.base_ang_vel[:, 2].cpu().numpy()
            log_dict['contact_forces_x'] = env.env.contact_forces[:, env.env.feet_indices, 0].cpu().numpy()
            log_dict['contact_forces_y'] = env.env.contact_forces[:, env.env.feet_indices, 1].cpu().numpy()
            log_dict['contact_forces_z'] = env.env.contact_forces[:, env.env.feet_indices, 2].cpu().numpy()
            log_dict['reward'] = env.env.rew_buf[:].detach().clone().cpu().numpy()
            log_dict['energy_consume'] = env.env.energy_consume[:].cpu().numpy()
            log_dict['dof_pos'] = env.env.dof_pos.cpu().numpy()
            log_dict['dof_vel'] = env.env.dof_vel.cpu().numpy()
            log_dict['dof_acc'] = env.env.dof_acc.cpu().numpy()
            log_dict['dof_torque'] = env.env.torques.detach().clone().cpu().numpy()
            log_dict['action_scaled'] = env.env.actions.detach().clone().cpu().numpy() # * env_cfg.control.action_scale
            logger.log_states(log_dict)

        measured_x_vels[i] = env.base_lin_vel[0, 0]
        joint_positions[i] = env.dof_pos[0, :].cpu()
    
    logger.plot_save_states(f"{lin_x_speed}_{yaw_speed}", [0])

    video_save_paths = [os.path.join(model_dir, "analysis", f"{lin_x_speed}_{yaw_speed}_env_{i}.mp4") for i in env.env.cfg.viewer.cam_env_ids]
    env.env.stop_video_recording(video_save_paths)


if __name__ == '__main__':

    parser = ArgumentParser()
    # model_dir is the relative path starting from this repo root.
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--model_dir", type=str, default="checkpoints/train_min_cmd/2024-02-19-044826.014731")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--lin_speed", type=float, default=2.0)
    parser.add_argument("--ang_speed", type=float, default=0.0)
    parser.add_argument("--terrain_choice", type=str, default="flat")
    args = parser.parse_args()

    play_go1(model_dir=args.model_dir, lin_x_speed=args.lin_speed, yaw_speed=args.ang_speed, headless=args.headless)
