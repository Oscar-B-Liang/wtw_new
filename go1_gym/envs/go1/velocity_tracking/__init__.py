from isaacgym import gymutil, gymapi
import torch
from params_proto import Meta
from typing import Union

from go1_gym.envs.base.legged_robot import LeggedRobot
from go1_gym.envs.base.legged_robot_config import Cfg


class VelocityTrackingEasyEnv(LeggedRobot):
    def __init__(self, sim_device, headless, num_envs=None, prone=False, deploy=False, temp_cap_dir: str = None, enable_camera_sensor: bool = False,
                 cfg: Cfg = None, eval_cfg: Cfg = None, initial_dynamics_dict=None, physics_engine="SIM_PHYSX"):

        if num_envs is not None:
            cfg.env.num_envs = num_envs

        sim_params = gymapi.SimParams()
        # gymutil.parse_sim_config(vars(cfg.sim), sim_params)
        sim_params.dt = cfg.sim.dt
        sim_params.substeps = cfg.sim.substeps
        sim_params.gravity = gymapi.Vec3(cfg.sim.gravity[0], cfg.sim.gravity[1], cfg.sim.gravity[2])
        sim_params.up_axis = gymapi.UpAxis(cfg.sim.up_axis)
        sim_params.use_gpu_pipeline = cfg.sim.use_gpu_pipeline
        sim_params.physx.num_threads = cfg.sim.physx.num_threads
        sim_params.physx.solver_type = cfg.sim.physx.solver_type
        sim_params.physx.num_position_iterations = cfg.sim.physx.num_position_iterations
        sim_params.physx.num_velocity_iterations = cfg.sim.physx.num_velocity_iterations
        sim_params.physx.contact_offset = cfg.sim.physx.contact_offset
        sim_params.physx.rest_offset = cfg.sim.physx.rest_offset
        sim_params.physx.bounce_threshold_velocity = cfg.sim.physx.bounce_threshold_velocity
        sim_params.physx.max_depenetration_velocity = cfg.sim.physx.max_depenetration_velocity
        sim_params.physx.max_gpu_contact_pairs = cfg.sim.physx.max_gpu_contact_pairs
        sim_params.physx.default_buffer_size_multiplier = cfg.sim.physx.default_buffer_size_multiplier
        sim_params.physx.contact_collection = gymapi.ContactCollection(cfg.sim.physx.contact_collection)

        super().__init__(cfg, sim_params, physics_engine, sim_device, headless, temp_cap_dir, eval_cfg, enable_camera_sensor, initial_dynamics_dict)


    def step(self, actions):
        self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras = super().step(actions)

        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices,
                               0:3]

        self.extras.update({
            "privileged_obs": self.privileged_obs_buf,
            "joint_pos": self.dof_pos.cpu().numpy(),
            "joint_vel": self.dof_vel.cpu().numpy(),
            "joint_pos_target": self.joint_pos_target.cpu().detach().numpy(),
            "joint_vel_target": torch.zeros(12),
            "body_linear_vel": self.base_lin_vel.cpu().detach().numpy(),
            "body_angular_vel": self.base_ang_vel.cpu().detach().numpy(),
            "body_linear_vel_cmd": self.commands.cpu().numpy()[:, 0:2],
            "body_angular_vel_cmd": self.commands.cpu().numpy()[:, 2:],
            "contact_states": (self.contact_forces[:, self.feet_indices, 2] > 1.).detach().cpu().numpy().copy(),
            "foot_positions": (self.foot_positions).detach().cpu().numpy().copy(),
            "body_pos": self.root_states[:, 0:3].detach().cpu().numpy(),
            "torques": self.torques.detach().cpu().numpy()
        })

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs

