import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import itertools
import os
import pickle
import yaml


class Logger:

    def __init__(self, dt, dof_names, feet_names, lin_speed, ang_speed, model_dir, load_iteration):
        self.state_log = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0
        self.feet_names = feet_names
        self.dof_names = dof_names
        self.lin_speed = lin_speed
        self.ang_speed = ang_speed
        self.model_dir = model_dir
        self.load_iteration = load_iteration

    def log_state(self, key, value):
        self.state_log[key].append(value)

    def log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)

    def reset(self):
        self.state_log.clear()
        self.rew_log.clear()
    
    def get_vel_x_stats(self) -> (float, float):
        """Compute the x-direction average velocity in robot frame (by time).
        Averaged over all sub environments.
        Return the two values, mean velocity (by time) and its variance.
        """
        base_vel_x = np.array(self.state_log["base_vel_x"])
        base_vel_x = np.mean(base_vel_x, axis=0)
        return np.mean(base_vel_x).item(), np.std(base_vel_x).item()
    
    def get_vel_x(self, robot_index: int) -> float:
        """Compute the x-direction average velocity in robot frame (by time).
        For specified sub environment.
        """
        base_vel = np.array(self.state_log["base_vel_x"])
        return np.mean(base_vel[:, robot_index]).item()

    def get_vel_y_stats(self) -> (float, float):
        """Compute the y-direction average velocity in robot frame (by time).
        Averaged over all sub environments.
        Return the two values, mean velocity by time and its variance.
        """
        base_vel_y = np.array(self.state_log["base_vel_y"])
        base_vel_y = np.mean(base_vel_y, axis=0)
        return np.mean(base_vel_y).item(), np.std(base_vel_y).item()
    
    def get_vel_y(self, robot_index: int) -> float:
        """Compute the y-direction average velocity in robot frame (by time).
        For specified sub environment.
        """
        base_vel = np.array(self.state_log["base_vel_y"])
        return np.mean(base_vel[:, robot_index]).item()

    def get_energy_consume_watts_stats(self) -> (float, float):
        """Compute the average energy consumption (by time) of this test.
        Averaged over all sub environments.
        Return the two values, mean consumed energy by time and its variance.
        """
        energy_consume_watts = np.array(self.state_log["energy_consume"])
        return np.mean(np.mean(energy_consume_watts, axis=0)).item(), np.std(np.mean(energy_consume_watts, axis=0)).item()
    
    def get_energy_consume_watts(self, robot_index: int) -> float:
        """Compute the average energy consumption (by time).
        For specified sub environment.
        """
        energy_consume_watts = np.array(self.state_log["energy_consume"])
        return np.mean(energy_consume_watts[:, robot_index], axis=0).item()
    
    def get_energy_consume_dist_stats(self) -> (float, float):
        """Compute the average energy consumption (by distance) of this test.
        Averaged over all sub environments.
        Return the two values, mean consumed energy by distance and its variance.
        """
        energy_consume = np.sum(np.array(self.state_log["energy_consume"]) * self.dt, axis=0)
        distance_moved = np.absolute(self.state_log["base_pos_x"][-1] - self.state_log["base_pos_x"][0])
        distance_moved[distance_moved <= 0.001] = 0.001
        energy_consume_dist = energy_consume / distance_moved
        return np.mean(energy_consume_dist).item(), np.std(energy_consume_dist).item()
    
    def get_energy_consume_dist(self, robot_index: int) -> float:
        """Compute the average energy consumption (by distance) of this test.
        For specific sub environment.
        """
        energy_consume = np.sum(np.array(self.state_log["energy_consume"]) * self.dt, axis=0)
        distance_moved = np.absolute(self.state_log["base_pos_x"][-1] - self.state_log["base_pos_x"][0])
        energy_consume_dist = energy_consume[robot_index] / max(distance_moved[robot_index], 0.001)
        return energy_consume_dist.item()

    def get_rewards_stats(self) -> (float, float):
        """Get the average reward obtained from this trajectory.
        Averaged over all sub environments.
        """
        rewards = np.array(self.state_log["reward"])
        return np.mean(np.mean(rewards, axis=0)).item(), np.std(np.mean(rewards, axis=0)).item()
    
    def get_rewards(self, robot_index: int) -> float:
        """Get the average reward obtained from this trajectory.
        Averaged over all sub environments.
        """
        rewards = np.array(self.state_log["reward"])
        return np.mean(rewards[:, robot_index]).item()

    def plot_save_states(self, test_name: str, robot_indices: list):
        self._save_logs(test_name)
        for idx in robot_indices:
            self._plot_velocities(test_name, idx)
            self._plot_dof_pos(test_name, idx)
            self._plot_dof_vels(test_name, idx)
            self._plot_dof_accs(test_name, idx)
            self._plot_dof_torques(test_name, idx)
            self._plot_actions(test_name, idx)
            self._plot_gaits(test_name, idx)

    def _save_logs(self, test_name: str):
        """Save the logging dictionary into the model directory.
        File name is "{test_name}_log_data_{self.load_iteration}.pkl".
        Also, save statistics data of energy_consume_watts, energy_consume_dists, rewards, vel_x and vel_y.
        File name is "{test_name}_statistics_{self.load_iteration}.yaml"
        """
        with open(os.path.join(self.model_dir, "analysis", f"{test_name}_log_data_{self.load_iteration}.pkl"), 'wb') as file:
            pickle.dump(self.state_log, file)
            file.close()

        # Save data statistics.
        vel_x_mean, vel_x_std = self.get_vel_x_stats()
        vel_y_mean, vel_y_std = self.get_vel_y_stats()
        energy_consume_watts_mean, energy_consume_watts_std = self.get_energy_consume_watts_stats()
        energy_consume_dist_mean, energy_consume_dist_std = self.get_energy_consume_dist_stats()
        rewards_mean, rewards_std = self.get_rewards_stats()
        dump_dict = {
            'vel_x_mean': vel_x_mean,
            'vel_x_std': vel_x_std,
            'vel_y_mean': vel_y_mean,
            'vel_y_std': vel_y_std,
            'energy_consume_watts_mean': energy_consume_watts_mean,
            'energy_consume_watts_std': energy_consume_watts_std,
            'energy_consume_dist_mean': energy_consume_dist_mean,
            'energy_consume_dist_std': energy_consume_dist_std,
            'rewards_mean': rewards_mean,
            'rewards_std': rewards_std
        }
        with open(os.path.join(self.model_dir, "analysis", f"{test_name}_statistics_{self.load_iteration}.yaml"), 'w') as file:
            yaml.dump(dump_dict, file)
            file.close()

    def _plot_velocities(self, test_name: str, robot_index: int):
        """Plot velocity tracking and save it into the model directory.
        File name is "{test_name}_vel_plot_{self.load_iteration}.png"
        """
        nb_rows = 2
        nb_cols = 3
        fig, axs = plt.subplots(nb_rows, nb_cols)
        fig.set_size_inches(12, 6)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value)*self.dt, len(value))
            break
        log= self.state_log

        # plot base vel x
        a = axs[0, 0]
        if log["base_vel_x"]:
            log_base_vel_x = np.array(log["base_vel_x"])
            a.plot(time, log_base_vel_x[:, robot_index], label='measured')
        if log["command_x"]:
            log_command_x = np.array(log["command_x"])
            a.plot(time, log_command_x[:, robot_index], label='commanded')
        a.set_xlabel('Time [s]', fontsize=8)
        a.set_ylabel('Base lin vel [m/s]', fontsize=8)
        a.set_title('Base velocity x', fontsize=10)
        a.set_ylim([0, 3])
        a.legend()

        # plot base vel y
        a = axs[0, 1]
        if log["base_vel_y"]:
            log_base_vel_y = np.array(log["base_vel_y"])
            a.plot(time, log_base_vel_y[:, robot_index], label='measured')
        if log["command_y"]:
            log_command_y = np.array(log["command_y"])
            a.plot(time, log_command_y[:, robot_index], label='commanded')
        a.set_xlabel('Time [s]', fontsize=8)
        a.set_ylabel('Base lin vel [m/s]', fontsize=8)
        a.set_title('Base velocity y', fontsize=10)
        a.set_ylim([-1, 1])
        a.legend()

        # plot base vel z
        a = axs[0, 2]
        if log["base_vel_z"]:
            log_base_vel_z = np.array(log["base_vel_z"])
            a.plot(time, log_base_vel_z[:, robot_index], label='measured')
        if log["command_z"]:
            log_command_z = np.array(log["command_z"])
            a.plot(time, log_command_z[:, robot_index], label='commanded')
        a.set_xlabel('Time [s]', fontsize=8)
        a.set_ylabel('Base lin vel [m/s]', fontsize=8)
        a.set_title('Base velocity z', fontsize=10)
        a.set_ylim([-1, 1])
        a.legend()

        # plot base vel yaw
        a = axs[1, 0]
        if log["base_vel_yaw"]:
            log_base_vel_yaw = np.array(log["base_vel_yaw"])
            a.plot(time, log_base_vel_yaw[:, robot_index], label='measured')
        if log["command_yaw"]:
            log_command_yaw = np.array(log["command_yaw"])
            a.plot(time, log_command_yaw[:, robot_index], label='commanded')
        a.set_xlabel('Time [s]', fontsize=8)
        a.set_ylabel('Base ang vel [rad/s]', fontsize=8)
        a.set_title('Base velocity yaw', fontsize=10)
        a.set_ylim([-3, 3])
        a.legend()

        # plot base vel pitch
        a = axs[1, 1]
        if log["base_vel_pitch"]:
            log_base_vel_pitch = np.array(log["base_vel_pitch"])
            a.plot(time, log_base_vel_pitch[:, robot_index], label='measured')
        if log["command_pitch"]:
            log_command_pitch = np.array(log["command_pitch"])
            a.plot(time, log_command_pitch[:, robot_index], label='commanded')
        a.set_xlabel('Time [s]', fontsize=8)
        a.set_ylabel('Base ang vel [rad/s]', fontsize=8)
        a.set_title('Base velocity pitch', fontsize=10)
        a.set_ylim([-3, 3])
        a.legend()

        # plot contact roll
        a = axs[1, 2]
        if log["base_vel_roll"]:
            log_base_vel_roll = np.array(log["base_vel_roll"])
            a.plot(time, log_base_vel_roll[:, robot_index], label='measured')
        if log["command_roll"]:
            log_command_roll = np.array(log["command_roll"])
            a.plot(time, log_command_roll[:, robot_index], label='commanded')
        a.set_xlabel('Time [s]', fontsize=8)
        a.set_ylabel('Base ang vel [rad/s]', fontsize=8)
        a.set_title('Base velocity roll', fontsize=10)
        a.set_ylim([-3, 3])
        a.legend()

        fig.tight_layout()
        fig.savefig(os.path.join(self.model_dir, "analysis", f"{test_name}_env_{robot_index}_vel_plot_{self.load_iteration}.png"), dpi=100)
        plt.close()
    
    def _plot_dof_pos(self, test_name: str, robot_index: int):
        nb_rows = 3
        nb_cols = 4
        fig, axs = plt.subplots(nb_rows, nb_cols)
        fig.set_size_inches(16, 6)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value)*self.dt, len(value))
            break
        dof_pos = np.array(self.state_log['dof_pos'])
        time = np.linspace(0, len(value) * self.dt, len(value))

        # Plot the status of every joint.
        for i, j in itertools.product(range(nb_rows), range(nb_cols)):
            dof_id = nb_cols * i + j
            a = axs[i, j]
            a.plot(time, dof_pos[:, robot_index, dof_id])
            a.set_xlabel('Time [s]', fontsize=8)
            a.set_ylabel('Dof Pos [rad]', fontsize=8)
            a.set_title(self.dof_names[dof_id], fontsize=10)

        fig.tight_layout()
        fig.savefig(os.path.join(self.model_dir, "analysis", f"{test_name}_env_{robot_index}_dof_pos_plot_{self.load_iteration}.png"), dpi=100)
        plt.close()
    
    def _plot_dof_vels(self, test_name: str, robot_index: int):
        nb_rows = 3
        nb_cols = 4
        fig, axs = plt.subplots(nb_rows, nb_cols)
        fig.set_size_inches(16, 6)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value)*self.dt, len(value))
            break
        dof_vels = np.array(self.state_log['dof_vel'])
        time = np.linspace(0, len(value) * self.dt, len(value))

        # Plot the status of every joint.
        for i, j in itertools.product(range(nb_rows), range(nb_cols)):
            dof_id = nb_cols * i + j
            a = axs[i, j]
            a.plot(time, dof_vels[:, robot_index, dof_id])
            a.set_xlabel('Time [s]', fontsize=8)
            a.set_ylabel('Dof Vel [rad/s]', fontsize=8)
            a.set_title(self.dof_names[dof_id], fontsize=10)

        fig.tight_layout()
        fig.savefig(os.path.join(self.model_dir, "analysis", f"{test_name}_env_{robot_index}_dof_vel_plot_{self.load_iteration}.png"), dpi=100)
        plt.close()
    
    def _plot_dof_accs(self, test_name: str, robot_index: int):
        nb_rows = 3
        nb_cols = 4
        fig, axs = plt.subplots(nb_rows, nb_cols)
        fig.set_size_inches(16, 6)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value)*self.dt, len(value))
            break
        dof_accs = np.array(self.state_log['dof_acc'])
        time = np.linspace(0, len(value) * self.dt, len(value))

        # Plot the status of every joint.
        for i, j in itertools.product(range(nb_rows), range(nb_cols)):
            dof_id = nb_cols * i + j
            a = axs[i, j]
            a.plot(time, dof_accs[:, robot_index, dof_id])
            a.set_xlabel('Time [s]', fontsize=8)
            a.set_ylabel('Dof Acc [rad/s2]', fontsize=8)
            a.set_title(self.dof_names[dof_id], fontsize=10)

        fig.tight_layout()
        fig.savefig(os.path.join(self.model_dir, "analysis", f"{test_name}_env_{robot_index}_dof_acc_plot_{self.load_iteration}.png"), dpi=100)
        plt.close()
    
    def _plot_dof_torques(self, test_name: str, robot_index: int):
        nb_rows = 3
        nb_cols = 4
        fig, axs = plt.subplots(nb_rows, nb_cols)
        fig.set_size_inches(16, 6)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value)*self.dt, len(value))
            break
        dof_torques = np.array(self.state_log['dof_torque'])
        time = np.linspace(0, len(value) * self.dt, len(value))

        # Plot the status of every joint.
        for i, j in itertools.product(range(nb_rows), range(nb_cols)):
            dof_id = nb_cols * i + j
            a = axs[i, j]
            a.plot(time, dof_torques[:, robot_index, dof_id])
            a.set_xlabel('Time [s]', fontsize=8)
            a.set_ylabel('Dof Torque [N x m]', fontsize=8)
            a.set_title(self.dof_names[dof_id], fontsize=10)

        fig.tight_layout()
        fig.savefig(os.path.join(self.model_dir, "analysis", f"{test_name}_env_{robot_index}_dof_torque_plot_{self.load_iteration}.png"), dpi=100)
        plt.close()
    
    def _plot_actions(self, test_name: str, robot_index: int):
        nb_rows = 3
        nb_cols = 4
        fig, axs = plt.subplots(nb_rows, nb_cols)
        fig.set_size_inches(16, 6)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value)*self.dt, len(value))
            break
        actions_scaled = np.array(self.state_log['action_scaled'])
        time = np.linspace(0, len(value) * self.dt, len(value))

        # Plot the status of every joint.
        for i, j in itertools.product(range(nb_rows), range(nb_cols)):
            dof_id = nb_cols * i + j
            a = axs[i, j]
            a.plot(time, actions_scaled[:, robot_index, dof_id])
            a.set_xlabel('Time [s]', fontsize=8)
            a.set_ylabel('Actions Scaled [rad]', fontsize=8)
            a.set_title(self.dof_names[dof_id], fontsize=10)

        fig.tight_layout()
        fig.savefig(os.path.join(self.model_dir, "analysis", f"{test_name}_env_{robot_index}_dof_action_plot_{self.load_iteration}.png"), dpi=100)
        plt.close()
    
    def _plot_gaits(self, test_name: str, robot_index: int):
        """Plot gait and save it into model directory.
        File name is "{test_name}_gait_plot_{self.load_iteration}.png".
        Save energy consumption, gait frequency, average rewards and step information into a yaml file.
        File name is "{test_name}_gait_info_{self.load_iteration}.yaml".
        """
        fig, axs = plt.subplots(1, 1)
        fig.set_size_inches(16, 6)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value) * self.dt, len(value))
            break
        forces = np.array(self.state_log["contact_forces_z"])
        contact_gaps = []
        for i in range(forces.shape[2]):
            this_contact_ranges = []
            this_contact_steps = []
            for j in range(forces.shape[0]):
                if forces[j, robot_index, i] > 5:
                    this_contact_steps.append(j)
                    if j == forces.shape[0] - 1:
                        this_contact_ranges.append([this_contact_steps[0], this_contact_steps[-1]])
                        this_contact_steps = []
                elif len(this_contact_steps) != 0:
                    this_contact_ranges.append([this_contact_steps[0], this_contact_steps[-1]])
                    this_contact_steps = []
            contact_gaps.append(this_contact_ranges)
        contact_gaps_times = []
        for i in range(forces.shape[2]):
            this_contact_gap_times = []
            for j in range(len(contact_gaps[i])):
                this_contact_gap_times.append([time[contact_gaps[i][j][0]], time[contact_gaps[i][j][1]]])
            contact_gaps_times.append(this_contact_gap_times)

        color_code = ['r', 'y', 'b', 'g']
        for i in range(forces.shape[2]):
            axs.add_patch(plt.Rectangle((time[0], i - 0.2), time[-1] - time[0], 0.4, edgecolor='none', facecolor=color_code[i], alpha=0.1))
            for j in range(len(contact_gaps_times[i])):
                axs.add_patch(plt.Rectangle((contact_gaps_times[i][j][0], i - 0.2), contact_gaps_times[i][j][1] - contact_gaps_times[i][j][0], 0.4, edgecolor='none', facecolor=color_code[i]))
        axs.set_xlim([time[0], time[-1]])
        axs.set_ylim([-0.5, 3.5])
        axs.set_xlabel("Running time (seconds)", weight='bold')
        axs.set_ylabel("Foot name", weight='bold')
        axs.set_yticks(range(len(self.feet_names)), self.feet_names)
        axs.set_title("Gait Graph under Test Speed {:.3f} m/s".format(self.lin_speed), weight='bold')

        fig.tight_layout()
        fig.savefig(os.path.join(self.model_dir, "analysis", f"{test_name}_env_{robot_index}_gait_plot_{self.load_iteration}.png"), dpi=100)
        plt.close()

        # Save the energy and gait informations into YAML file.
        # The velocity tracking error and rewards are averaged in time.
        total_steps = sum([len(contact_gaps[i]) for i in range(len(contact_gaps))])
        step_count = total_steps / len(contact_gaps)

        dump_dict = {
            "step_count": step_count,
            "gait": None,
            'vel_x': self.get_vel_x(robot_index),
            'vel_y': self.get_vel_y(robot_index),
            'energy_consume_watts': self.get_energy_consume_watts(robot_index),
            'energy_consume_dist': self.get_energy_consume_dist(robot_index),
            'rewards': self.get_rewards(robot_index),
        }
        with open(os.path.join(self.model_dir, "analysis", f"{test_name}_env_{robot_index}_gait_info_{self.load_iteration}.yaml"), 'w') as file:
            yaml.dump(dump_dict, file)
            file.close()
