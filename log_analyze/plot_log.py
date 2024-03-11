import argparse
from matplotlib import pyplot as plt
import numpy as np
import os
import pickle

from go1_gym import MINI_GYM_ROOT_DIR


def plot_lin_commands(log_dir_path: str, logged_data: dict):

    plot_path = os.path.join(log_dir_path, "lin_cmd.png")

    obs_x_vels = []
    cmd_x_vels = []
    time_stamps = []
    start_time = logged_data['hardware_closed_loop'][1][0]['time']
    end_time = logged_data['hardware_closed_loop'][1][-1]['time']
    for data_point in logged_data['hardware_closed_loop'][1]:
        obs_x_vels.append(data_point['observation']['obs'][0, 3])
        cmd_x_vels.append(data_point['body_linear_vel_cmd'][0, 0])
        time_stamps.append(data_point['time'])
    fig, ax = plt.subplots(1, 1)
    ax.plot(time_stamps, obs_x_vels, label="Observed")
    ax.plot(time_stamps, cmd_x_vels, label="Commanded")
    ax.set_title(f"Linear Velocity Commands (Time Duration {end_time - start_time:.2f} s)")
    ax.set_xlim(start_time, end_time)
    ax.set_ylim(-5, 5)
    ax.set_ylabel("Command Velocity (m/s)")
    ax.set_xlabel("Time (s)")
    ax.legend()
    plt.savefig(plot_path)
    plt.close()


def plot_yaw_commands(log_dir_path: str, logged_data: dict):
    cmd_plot_path = os.path.join(log_dir_path, "yaw_cmd.png")

    cmd_x_vels = []
    time_stamps = []
    start_time = logged_data['hardware_closed_loop'][1][0]['time']
    end_time = logged_data['hardware_closed_loop'][1][-1]['time']
    for data_point in logged_data['hardware_closed_loop'][1]:
        cmd_x_vels.append(data_point['observation']['obs'][0, 5])
        time_stamps.append(data_point['time'])
    fig, ax = plt.subplots(1, 1)
    ax.plot(time_stamps, cmd_x_vels)
    ax.set_title(f"Angular Velocity Commands (Time Duration {end_time - start_time:.2f} s)")
    ax.set_xlim(start_time, end_time)
    # ax.set_ylim(-5, 5)
    ax.set_ylabel("Command Velocity (rad/s)")
    ax.set_xlabel("Time (s)")
    plt.savefig(cmd_plot_path)
    plt.close()


def plot_contact_states(log_dir_path: str, logged_data: dict):
    cmd_plot_path = os.path.join(log_dir_path, "gait.png")

    feet_touch = []
    time_stamps = []
    start_time = logged_data['hardware_closed_loop'][1][0]['time']
    end_time = logged_data['hardware_closed_loop'][1][-1]['time']
    for data_point in logged_data['hardware_closed_loop'][1]:
        feet_touch.append(data_point['contact_state'][0])
        time_stamps.append(data_point['time'])
    feet_touch = np.array(feet_touch)

    fig, ax = plt.subplots(4, 1, sharex=True)
    for i in range(4):
        ax[i].plot(time_stamps[500: 600], feet_touch[500: 600, i])
        # ax[i].set_xlim(start_time, end_time)
    ax[0].set_title(f"Gait Plot (Time Duration {end_time - start_time:.2f} s)")
    ax[3].set_xlabel("Time (s)")
    plt.savefig(cmd_plot_path)
    plt.close()


def plot_log():

    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, required=True)
    args = parser.parse_args()

    log_dir_path = os.path.join(MINI_GYM_ROOT_DIR, args.logdir)
    log_file_path = os.path.join(log_dir_path, "log.pkl")

    # Load the pickle data dictionary.
    with open(log_file_path, 'rb') as file:
        logged_data: dict = pickle.load(file)
        file.close()

    # Plot the velocity commands.
    plot_lin_commands(log_dir_path, logged_data)

    # Plot the joint motions.
    plot_yaw_commands(log_dir_path, logged_data)

    # Plot gait.
    plot_contact_states(log_dir_path, logged_data)


if __name__ == "__main__":
    plot_log()
