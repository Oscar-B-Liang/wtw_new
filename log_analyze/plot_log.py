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
    gait_plot_path = os.path.join(log_dir_path, "gait.png")

    contacts = []
    time_stamps = []
    for data_point in logged_data['hardware_closed_loop'][1]:
        contacts.append(data_point['contact_state'][0])
        time_stamps.append(data_point['time'])
    contacts = np.array(contacts)

    middle_idx = int(len(time_stamps) / 2)
    contacts = contacts[middle_idx - 50: middle_idx + 50]
    time_stamps = time_stamps[middle_idx - 50: middle_idx + 50]
    
    contact_gaps = []
    for i in range(contacts.shape[1]):
        this_contact_ranges = []
        this_contact_steps = []
        for j in range(contacts.shape[0]):
            if contacts[j, i]:
                this_contact_steps.append(j)
                if j == contacts.shape[0] - 1:
                    this_contact_ranges.append([this_contact_steps[0], this_contact_steps[-1]])
                    this_contact_steps = []
            elif len(this_contact_steps) != 0:
                this_contact_ranges.append([this_contact_steps[0], this_contact_steps[-1]])
                this_contact_steps = []
        contact_gaps.append(this_contact_ranges)
    
    contact_gaps_times = []
    for i in range(contacts.shape[1]):
        this_contact_gap_times = []
        for j in range(len(contact_gaps[i])):
            this_contact_gap_times.append([time_stamps[contact_gaps[i][j][0]], time_stamps[contact_gaps[i][j][1]]])
        contact_gaps_times.append(this_contact_gap_times)

    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(16, 6)
    color_code = ['r', 'y', 'b', 'g']
    feet_names = ['RR', 'RL', 'FR', 'FL']
    for i in range(contacts.shape[1]):
        axs.add_patch(plt.Rectangle((time_stamps[0], i - 0.2), time_stamps[-1] - time_stamps[0], 0.4, edgecolor='none', facecolor=color_code[i], alpha=0.1))
        for j in range(len(contact_gaps_times[i])):
            axs.add_patch(plt.Rectangle((contact_gaps_times[i][j][0], i - 0.2), contact_gaps_times[i][j][1] - contact_gaps_times[i][j][0], 0.4, edgecolor='none', facecolor=color_code[i]))
    axs.set_xlim([time_stamps[0], time_stamps[-1]])
    axs.set_ylim([-0.5, 3.5])
    axs.set_xlabel("Running time (seconds)", weight='bold')
    axs.set_ylabel("Foot name", weight='bold')
    axs.set_yticks(range(len(feet_names)), feet_names)
    axs.set_title("Gait Graph", weight='bold')

    fig.tight_layout()
    fig.savefig(gait_plot_path, dpi=100)
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
