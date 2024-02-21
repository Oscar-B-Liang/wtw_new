import time
import numpy as np
import itertools
from gpu_utils.gputracker import DispatchThread, get_logger


def train_schedule(gpu_list):

    # Experiment started on 26 Jan 2024 night.
    # Fix the training velocity at 1.0 m/s.
    # the energy consumption alpha search from 0.6 to 3.0, with 0.1 gap.
    # the energy sigma is fixed at 700.
    # For each energy consumption, we take the (10 / 7) of its value and penalize vsc range from -0.4 to + 0.4 with 0.1 gap.
    # For example, if alpha = 0.7,
    # We shall search [-0.6, -0.7, -0.8, -0.9, -1.0, -1.1, -1.2, -1.3, -1.4]
    # Logging directory is checkpoints/lb_fixed_vel_terrain_sign_change_energy

    logger = get_logger('checkpoints', 'scheduler.log')
    alphas = np.arange(1.2, 1.5, 0.1).tolist()
    sigmas = np.arange(50, 300, 50).tolist()
    velocities = [0.5, 1.0, 1.5]
    combinations = itertools.product(alphas, sigmas, velocities)
    BASH_COMMAND_LIST = []

    # First layer for loop: alpha value.
    for (alpha, sigma, velocity) in combinations:
        BASH_COMMAND_LIST.append(
            f"python train_min_cmd_energy_only.py --headless --energy {alpha:.1f} --sigma {sigma:.1f} --train_speed {velocity:.1f}"
        )

    dispatch_thread = DispatchThread(
        "search energy weight and sigma",
        BASH_COMMAND_LIST[:],
        logger,
        gpu_m_th=14000,
        gpu_list=gpu_list,
        maxcheck=5
    )

    # Start new Threads
    dispatch_thread.start()
    dispatch_thread.join()

    time.sleep(5)
    logger.info("Exiting Main Thread")


if __name__ == "__main__":
    gpu_list = [3, 4, 5]
    train_schedule(gpu_list)
