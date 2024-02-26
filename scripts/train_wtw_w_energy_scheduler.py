import time
import numpy as np
import itertools
from gpu_utils.gputracker import DispatchThread, get_logger


def train_schedule(gpu_list):

    logger = get_logger('checkpoints', 'scheduler.log')
    energy_legs = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
    energys = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
    BASH_COMMAND_LIST = []

    # First layer for loop: alpha value.
    for energy_leg in energy_legs:
        BASH_COMMAND_LIST.append(
            f"python train_wtw_w_energy.py --headless --energy_legs {energy_leg:.1f}"
        )
    for energy in energys:
        BASH_COMMAND_LIST.append(
            f"python train_wtw_w_energy.py --headless --energy {energy:.1f}"
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
    gpu_list = [0, 1]
    train_schedule(gpu_list)
