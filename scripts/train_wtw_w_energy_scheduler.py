import time
import numpy as np
import itertools
from gpu_utils.gputracker import DispatchThread, get_logger


def train_schedule(gpu_list):

    logger = get_logger('checkpoints', 'scheduler.log')
    energys = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
    speeds = [0.5, 1.0, 1.5, 2.0]
    combinations = itertools.combinations(energys, speeds)
    BASH_COMMAND_LIST = []

    # First layer for loop: alpha value.
    for energy, speed in combinations:
        BASH_COMMAND_LIST.append(
            f"python train_wtw_w_energy.py --headless --energy {energy:.1f} --train_speed {speed:.1f} --orientation 5.0"
        )

    dispatch_thread = DispatchThread(
        "search energy weight and sigma",
        BASH_COMMAND_LIST[:],
        logger,
        gpu_m_th=5000,
        gpu_list=gpu_list,
        maxcheck=10
    )

    # Start new Threads
    dispatch_thread.start()
    dispatch_thread.join()

    time.sleep(5)
    logger.info("Exiting Main Thread")


if __name__ == "__main__":
    gpu_list = [0, 1]
    train_schedule(gpu_list)
