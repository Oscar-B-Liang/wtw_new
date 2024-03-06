import time
import numpy as np
import itertools
from gpu_utils.gputracker import DispatchThread, get_logger


def train_schedule(gpu_list):

    logger = get_logger('checkpoints', 'scheduler.log')
    BASH_COMMAND_LIST = []

    # First layer for loop: alpha value.
    for seed in [7, 31, 42, 60]:
        BASH_COMMAND_LIST.append(
            f"python train_wtw_w_energy.py --headless --energy 0.7 --seed {seed}"
        )

    dispatch_thread = DispatchThread(
        "search energy weight and sigma",
        BASH_COMMAND_LIST[:],
        logger,
        gpu_m_th=2000,
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
