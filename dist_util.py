import logging
import torch

logger = logging.getLogger(__name__)

def get_rank():
    """
    Returns the rank of the current process in distributed training.
    If not using distributed, returns 0.
    """
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        logger.debug("torch.distributed not initialized, default rank=0")
        return 0
    return torch.distributed.get_rank()


def get_world_size():
    """
    Returns the total number of processes in distributed training.
    If not using distributed, returns 1.
    """
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        logger.debug("torch.distributed not initialized, default world_size=1")
        return 1
    return torch.distributed.get_world_size()


def is_main_process():
    """
    Checks if this is the main (rank 0) process.
    """
    return get_rank() == 0


def format_step(step):
    """
    Nicely formats a training step tuple for logging.
    Example:
        (epoch, iteration, validation_iter) -> "Training Epoch: X Training Iteration: Y Validation Iteration: Z"
    """
    if isinstance(step, str):
        return step

    parts = []
    if len(step) > 0:
        parts.append(f"Training Epoch: {step[0]}")
    if len(step) > 1:
        parts.append(f"Training Iteration: {step[1]}")
    if len(step) > 2:
        parts.append(f"Validation Iteration: {step[2]}")

    return " ".join(parts)
