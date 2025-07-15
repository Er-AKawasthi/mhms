import logging
import math
from torch.optim.lr_scheduler import LambdaLR

logger = logging.getLogger(__name__)


class ConstantLRSchedule(LambdaLR):
    """
    Keeps the learning rate constant throughout training.
    """
    def __init__(self, optimizer, last_epoch=-1):
        logger.info("Using Constant Learning Rate Schedule.")
        super().__init__(optimizer, lambda _: 1.0, last_epoch=last_epoch)


class WarmupConstantSchedule(LambdaLR):
    """
    Linear warmup from 0 to 1 over `warmup_steps`, then keeps it constant.
    """
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        logger.info(f"Using WarmupConstant Schedule with warmup_steps={warmup_steps}")
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        return step / max(1, self.warmup_steps) if step < self.warmup_steps else 1.0


class WarmupLinearSchedule(LambdaLR):
    """
    Linear warmup from 0 to 1, then linear decay to 0.
    """
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        logger.info(f"Using WarmupLinear Schedule with warmup_steps={warmup_steps}, total_steps={t_total}")
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super().__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return step / max(1, self.warmup_steps)
        return max(0.0, (self.t_total - step) / max(1, self.t_total - self.warmup_steps))


class WarmupCosineSchedule(LambdaLR):
    """
    Linear warmup followed by cosine decay.
    `cycles` determines how many cosine waves over the decay period.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=0.5, last_epoch=-1):
        logger.info(f"Using WarmupCosine Schedule with warmup_steps={warmup_steps}, total_steps={t_total}, cycles={cycles}")
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super().__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return step / max(1, self.warmup_steps)
        progress = (step - self.warmup_steps) / max(1, self.t_total - self.warmup_steps)
        return max(0.0, 0.5 * (1. + math.cos(math.pi * 2 * self.cycles * progress)))
