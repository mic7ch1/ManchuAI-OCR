import torch.optim as optim


class WarmupScheduler:
    def _init__(self, optimizer, scheduler, warmup_epochs=5, warmup_factor=0.1):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warmup_epochs = warmup_epochs
        self.warmup_factor = warmup_factor
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.current_epoch = 0

    def step(self):
        if self.current_epoch < self.warmup_epochs:
            warmup_lr_factor = self.warmup_factor + (1.0 - self.warmup_factor) * (
                self.current_epoch / self.warmup_epochs
            )
            for i, group in enumerate(self.optimizer.param_groups):
                group["lr"] = self.base_lrs[i] * warmup_lr_factor
        else:
            self.scheduler.step()

        self.current_epoch += 1

    def get_last_lr(self):
        if self.current_epoch < self.warmup_epochs:
            warmup_lr_factor = self.warmup_factor + (1.0 - self.warmup_factor) * (
                self.current_epoch / self.warmup_epochs
            )
            return [lr * warmup_lr_factor for lr in self.base_lrs]
        else:
            return self.scheduler.get_last_lr()

    def state_dict(self):
        return {
            "scheduler": self.scheduler.state_dict(),
            "current_epoch": self.current_epoch,
            "base_lrs": self.base_lrs,
        }

    def load_state_dict(self, state_dict):
        self.scheduler.load_state_dict(state_dict["scheduler"])
        self.current_epoch = state_dict["current_epoch"]
        self.base_lrs = state_dict["base_lrs"]


def create_scheduler(optimizer, config):
    if config["scheduler"]["type"] == "CosineAnnealingWarmRestarts":
        sch_config = config["scheduler"]
        eta_min = sch_config["eta_min"]
        if isinstance(eta_min, str):
            eta_min = float(eta_min)
        base_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=sch_config["T_0"],
            T_mult=sch_config["T_mult"],
            eta_min=eta_min,
            last_epoch=sch_config["last_epoch"],
        )
    else:
        base_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config["scheduler"]["step_size"],
            gamma=config["scheduler"]["gamma"],
        )

    warmup_epochs = config["warmup_epochs"]
    return WarmupScheduler(optimizer, base_scheduler, warmup_epochs=warmup_epochs)
