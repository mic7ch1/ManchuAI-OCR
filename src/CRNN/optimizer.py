import torch.optim as optim


def create_optimizer(model_parameters, config):
    if config["optimizer"]["type"] == "AdamW":
        opt_config = config["optimizer"]
        lr = opt_config["lr"]
        if isinstance(lr, str):
            lr = float(lr)
        eps = opt_config["eps"]
        if isinstance(eps, str):
            eps = float(eps)
        return optim.AdamW(
            model_parameters,
            lr=lr,
            betas=opt_config["betas"],
            weight_decay=opt_config["weight_decay"],
            eps=eps,
        )
    else:
        lr = config["learning_rate"]
        if isinstance(lr, str):
            lr = float(lr)
        return optim.AdamW(
            model_parameters,
            lr=lr,
            betas=(0.9, 0.98),
        )
