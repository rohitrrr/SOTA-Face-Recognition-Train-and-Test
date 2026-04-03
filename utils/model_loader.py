from os import path
import torch
from .utils import get_time


def save_state(model, optimizer, config, accuracy, step=0, model_only=False, head=None, epoch=0):
    save_path = config.model_path

    if head is None:
        model_weights = model.module.model.state_dict()
        head_weights = model.module.head.state_dict()
    else:
        model_weights = model.module.state_dict()
        head_weights = head.state_dict()

    torch.save(
        model_weights,
        path.join(
            save_path,
            "model_{}_accuracy;{:.4f}_step;{}.pth".format(get_time(), accuracy, step),
        ),
    )

    if not model_only:
        torch.save(
            head_weights,
            path.join(
                save_path,
                "head_{}_accuracy;{:.4f}_step;{}.pth".format(
                    get_time(), accuracy, step
                ),
            ),
        )
        torch.save(
            optimizer.state_dict(),
            path.join(
                save_path,
                "optimizer_{}_accuracy;{:.4f}_step;{}.pth".format(
                    get_time(), accuracy, step
                ),
            ),
        )

    # Save a single checkpoint file for easy resume
    torch.save(
        {
            "model": model_weights,
            "head": head_weights,
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "accuracy": accuracy,
        },
        path.join(save_path, "checkpoint_latest.pth"),
    )
