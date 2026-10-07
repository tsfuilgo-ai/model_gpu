from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class TikhonovLoss(nn.Module):
    """First-order isotropic Tikhonov regularization for 2D images.

    The input must have shape ``[batch, channels, depth, traces]``. Forward
    differences along depth and traces are penalized with equal weights:

        R(x) = mean((d_depth x)^2) + mean((d_lateral x)^2)

    Computing the directional means separately prevents the unequal numbers
    of valid differences on the two axes from changing their relative weight.
    """

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError(
                "TikhonovLoss expects [batch, channels, depth, traces], "
                f"got {tuple(image.shape)}"
            )
        if image.shape[-2] < 2 or image.shape[-1] < 2:
            raise ValueError(
                "TikhonovLoss requires depth and trace dimensions to be at least 2, "
                f"got {tuple(image.shape[-2:])}"
            )

        depth_gradient = image[:, :, 1:, :] - image[:, :, :-1, :]
        lateral_gradient = image[:, :, :, 1:] - image[:, :, :, :-1]
        return depth_gradient.square().mean() + lateral_gradient.square().mean()


class TrainArgParser:
    def __init__(self) -> None:
        parser = argparse.ArgumentParser("HSFD TomoNet DDP training")

        # Dataset.
        parser.add_argument(
            "--dataset-root",
            type=str,
            default="/root/self_data/fresnel_tomo/GPRInvNet_code/dataset",
        )
        parser.add_argument(
            "--data-dirs",
            nargs="+",
            default=["2D_data_v10", "2D_data_v10_schema7"],
        )
        parser.add_argument(
            "--aligned-data-dir",
            type=str,
            default="v10_samples_velocity_aligned_v1",
            help=(
                "Directory below dataset-root whose samples/ directory is included in "
                "training and used as the validation monitoring set"
            ),
        )
        parser.add_argument("--train-split", choices=("train", "val", "test"), default="train")
        parser.add_argument("--val-split", choices=("train", "val", "test"), default="val")

        # Training.
        parser.add_argument(
            "--save-dir",
            type=str,
            default="/root/self_data/fresnel_tomo/GPRInvNet_code/HSFD/checkpoints",
        )
        parser.add_argument("--resume", type=str, default="")
        parser.add_argument("--epochs", type=int, default=100)
        parser.add_argument("--batch-size", type=int, default=1, help="Per-process tile batch size")
        parser.add_argument("--learning-rate", type=float, default=5e-4)
        parser.add_argument("--loss", choices=("mse", "l1"), default="mse")
        parser.add_argument(
            "--tikhonov-weight",
            type=float,
            default=1e-1,
            help=(
                "Weight of first-order isotropic Tikhonov regularization in "
                "the training objective"
            ),
        )

        # Runtime and DDP.
        parser.add_argument("--num-workers", type=int, default=4)
        parser.add_argument("--seed", type=int, default=666)
        parser.add_argument("--val-interval", type=int, default=1)
        parser.add_argument("--log-interval", type=int, default=10)
        parser.add_argument("--visible-gpus", type=str, default="")

        self.parser = parser

    def parse_args(self) -> argparse.Namespace:
        args = self.parser.parse_args()
        if args.epochs <= 0 or args.batch_size <= 0:
            self.parser.error("--epochs and --batch-size must be positive")
        if args.learning_rate <= 0:
            self.parser.error("--learning-rate must be positive")
        if args.tikhonov_weight < 0:
            self.parser.error("--tikhonov-weight must be non-negative")
        if args.num_workers < 0:
            self.parser.error("--num-workers must be non-negative")
        if args.val_interval <= 0 or args.log_interval <= 0:
            self.parser.error("--val-interval and --log-interval must be positive")
        return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_checkpoint(state: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
