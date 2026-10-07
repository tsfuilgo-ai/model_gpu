from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


DEFAULT_DATASET_ROOT = Path("/data/1/lwt/archive")
DEFAULT_DATA_DIRS = ("2D_data_v10", "2D_data_v10_schema7")
DEFAULT_ALIGNED_DATA_DIR = "v10_samples_velocity_aligned_v1"


@dataclass(frozen=True)
class SampleInfo:
    path: Path
    scale: float
    water_depth: float


class SeismicTileDataset(Dataset):
    """Load each full seismic sample as sixteen overlapping tile samples.

    Dataset composition is intentionally configured for the overfitting run:

    * ``train`` loads train/val/test from every legacy ``data_dirs`` entry and
      also every sample from ``aligned_data_dir/samples``.
    * ``val`` loads only every sample from ``aligned_data_dir/samples``.  This
      is the same measured-data subset that is present in training and is
      therefore a validation monitor for the overfitting run, not an
      independent generalization estimate.  ``test`` currently resolves to the
      same aligned subset, but is not used by the training script.
    """

    FORWARD_MIN = -1.21907763e-6
    FORWARD_MAX = 1.21907763e-6
    VELOCITY_MIN = 1000.0
    VELOCITY_MAX = 2800.0
    WATER_DEPTH_MIN = -2.0
    WATER_DEPTH_MAX = 37.0

    FULL_WIDTH = 5120
    NUM_TILES = 16
    CORE_WIDTH = 320
    HALO = 17
    INPUT_WIDTH = CORE_WIDTH + 2 * HALO
    TIME_SAMPLES = 3000
    DEPTH_SAMPLES = 880

    def __init__(
        self,
        split: str,
        dataset_root: str | Path = DEFAULT_DATASET_ROOT,
        data_dirs: Sequence[str] = DEFAULT_DATA_DIRS,
        aligned_data_dir: str = DEFAULT_ALIGNED_DATA_DIR,
        full_width: int = 5120,
        num_tiles: int = 16,
        core_width: int = 320,
        halo: int = 17,
        time_samples: int = 3000,
        depth_samples: int = 880,
        forward_min: float = -1.21907763e-6,
        forward_max: float = 1.21907763e-6,
        velocity_min: float = 1000.0,
        velocity_max: float = 2800.0,
        water_depth_min: float = -2.0,
        water_depth_max: float = 37.0,
    ) -> None:
        super().__init__()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be train, val or test, got {split!r}")
        if full_width != num_tiles * core_width:
            raise ValueError("full_width must equal num_tiles * core_width")
        if halo < 0 or time_samples <= 0 or depth_samples <= 0:
            raise ValueError("halo must be non-negative and sample dimensions must be positive")
        if not (forward_min < forward_max):
            raise ValueError("forward_min must be smaller than forward_max")
        if not (velocity_min < velocity_max):
            raise ValueError("velocity_min must be smaller than velocity_max")
        if not (water_depth_min < water_depth_max):
            raise ValueError("water_depth_min must be smaller than water_depth_max")

        self.split = split
        self.dataset_root = Path(dataset_root)
        self.data_dirs = tuple(data_dirs)
        self.aligned_data_dir = aligned_data_dir

        self.FULL_WIDTH = full_width
        self.NUM_TILES = num_tiles
        self.CORE_WIDTH = core_width
        self.HALO = halo
        self.INPUT_WIDTH = core_width + 2 * halo
        self.TIME_SAMPLES = time_samples
        self.DEPTH_SAMPLES = depth_samples
        self.FORWARD_MIN = forward_min
        self.FORWARD_MAX = forward_max
        self.VELOCITY_MIN = velocity_min
        self.VELOCITY_MAX = velocity_max
        self.WATER_DEPTH_MIN = water_depth_min
        self.WATER_DEPTH_MAX = water_depth_max

        self.samples = self._find_samples()
        if not self.samples:
            raise RuntimeError(
                f"No complete samples found for split={split!r} under {self.dataset_root}"
            )

    def _find_samples(self) -> list[SampleInfo]:
        samples: list[SampleInfo] = []

        if self.split == "train":
            for data_dir in self.data_dirs:
                for legacy_split in ("train", "val", "test"):
                    split_dir = self.dataset_root / data_dir / "samples" / legacy_split
                    samples.extend(self._load_samples_from_directory(split_dir))

        aligned_samples_dir = self.dataset_root / self.aligned_data_dir / "samples"
        samples.extend(self._load_samples_from_directory(aligned_samples_dir))

        return samples

    def _load_samples_from_directory(self, samples_dir: Path) -> list[SampleInfo]:
        if not samples_dir.is_dir():
            raise FileNotFoundError(f"Dataset samples directory does not exist: {samples_dir}")

        samples: list[SampleInfo] = []
        for sample_dir in sorted(path for path in samples_dir.iterdir() if path.is_dir()):
            required_files = [
                sample_dir / "forward.npy",
                sample_dir / "vp.npy",
                sample_dir / "meta.json",
            ]
            missing = [path.name for path in required_files if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    f"Sample {sample_dir} is missing required file(s): {', '.join(missing)}"
                )

            self._validate_array_shapes(sample_dir)
            with (sample_dir / "meta.json").open("r", encoding="utf-8") as file:
                metadata = json.load(file)

            scale = float(metadata["normalization"]["scale"])
            water_depth = float(metadata["seafloor_depth_m"]["mean"]) - 3.0
            samples.append(SampleInfo(sample_dir, scale, water_depth))
        return samples

    def _validate_array_shapes(self, sample_dir: Path) -> None:
        forward = np.load(sample_dir / "forward.npy", mmap_mode="r")
        velocity = np.load(sample_dir / "vp.npy", mmap_mode="r")

        valid_forward_shape = (
            forward.ndim == 3
            and forward.shape[0] == 1
            and forward.shape[1] >= self.TIME_SAMPLES
            and forward.shape[2] == self.FULL_WIDTH
        )
        if not valid_forward_shape:
            raise ValueError(
                f"Expected forward.npy shape [1, >= {self.TIME_SAMPLES}, {self.FULL_WIDTH}], "
                f"got {forward.shape} in {sample_dir}"
            )
        if velocity.shape != (self.DEPTH_SAMPLES, self.FULL_WIDTH):
            raise ValueError(
                f"Expected vp.npy shape ({self.DEPTH_SAMPLES}, {self.FULL_WIDTH}), "
                f"got {velocity.shape} in {sample_dir}"
            )

    @property
    def num_full_samples(self) -> int:
        return len(self.samples)

    def __len__(self) -> int:
        return self.num_full_samples * self.NUM_TILES

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        sample_index, tile_index = divmod(index, self.NUM_TILES)
        sample = self.samples[sample_index]

        core_start = tile_index * self.CORE_WIDTH
        core_end = core_start + self.CORE_WIDTH
        input_start = core_start - self.HALO
        input_end = core_end + self.HALO

        forward = np.load(sample.path / "forward.npy", mmap_mode="r")
        data = self._extract_forward_tile(forward, input_start, input_end)
        data = data * np.float32(sample.scale)
        data = self._minmax_normalize(data, self.FORWARD_MIN, self.FORWARD_MAX)

        # Model input order is [channel, trace, time].
        data = np.ascontiguousarray(data.transpose(0, 2, 1))

        velocity = np.load(sample.path / "vp.npy", mmap_mode="r")
        target = self._extract_velocity_tile(velocity, input_start, input_end)
        target = self._minmax_normalize(target, self.VELOCITY_MIN, self.VELOCITY_MAX)
        target = np.ascontiguousarray(target[None, ...])

        water_depth = self._normalize_scalar(
            sample.water_depth,
            self.WATER_DEPTH_MIN,
            self.WATER_DEPTH_MAX,
        )

        return {
            "data": torch.from_numpy(data),
            "velocity": torch.from_numpy(target),
            "water_depth": torch.tensor(water_depth, dtype=torch.float32),
            "sample_index": torch.tensor(sample_index, dtype=torch.long),
            "tile_index": torch.tensor(tile_index, dtype=torch.long),
        }

    def _extract_forward_tile(
        self,
        forward: np.ndarray,
        start: int,
        end: int,
    ) -> np.ndarray:
        source_start = max(start, 0)
        source_end = min(end, self.FULL_WIDTH)
        tile = np.asarray(
            forward[:, : self.TIME_SAMPLES, source_start:source_end],
            dtype=np.float32,
        )

        pad_left = max(0, -start)
        pad_right = max(0, end - self.FULL_WIDTH)
        if pad_left or pad_right:
            tile = np.pad(
                tile,
                ((0, 0), (0, 0), (pad_left, pad_right)),
                mode="edge",
            )

        if tile.shape != (1, self.TIME_SAMPLES, self.INPUT_WIDTH):
            raise RuntimeError(f"Unexpected forward tile shape: {tile.shape}")
        return tile

    def _extract_velocity_tile(
        self,
        velocity: np.ndarray,
        start: int,
        end: int,
    ) -> np.ndarray:
        source_start = max(start, 0)
        source_end = min(end, self.FULL_WIDTH)
        tile = np.asarray(velocity[:, source_start:source_end], dtype=np.float32)

        pad_left = max(0, -start)
        pad_right = max(0, end - self.FULL_WIDTH)
        if pad_left or pad_right:
            tile = np.pad(tile, ((0, 0), (pad_left, pad_right)), mode="edge")

        if tile.shape != (self.DEPTH_SAMPLES, self.INPUT_WIDTH):
            raise RuntimeError(f"Unexpected velocity tile shape: {tile.shape}")
        return tile

    @staticmethod
    def _minmax_normalize(array: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
        normalized = (array - minimum) / (maximum - minimum)
        return np.clip(normalized, 0.0, 1.0).astype(np.float32, copy=False)

    @staticmethod
    def _normalize_scalar(value: float, minimum: float, maximum: float) -> float:
        normalized = (value - minimum) / (maximum - minimum)
        return float(np.clip(normalized, 0.0, 1.0))


if __name__ == "__main__":
    dataset = SeismicTileDataset(split="train")
    sample = dataset[0]

    print("full samples:", dataset.num_full_samples)
    print("tile samples:", len(dataset))
    print("data:", tuple(sample["data"].shape))
    print("velocity:", tuple(sample["velocity"].shape))
    print("water depth:", float(sample["water_depth"]))
