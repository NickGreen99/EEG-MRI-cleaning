import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt
import random, time
from pathlib import Path
from torch.utils.data import IterableDataset

_SOS_HP = butter(4, 100, "highpass", fs=1000, output="sos")

class EEGWindowStream(IterableDataset):
    """
    Each sample
        x : (2, win)  =  [r0, alpha*artefact]
        y : (1, win)  =  clean EEG target
    """
    def __init__(self, split: str, win: int = 512, hop: int = 200, alpha: tuple[float, float] = (0.5, 2.0),
        examples_per_epoch: int = 32_000,   # 1 k batches @ bs=32
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be 'train', 'val', or 'test'")

        root = Path("split_data") / split
        self.mri_paths = sorted(root.glob("*_mri.npy"))
        if not self.mri_paths:
            raise RuntimeError(f"No *_mri.npy files found in {root}")

        self.win   = int(win)
        self.hop   = int(hop)
        self.alpha = alpha
        self.size  = int(examples_per_epoch)

    # --------------------------------------------------------
    #  IterableDataset protocol
    # --------------------------------------------------------
    def __len__(self):
        # examples, not batches
        return self.size

    def _hp(self, x: np.ndarray) -> np.ndarray:
        """High-pass filter (zero-phase)."""
        return sosfiltfilt(_SOS_HP, x, axis=-1, padtype="odd", padlen=12)

    def __iter__(self):
        # one RNG per worker -- keeps each worker’s sampling independent
        info = torch.utils.data.get_worker_info()
        seed = int(time.time() * 1e6) % 2**31
        rng  = np.random.default_rng(seed if info is None else seed + info.id)

        for _ in range(self.size):
            mri_path   = rng.choice(self.mri_paths)
            clean_path = mri_path.with_name(mri_path.name.replace("_mri", "_clean"))

            mri  = np.load(mri_path,   mmap_mode="r")  # (26, Tm)
            cln  = np.load(clean_path, mmap_mode="r")  # (26, Tc)

            lead = rng.integers(26)

            # clean slice
            start_c = rng.integers(0, cln.shape[1] - self.win)
            clean   = cln[lead, start_c : start_c + self.win]

            # artefact slice
            start_a = rng.integers(0, mri.shape[1] - self.win)
            art_ref = self._hp(mri[lead, start_a : start_a + self.win])

            alpha   = rng.uniform(*self.alpha)
            r0      = clean + alpha * art_ref

            X = np.stack([r0, alpha * art_ref]).astype(np.float32)
            Y = clean[np.newaxis, :].astype(np.float32)

            yield torch.from_numpy(X), torch.from_numpy(Y)
