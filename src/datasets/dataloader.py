import os
import random

import torch
from torch.utils.data import Dataset, get_worker_info

import pandas as pd
from PIL import Image


AVAILABLE_DTYPES = {
    'path': 'string',
    'lat': 'float',
    'lon': 'float',
    'unix_ts': 'int',
    'year': 'int',
    'month': 'int',
    'day': 'int',
    'hour': 'int',
    'minute': 'int',
    'second': 'int',
}


class retvals:
    GEO = ['path', 'lat', 'lon']
    GEOTEMP = ['path', 'lat', 'lon', 'unix_ts', 'month', 'day', 'hour', 'minute', 'second']


class GeoTemporalDataset(Dataset):
    def __init__(self, metadata_path, imgs_path, transform, fields=retvals.GEOTEMP, sample=1.0, seed=23):
        self.metadata_path = metadata_path
        self.imgs_path = imgs_path
        self.transform = transform
        self.fields = fields
        self.sample = sample
        self.seed = seed
        self.n_samples = 0

        # Base dtypes for the fields you always want
        base_dtype = {col: AVAILABLE_DTYPES[col] for col in fields}
        dtype = dict(base_dtype)

        def usecols(col: str) -> bool:
            return (col in base_dtype)

        self.dataset = pd.read_csv(
            metadata_path,
            dtype=dtype,
            usecols=usecols
        )

        # Sample rows
        if sample < 1.0:
            self.dataset = self.dataset.sample(frac=sample, random_state=self.seed)

    def _prepare_metadata(self, row):
        out = {}
        if 'path' in self.fields:
            out['path'] = row['path']
        if 'lat' in self.fields and 'lon' in self.fields:
            out['gps'] = torch.tensor([row['lat'], row['lon']], dtype=torch.float32)
        if 'unix_ts' in self.fields:
            out['time'] = torch.tensor([row['month'], row['day'], row['hour'], row['minute'], row['second']], dtype=torch.int64)
        return out

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = self.dataset.iloc[index]

        # Load image
        img_path = os.path.join(self.imgs_path, row['path'])
        img = Image.open(img_path)
        img = self.transform(img)

        # Get metadata
        metadata = self._prepare_metadata(row)

        return img, metadata


class SkyFinderDataset(GeoTemporalDataset):
    """
    This dataset yields batches balanced across camera IDs.
    It should only be used during training.
    """
    def __init__(self, *args, **kwargs):
        super(SkyFinderDataset, self).__init__(*args, **kwargs)

        # Per-worker RNG cache: {worker_id (or None): random.Random}
        self._rng_by_worker = {}

        # Find camera IDs
        self.dataset['cam_id'] = self.dataset['path'].apply(lambda x: x.split('/')[0])

        # Split dataset into cameras
        self.datasets = {}
        for cam_id in self.dataset['cam_id'].unique():
            cam_dataset = (
                self.dataset[self.dataset['cam_id'] == cam_id]
                .reset_index(drop=False)
                .rename(columns={"index": "_rowidx"})   # (9) rename preserved index column
            )
            self.datasets[cam_id] = cam_dataset

        # (7) precompute camera id list once
        self.cam_ids = list(self.datasets.keys())

    def _get_worker_rng(self) -> random.Random:
        """
        Return a deterministic RNG for the current DataLoader worker.
        - In the main process (no worker), seed with self.seed.
        - In a worker process, seed with that worker's PyTorch-provided seed.
        """
        wi = get_worker_info()
        worker_id = None if wi is None else wi.id

        if worker_id in self._rng_by_worker:
            return self._rng_by_worker[worker_id]

        if wi is None:
            seed = int(self.seed)
        else:
            # wi.seed is a large, distinct seed per worker and per epoch (when DataLoader resets workers)
            seed = int(wi.seed)

        rng = random.Random(seed)
        self._rng_by_worker[worker_id] = rng
        return rng

    def __getitem__(self, _):
        rng = self._get_worker_rng()  # (6) per-worker RNG

        # Randomly select a camera (uses precomputed list)
        cam_id = rng.choice(self.cam_ids)
        cam_dataset = self.datasets[cam_id]

        # Randomly select an image from that camera.
        # Use a deterministic pandas random_state derived from our RNG.
        pandas_seed = rng.getrandbits(32)
        index = cam_dataset.sample(n=1, random_state=pandas_seed).iloc[0]['_rowidx']

        return super(SkyFinderDataset, self).__getitem__(index)
