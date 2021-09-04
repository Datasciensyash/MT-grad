import typing as tp

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class MTDataset(Dataset):
    def __init__(self, filename: str, use_log: bool = False):
        self.data_array = pd.read_csv(filename).values
        self.use_log = use_log

    def format_row(
            self, row: np.ndarray
    ) -> tp.Tuple[tp.Tuple[np.ndarray, np.ndarray], tp.Tuple[np.ndarray, np.ndarray], np.ndarray]:
        # Too much hard coding, should remove magic ints somehow.
        rho_yx = row[:1638].reshape(-1, 126)
        rho_xy = row[1638:1638 * 2].reshape(-1, 126)
        phi_yx = row[1638 * 2:1638 * 3].reshape(-1, 126)
        phi_xy = row[1638 * 3:1638 * 4].reshape(-1, 126)
        target = row[1638 * 4:]
        return (rho_yx, rho_xy), (phi_yx, phi_xy), target

    def __getitem__(self, index: int):
        row = self.data_array[index, :-1]
        (rho_yx, rho_xy), (phi_yx, phi_xy), target = self.format_row(row)
        rho_yx, rho_xy = torch.Tensor(rho_yx), torch.Tensor(rho_xy)
        phi_yx, phi_xy = torch.Tensor(phi_yx), torch.Tensor(phi_xy)
        target = torch.Tensor(target)
        return (rho_yx, rho_xy), (phi_yx, phi_xy), np.log(target)

    def __len__(self):
        return self.data_array.shape[0]
