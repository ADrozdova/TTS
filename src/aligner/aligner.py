import os

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence


class Aligner(nn.Module):
    def __init__(self, root):
        super(Aligner, self).__init__()
        self.root = root

    @torch.no_grad()
    def forward(self, index):
        durations = []
        for idx in index:
            durations.append(torch.from_numpy(
                np.load(os.path.join(self.root, str(idx) + ".npy"))
            ))
        return pad_sequence(durations, batch_first=True)
