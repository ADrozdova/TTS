import torch.nn.functional as F
from torch import nn


class FastSpeechLossWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, true_mel, pred_mel, true_d, pred_d):
        duration_loss = self.mse_loss(true_d, pred_d)

        len = min(true_mel.shape[-1], pred_mel.shape[-1])
        mel_loss = self.mel_loss(true_mel[:, :, :len], pred_mel[:, :, :len])
        return mel_loss, duration_loss
