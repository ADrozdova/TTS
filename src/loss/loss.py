from torch import nn
import torch.nn.functional as F

class FastSpeechLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, batch):
        duration_loss = self.length_mse(batch.durations, batch.duration_pred)

        len = min(batch.melspec.shape[-1], batch.melspec_prediction.shape[-2])
        mel_loss = self.mel_spec_mse(batch.melspec[:, :, :len], batch.melspec_pred[:, :, :len])
        loss = mel_loss + duration_loss
        return loss, mel_loss, duration_loss
