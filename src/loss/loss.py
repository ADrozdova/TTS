from torch import nn


class FastSpeechLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, batch):
        l = min(batch.durations.shape[-1], batch.duration_pred.shape[-1])
        duration_loss = self.mse_loss(batch.durations[:, :l], batch.duration_pred[:, :l])

        l = min(batch.melspec.shape[-1], batch.melspec_pred.shape[-1])
        mel_loss = self.mse_loss(batch.melspec[:, :, :l], batch.melspec_pred[:, :, :l])
        loss = mel_loss + duration_loss
        return loss, mel_loss, duration_loss
