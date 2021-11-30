from torch import nn


class FastSpeechLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, batch):
        len = min(batch.durations.shape[-1], batch.duration_pred.shape[-1])
        duration_loss = self.mse_loss(batch.durations, batch.duration_pred)

        len = min(batch.melspec.shape[-1], batch.melspec_prediction.shape[-1])
        mel_loss = self.mse_loss(batch.melspec[:, :, :len], batch.melspec_pred[:, :, :len])
        loss = mel_loss + duration_loss
        return loss, mel_loss, duration_loss
