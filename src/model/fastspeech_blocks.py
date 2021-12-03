import math

import torch
from torch import Tensor
from torch import nn

from src.model.attention import MultiHeadAttention

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ConvFFT(nn.Module):
    def __init__(self, in_size, hidden, kernels, padding, dropout):
        super(ConvFFT, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_size, hidden, kernel_size=kernels[0], padding=padding[0]),
            nn.ReLU(),
            nn.Conv1d(hidden, in_size, kernel_size=kernels[1], padding=padding[1]),
            nn.Dropout(dropout)
        )
        self.ln = nn.LayerNorm(in_size)

    def forward(self, x):
        out = self.ln(x)
        out = self.layers(out.transpose(-1, -2))
        return out.transpose(-1, -2) + x


class FFTBlock(nn.Module):
    def __init__(self, in_size, n_heads, hidden, kernels, padding, attn_dropout=0.1, dropout=0.1):
        super(FFTBlock, self).__init__()
        self.attn = MultiHeadAttention(n_heads, in_size, attn_dropout)
        self.conv = ConvFFT(in_size, hidden, kernels, padding, dropout)
        self.ln = nn.LayerNorm(in_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        out = self.ln(x)
        out, attn = self.attn(out, out, out, attn_mask)
        out = x + self.dropout(out)
        out = self.conv(out)
        return out, attn


class FFT(nn.Module):
    def __init__(self, n_layers, in_size, n_heads, hidden, kernels, padding, attn_dropout=0.1, dropout=0.1):
        super().__init__()
        self.fft_blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.fft_blocks.append(FFTBlock(in_size, n_heads, hidden, kernels, padding, attn_dropout=attn_dropout,
                                            dropout=dropout))

    def forward(self, x, attn_mask=None):
        for layer in self.fft_blocks:
            x, _ = layer(x, attn_mask)
        return x


# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class ConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ConvTranspose, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding)

    def forward(self, x):
        return self.conv(x.transpose(-1, -2)).transpose(-1, -2)


class DurationPredictor(nn.Module):
    def __init__(self, in_size, hidden, kernel, dropout=0.1):
        super(DurationPredictor, self).__init__()
        self.layers = nn.Sequential(
            ConvTranspose(in_size, hidden, kernel, kernel // 2),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            ConvTranspose(hidden, hidden, kernel, kernel // 2),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        out = self.layers(x).squeeze(-1)
        return out


class LengthRegulator(nn.Module):
    def __init__(self, in_size, hidden, kernel, dropout=0.1):
        super(LengthRegulator, self).__init__()
        self.dp = DurationPredictor(in_size, hidden, kernel, dropout=dropout)

    def LR(self, x, dur_pred):
        output = torch.zeros(dur_pred.shape[0], dur_pred.shape[-1], round(dur_pred.sum(-1).max().item()))
        for i in range(dur_pred.shape[0]):
            start = 0
            finish = 0
            for j in range(dur_pred.shape[-1]):
                diff = round(dur_pred[i][j].item())
                finish += diff
                output[i, j, start:finish] = 1
                start += diff
        output = output.to(device)
        output = (x.transpose(-1, -2)) @ output

        return output.transpose(-1, -2)

    def forward(self, x, target=None):
        durations = torch.exp(self.dp(x))
        if target is not None:
            output = self.LR(x, target)
            return output, durations
        else:
            output = self.LR(x, durations)
            return output, durations
