import torch
import torch.nn.functional as F
from torch import nn

from src.model.attention import MultiHeadAttention

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ConvFFT(nn.Module):
    def __init__(self, conv_in, conv_hidden, conv_out, kernels, padding, dropout):
        super(ConvFFT, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(conv_in, conv_hidden, kernel_size=kernels[0], padding=padding[0]),
            nn.ReLU(),
            nn.Conv1d(conv_hidden, conv_out, kernel_size=kernels[1], padding=padding[1]),
            nn.Dropout(dropout)
        )
        self.ln = nn.LayerNorm(conv_out)

    def forward(self, x):
        out = self.layers(x.transpose(-1, -2))
        return self.ln(out.transpose(-1, -2) + x)


class FFTBlock(nn.Module):
    def __init__(self, n_heads, fft_emb, attn_dropout,
                 conv_in, conv_hidden, conv_out, kernels, padding, dropout):
        super(FFTBlock, self).__init__()
        self.attn = MultiHeadAttention(n_heads, conv_in, attn_dropout)
        self.conv = ConvFFT(conv_in, conv_hidden, conv_out, kernels, padding, dropout)

    def forward(self, x, mask=None):
        out, attn = self.attn(x, x, x, mask)
        out = self.conv(out)
        return out, attn


class Encoder(nn.Module):
    def __init__(self, voc_size, encoder_in_size, encoder_blocks,
                 n_heads, fft_emb, attn_dropout,
                 conv_in, conv_hidden, conv_out, kernels, padding, dropout
                 ):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(voc_size, encoder_in_size)
        blocks = [FFTBlock(n_heads, fft_emb, attn_dropout, conv_in, conv_hidden, conv_out, kernels, padding, dropout)
                  for _ in range(encoder_blocks)]
        self.layers = nn.Sequential(*blocks)

    def get_attn_mask(self, x):
        mask = x.eq(0)
        mask = mask.unsqueeze(1).expand(-1, x.shape[1], -1)
        return mask

    def forward(self, x):
        out = self.embedding(x)
        mask = self.get_attn_mask(x)
        for layer in self.layers:
            out, _ = layer(out, mask)
        return out


class Decoder(nn.Module):
    def __init__(self, decoder_blocks,
                 n_heads, fft_emb, attn_dropout,
                 conv_in, conv_hidden, conv_out, kernels, padding, dropout):
        super(Decoder, self).__init__()
        blocks = [FFTBlock(n_heads, fft_emb, attn_dropout, conv_in, conv_hidden, conv_out, kernels, padding, dropout)
                  for _ in range(decoder_blocks)]
        self.layers = nn.Sequential(*blocks)

    def forward(self, x, mask=None):
        out = x
        for layer in self.layers:
            out, _ = layer(out, mask)
        return out


class ConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ConvTranspose, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding)

    def forward(self, x):
        return self.conv(x.transpose(-1, -2)).transpose(-1, -2)


class DurationPredictor(nn.Module):
    def __init__(self, dp_in_size, dp_hidden, dp_kernel_size, dp_dropout, dp_out_size):
        super(DurationPredictor, self).__init__()
        self.layers = nn.Sequential(
            ConvTranspose(dp_in_size, dp_hidden, dp_kernel_size, dp_kernel_size // 2),
            nn.LayerNorm(dp_hidden),
            nn.ReLU(),
            nn.Dropout(dp_dropout),
            ConvTranspose(dp_hidden, dp_out_size, dp_kernel_size, dp_kernel_size // 2),
            nn.LayerNorm(dp_out_size),
            nn.ReLU(),
            nn.Dropout(dp_dropout),
            nn.Linear(dp_out_size, 1),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.layers(x).squeeze()
        if not self.training:
            out = out.unsqueeze(0)
        return out


def create_alignment(x, durations):
    N, L = durations.shape
    for i in range(N):
        count = 0
        for j in range(L):
            for k in range(durations[i][j]):
                x[i][count + k][j] = 1
            count = count + durations[i][j]
    return x


class LengthRegulator(nn.Module):
    def __init__(self, dp_in_size, dp_hidden, dp_kernel_size, dp_dropout, dp_out_size):
        super(LengthRegulator, self).__init__()
        self.dp = DurationPredictor(dp_in_size, dp_hidden, dp_kernel_size, dp_dropout, dp_out_size)

    def LR(self, x, dp_output, alpha=1.0, inference=False):
        L, B, D = x.size()
        dp_output = torch.round(dp_output * alpha).to(torch.long)
        if inference:
            dp_output[dp_output <= 0] = 1
        T = int(torch.sum(dp_output, dim=-1).max().item())
        expanded = x.new_zeros(T, B, D)

        for i, d in enumerate(dp_output):
            mel_len = torch.sum(d).item()
            expanded[:mel_len, i] = torch.repeat_interleave(x[:, i], d, dim=0)
        return expanded

    def forward(self, x, alpha=1.0, target=None):
        durations = torch.exp(self.dp(x))
        if target is not None:
            output = self.LR(x, target)
            return output, durations
        else:
            duration_predictor_output = ((durations + 0.5) * alpha).int()
            output = self.LR(x, duration_predictor_output)
            mel_pos = torch.stack([torch.Tensor([i + 1 for i in range(output.size(1))])]).long().to(device)

            return output, mel_pos
