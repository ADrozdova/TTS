from torch import nn

from src.base.base_model import BaseModel
from src.model.fastspeech_blocks import FFT, LengthRegulator, PositionalEncoding
from src.model.utils import mask_padding


class FastSpeech(BaseModel):
    def __init__(self, voc_size, enc_layers, in_size, n_heads, hidden, kernels, padding, dp_in_size, dp_hidden,
                 dp_kernel, dec_layers, fc_in, fc_mel, attn_dropout=0.1, dropout=0.1):
        super(FastSpeech, self).__init__()
        self.encoder = FFT(enc_layers, in_size, n_heads, hidden, kernels, padding, attn_dropout=attn_dropout,
                           dropout=dropout)
        self.emb = nn.Embedding(voc_size, in_size, padding_idx=0)
        self.pos_enc = PositionalEncoding(in_size, dropout)

        self.len_reg = LengthRegulator(dp_in_size, dp_hidden, dp_kernel, dropout=dropout)
        self.decoder = FFT(dec_layers, in_size, n_heads, hidden, kernels, padding, attn_dropout=attn_dropout,
                           dropout=dropout)
        self.fc = nn.Linear(fc_in, fc_mel)

    def forward(self, x, durations=None):
        out = self.pos_enc(self.emb(x))
        out = self.encoder(out, mask_padding(x))
        out, dp = self.len_reg(out, target=durations)
        out = self.pos_enc(out)
        out = self.decoder(out)
        return self.fc(out).transpose(1, 2), dp
