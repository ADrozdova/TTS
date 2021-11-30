from torch import nn

from fastspeech_blocks import Encoder, Decoder, LengthRegulator
from src.base.base_model import BaseModel


class FastSpeech(BaseModel):
    def __init__(self, voc_size, encoder_in_size, encoder_blocks,
                 n_heads, fft_emb, attn_dropout,
                 conv_in, conv_hidden, conv_out, kernels, padding, dropout,
                 decoder_blocks,
                 dp_in_size, dp_hidden, dp_kernel_size, dp_dropout, dp_out_size,
                 fc_in, fc_mel):
        super(FastSpeech, self).__init__()
        self.encoder = Encoder(voc_size, encoder_in_size, encoder_blocks,
                               n_heads, fft_emb, attn_dropout,
                               conv_in, conv_hidden, conv_out, kernels, padding, dropout,
                               )
        self.leb_reg = LengthRegulator(dp_in_size, dp_hidden, dp_kernel_size, dp_dropout, dp_out_size)
        self.decoder = Decoder(decoder_blocks,
                               n_heads, fft_emb, attn_dropout,
                               conv_in, conv_hidden, conv_out, kernels, padding, dropout)
        self.fc = nn.Linear(fc_in, fc_mel)

    def forward(self, input, durations=None):
        output = self.encoder(input)
        if self.training:
            output, dp = self.len_reg(output, target=durations)
            output = self.decoder(output)
            output = self.fc(output).transpose(1, 2)
            return output, dp.squeeze(-1)

        output, dp = self.len_reg(output)
        output = self.decoder(output)
        output = self.fc(output).transpose(1, 2)
        return output
