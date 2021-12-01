from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List

import torch
from torch.nn.utils.rnn import pad_sequence


@dataclass
class Batch:
    waveform: torch.Tensor
    waveform_length: torch.Tensor
    transcript: List[str]
    tokens: torch.Tensor
    token_lengths: torch.Tensor
    durations: Optional[torch.Tensor] = None
    duration_pred: Optional[torch.Tensor] = None
    melspec: Optional[torch.Tensor] = None
    melspec_pred: Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> 'Batch':
        self.waveform = self.waveform.to(device)
        self.waveform_length = self.waveform_length.to(device)
        self.tokens = self.tokens.to(device)
        self.token_lengths = self.token_lengths.to(device)
        if self.durations is not None:
            self.durations = self.durations.to(device)
        if self.duration_pred is not None:
            self.duration_pred = self.duration_pred.to(device)
        if self.melspec is not None:
            self.melspec = self.melspec.to(device)
        if self.melspec_pred is not None:
            self.melspec_pred = self.melspec_pred.to(device)
        return self


class LJSpeechCollator:

    def __call__(self, instances: List[Tuple]) -> Batch:
        waveform, waveform_length, transcript, tokens, token_lengths = list(
            zip(*instances)
        )

        waveform = pad_sequence([
            waveform_[0] for waveform_ in waveform
        ]).transpose(0, 1)
        waveform_length = torch.cat(waveform_length)

        tokens = pad_sequence([
            tokens_[0] for tokens_ in tokens
        ]).transpose(0, 1)
        token_lengths = torch.cat(token_lengths)

        return Batch(waveform, waveform_length, transcript, tokens, token_lengths)
