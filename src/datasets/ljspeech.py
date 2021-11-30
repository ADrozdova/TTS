import torch

import torchaudio


class LJSpeechDataset(torchaudio.datasets.LJSPEECH):

    def __init__(self, root, part):
        super().__init__(root=root)
        self._tokenizer = torchaudio.pipelines.TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH.get_text_processor()
        self.part = part
        len_all = super().__len__()
        self.train_len = int(0.8 * len_all)
        self.val_len = len_all - self.train_len

    def __len__(self):
        if self.part == "train":
            return self.train_len
        if self.part == "val":
            return self.train_len
        len = super().__len__()
        train_size = int(0.8 * full_size)
        if self.mode == "train":
            return train_size
        return full_size - train_size

    def __getitem__(self, index: int):
        waveform, _, _, transcript = super().__getitem__(index)
        waveforn_length = torch.tensor([waveform.shape[-1]]).int()

        tokens, token_lengths = self._tokenizer(transcript)

        return waveform, waveforn_length, transcript, tokens, token_lengths

    def decode(self, tokens, lengths):
        result = []
        for tokens_, length in zip(tokens, lengths):
            text = "".join([
                self._tokenizer.tokens[token]
                for token in tokens_[:length]
            ])
            result.append(text)
        return result
