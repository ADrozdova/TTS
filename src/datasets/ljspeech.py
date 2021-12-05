import torch

import torchaudio

LJ_ABBR = {
    "Mr.":	"Mister",
    "Mrs.":	"Misess",
    "Dr.":	"Doctor",
    "No.":	"Number",
    "St.":	"Saint",
    "Co.":	"Company",
    "Jr.":	"Junior",
    "Maj.":	"Major",
    "Gen.":	"General",
    "Drs.":	"Doctors",
    "Rev.":	"Reverend",
    "Lt.":	"Lieutenant",
    "Hon.":	"Honorable",
    "Sgt.":	"Sergeant",
    "Capt.":	"Captain",
    "Esq.":	"Esquire",
    "Ltd.":	"Limited",
    "Col.":	"Colonel",
    "Ft.":	"Fort",
}


def fix_text(text):
    to_replace = list('"üêàéâè”“’[]')
    replacements = [""] + list("ueaeae") + ["", "", "'", "", ""]
    for c, rep in zip(to_replace, replacements):
        text = text.replace(c, rep)
    for abbr, word in LJ_ABBR.items():
        text = text.replace(abbr, word)
    return text


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
            return self.val_len

    def __getitem__(self, index: int):
        if self.part == "val":
            index += self.train_len
        waveform, _, _, transcript = super().__getitem__(index)
        waveform_length = torch.tensor([waveform.shape[-1]]).int()
        transcript = fix_text(transcript)

        tokens, token_lengths = self._tokenizer(transcript)

        return index, waveform, waveform_length, transcript, tokens, token_lengths

    def decode(self, tokens, lengths):
        result = []
        for tokens_, length in zip(tokens, lengths):
            text = "".join([
                self._tokenizer.tokens[token]
                for token in tokens_[:length]
            ])
            result.append(text)
        return result
