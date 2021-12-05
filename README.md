# TTS FastSpeech



## Installation guide
```
pip install -r ./requirements.txt -q
```
Download LJSpeech dataset:

```
./get_ljspeech.sh
```

Download waveglow vocoder:

```
python get_waveglow.py
```
Download alignments:

```
./get_alignments.sh
```

Download checkpoint (model in `checkpoint.pth`, config in `config.json`):

```
python download_checkpoint.py
```

Test outputs are in audio folder.