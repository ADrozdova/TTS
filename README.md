# TTS



## Installation guide
```
pip install -r ./requirements.txt -q
```
Download LJSpeech dataset

```
./get_ljspeech.sh
```

Download waveglow vocoder

```
python get_waveglow.py
```
Download alignments

```
./get_alignments.sh
```

Test outputs are in audio folder.