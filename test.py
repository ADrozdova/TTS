import argparse
import os.path
import warnings

import numpy as np
import torch
import torchaudio
from scipy.io import wavfile

import src.model as module_arch
from src.aligner.aligner import Aligner
from src.utils import prepare_device
from src.utils.parse_config import ConfigParser
from src.vocoder import Vocoder

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

TEST_SENTENCES = [
    "A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac "
    "arrest",
    "Massachusetts Institute of Technology may be best known for its math, science and engineering education",
    "Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability "
    "distributions on a given metric space "
]


def main(config_path, checkpoint, output_path):
    config = ConfigParser.from_args(config_path)
    vocoder = Vocoder("waveglow_256channels_universal_v5.pt").eval()
    tokenizer = torchaudio.pipelines.TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH.get_text_processor()

    # build model architecture, then print to console
    model = config.init_obj(config["arch"], module_arch)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint)["state_dict"])
    vocoder = vocoder.to(device)

    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    for i in range(len(TEST_SENTENCES)):
        sentence = TEST_SENTENCES[i]
        tokens, _ = tokenizer(sentence)
        with torch.no_grad():
            melspec, _ = model(tokens)
            wav = vocoder.inference((melspec.unsqueeze(0)).squeeze(0))
            wavfile.write(os.path.join(output_path, "test_" + str(i) + ".wav"), 22050, wav.squeeze(0).cpu().numpy())


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-o",
        "--output",
        default="./test_output",
        type=str,
        help="output path (default: ./test_output)"
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    
    args_parsed = args.parse_args()

    main(args, args_parsed.resume, args_parsed.output)
