from operator import xor

from torch.utils.data import DataLoader, ChainDataset

import src.datasets
from src.collate_fn.collate import LJSpeechCollator
from src.utils.parse_config import ConfigParser


def get_dataloaders(configs: ConfigParser, root):
    dataloaders = {}
    for split, params in configs["data"].items():
        num_workers = params.get("num_workers", 1)

        # set train augmentations
        if split == 'train':
            wave_augs, spec_augs = src.augmentations.from_configs(configs)
        else:
            wave_augs, spec_augs = None, None

        # create and join datasets
        datasets = []
        for ds in params["datasets"]:
            datasets.append(configs.init_obj(
                ds, src.datasets, root=root))
        assert len(datasets)
        if len(datasets) > 1:
            dataset = ChainDataset(datasets)
        else:
            dataset = datasets[0]

        # select batch size or batch sampler
        assert xor("batch_size" in params, "batch_sampler" in params), \
            "You must provide batch_size or batch_sampler for each split"
        if "batch_size" in params:
            bs = params["batch_size"]
            shuffle = True
            batch_sampler = None
            bs, shuffle = 1, False
        else:
            raise Exception()

        # create dataloader
        dataloader = DataLoader(
            dataset, batch_size=bs, collate_fn=LJSpeechCollator,
            shuffle=shuffle, num_workers=num_workers, batch_sampler=batch_sampler)
        dataloaders[split] = dataloader
    return dataloaders
