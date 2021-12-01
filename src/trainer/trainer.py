import random
from random import shuffle

import PIL
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm

from src.base import BaseTrainer
from src.logger.utils import plot_spectrogram_to_buf
from src.utils import inf_loop, MetricTracker
from src.featurizer import MelSpectrogramConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            criterion,
            optimizer,
            config,
            device,
            data_loader,
            aligner,
            featurizer,
            vocoder,
            valid_data_loader=None,
            lr_scheduler=None,
            len_epoch=None,
            skip_oom=True,
    ):
        super().__init__(model, criterion, optimizer, config, device)
        self.skip_oom = skip_oom
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = 10
        self.vocoder = vocoder
        self.featurizer = featurizer
        self.aligner = aligner

        self.train_metrics = MetricTracker(
            "loss", "duration loss", "mel loss", "grad norm", writer=self.writer
        )
        self.valid_metrics = MetricTracker(
            "loss", "duration loss", "mel loss", writer=self.writer
        )

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        batch = batch.to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
                tqdm(self.data_loader, desc="train", total=self.len_epoch)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    is_train=True,
                    metrics=self.train_metrics,
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                elif "Error loading audio file" in str(e):
                    self.logger.warning("failed to open file on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            self.train_metrics.update("grad norm", self.get_grad_norm())
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                # self.logger.debug(
                #     "Train Epoch: {} {} Loss: {:.6f}".format(
                #         epoch, self._progress(batch_idx), batch["loss"].item()
                #     )
                # )
                self.writer.add_scalar(
                    "learning rate", self.lr_scheduler.get_last_lr()[0]
                )
                self._log_spectrogram(batch.melspec, batch.melspec_pred)
                self._log_audio(batch.waveform, batch.melspec_pred)
                self._log_scalars(self.train_metrics)
            if batch_idx >= self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{"val_" + k: v for k, v in val_log.items()})

        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):

        batch.durations = self.aligner(
            batch.waveform.to(device), batch.waveform_length, batch.transcript
        )
        mel_config = MelSpectrogramConfig()
        mel_len = batch.waveform_length / mel_config.hop_length
        batch = batch.to(torch.device('cpu'))
        mel_len = mel_len.cpu()
        batch.durations *= mel_len.repeat(batch.durations.shape[-1], 1).transpose(0, 1)
        mels = self.featurizer(batch.waveform)
        batch.melspec = mels
        batch = self.move_batch_to_device(batch, self.device)
        output, durations = self.model(batch.tokens, batch.durations)

        batch.melspec_pred = output
        batch.duration_pred = durations

        loss, mel_loss, duration_loss = self.criterion(batch)
        if is_train:
            loss.backward()
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        metrics.update("loss", loss.item())
        metrics.update("mel loss", mel_loss.item())
        metrics.update("duration loss", duration_loss.item())

        return batch

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                    enumerate(self.valid_data_loader),
                    desc="validation",
                    total=len(self.valid_data_loader),
            ):
                batch = self.process_batch(
                    batch,
                    is_train=False,
                    metrics=self.valid_metrics,
                )
            self.writer.set_step(epoch * self.len_epoch, "valid")
            self._log_scalars(self.valid_metrics)
            # self._log_predictions(part="val", **batch)
            self._log_spectrogram(batch.melspec, batch.melspec_pred)
            self._log_audio(batch.waveform, batch.melspec_pred)

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.data_loader, "n_samples"):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_predictions(
            self,
            text,
            log_probs,
            log_probs_length,
            examples_to_log=5,
            *args,
            **kwargs,
    ):
        if self.writer is None:
            return

        argmax_inds = log_probs.cpu().argmax(-1)
        argmax_inds = [
            inds[: int(ind_len)]
            for inds, ind_len in zip(argmax_inds, log_probs_length)
        ]
        argmax_texts_raw = [self.text_encoder.decode(inds) for inds in argmax_inds]
        argmax_texts = [self.text_encoder.ctc_decode(inds) for inds in argmax_inds]
        tuples = list(zip(argmax_texts, text, argmax_texts_raw))
        shuffle(tuples)
        to_log_pred = []
        to_log_pred_raw = []
        for pred, target, raw_pred in tuples[:examples_to_log]:
            to_log_pred_raw.append(f"true: '{target}' | pred: '{raw_pred}'\n")
        self.writer.add_text(f"predictions", "< < < < > > > >".join(to_log_pred))
        self.writer.add_text(
            f"predictions_raw", "< < < < > > > >".join(to_log_pred_raw)
        )

    def _log_spectrogram(self, spectrogram_batch_true, spectrogram_batch_pred):
        idx = random.choice(range(len(spectrogram_batch_true)))
        spectrogram_true = spectrogram_batch_true[idx]
        spectrogram_pred = spectrogram_batch_pred[idx]
        spectrogram_true = spectrogram_true.cpu().detach()
        spectrogram_pred = spectrogram_pred.cpu().detach()
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram_true))
        self.writer.add_image("spectrogram true", ToTensor()(image))
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram_pred))
        self.writer.add_image("spectrogram pred", ToTensor()(image))

    def _log_audio(self, waveform_true, melspec_pred):
        idx = random.choice(range(len(melspec_pred)))
        waveform_pred = self.vocoder.inference(melspec_pred[idx].unsqueeze(0)).squeeze(0)
        self.writer.add_audio("audio true", waveform_true[idx].cpu(), 22050)
        self.writer.add_audio("audio pred", waveform_pred.cpu(), 22050)

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
