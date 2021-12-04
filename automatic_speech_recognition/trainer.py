# MIT License
# code by Soohwan Kim @sooftware

import wandb
import time
import os
import torch
import logging
from tqdm import tqdm

from metric import CharacterErrorRate

logger = logging.getLogger(__name__)


class AutomaticSpeechRecognitionTrainer:
    def __init__(
            self,
            model,
            train_loader,
            valid_loader,
            test_loader,
            criterion,
            optimizer,
            scheduler,
            num_epochs,
            device,
            tokenizer,
            gradient_clip_val: float,
            accumulate_grad_batches: int,
            log_every: int = 20,
            save_every: int = 10_000,
            save_dir: str = 'ckpt',
    ) -> None:
        super(AutomaticSpeechRecognitionTrainer, self).__init__()
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.log_every = log_every
        self.save_every = save_every
        self.num_epochs = num_epochs
        self.tokenizer = tokenizer
        self.sos_id = tokenizer.sos_id
        self.eos_id = tokenizer.eos_id
        self.device = device
        self.gradient_clip_val = gradient_clip_val
        self.accumulate_grad_batches = accumulate_grad_batches
        self.metric = CharacterErrorRate(tokenizer)
        self.save_dir = save_dir

    def _get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def _train_epoch(self):
        total_loss = 0.0

        self.model.train()

        for step, (sources, targets, source_lengths, target_lengths) in enumerate(tqdm(self.train_loader)):
            sources = sources.to(self.device)
            source_lengths = source_lengths.to(self.device)
            targets = targets.to(self.device)
            target_lengths = target_lengths.to(self.device)

            batch_size = sources.size(0)

            del_eos_targets = targets[targets != self.eos_id].view(batch_size, -1)
            del_sos_targets = targets[:, 1:]

            logits = self.model(sources, source_lengths, del_eos_targets)
            loss = self.criterion(
                logits.contiguous().view(-1, logits.size(-1)), del_sos_targets.contiguous().view(-1)
            )

            loss = loss / self.accumulate_grad_batches

            self.optimizer.zero_grad()
            loss.backward()
            total_loss += loss.item()

            if step % self.accumulate_grad_batches == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                self.optimizer.step()
                self.scheduler.step()

            predictions = logits.max(-1)[1]
            cer = self.metric(del_sos_targets, predictions)

            if step % self.log_every == 0:
                train_mean_loss = total_loss / (step + 1)
                wandb.log({"lr": self._get_lr(),
                           "train_loss": train_mean_loss,
                           "train_cer": cer,
                           "batch_size": batch_size})

            if step % self.save_every == 0:
                if not os.path.exists(self.save_dir):
                    os.mkdir(self.save_dir)
                date_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())

                os.mkdir(os.path.join(self.save_dir, date_time))
                torch.save(self.model, os.path.join(self.save_dir, date_time, "model.pt"))

    def _validate(self, epoch):
        total_cer = 0.0

        self.model.eval()

        with torch.no_grad():
            for step, (sources, targets, source_lengths, target_lengths) in enumerate(tqdm(self.valid_loader)):
                sources = sources.to(self.device)
                targets = targets.to(self.device)

                del_sos_targets = targets[:, 1:]

                logits = self.model(sources, source_lengths, teacher_forcing_ratio=0.0)

                predictions = logits.max(-1)[1]
                cer = self.metric(del_sos_targets, predictions)

                total_cer += cer

        valid_mean_cer = total_cer / len(self.valid_loader)

        wandb.log({"valid_cer": valid_mean_cer})

    def test(self):
        total_cer = 0.0

        self.model.eval()

        with torch.no_grad():
            for batch_idx, (sources, targets, source_lengths, target_lengths) in enumerate(tqdm(self.test_loader)):
                sources = sources.to(self.device)
                targets = targets.to(self.device)

                del_sos_targets = targets[:, 1:]

                logits = self.model(sources, source_lengths, teacher_forcing_ratio=0.0)

                predictions = logits.max(-1)[1]
                cer = self.metric(del_sos_targets, predictions)

                total_cer += cer

        test_mean_cer = total_cer / len(self.valid_loader)

        wandb.log({"test_cer": test_mean_cer})

    def fit(self):
        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch} start..")
            self._train_epoch()

            logger.info(f"Validate..")
            self._validate(epoch)
