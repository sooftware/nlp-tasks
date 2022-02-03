# MIT License
# code by Soohwan Kim @sooftware

import wandb
import time
import os
import torch
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class LanguageModelingTrainer:
    def __init__(
            self,
            model,
            train_loader,
            valid_loader,
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
        super(LanguageModelingTrainer, self).__init__()
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.log_every = log_every
        self.save_every = save_every
        self.num_epochs = num_epochs
        self.tokenizer = tokenizer
        self.sos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id
        self.device = device
        self.gradient_clip_val = gradient_clip_val
        self.accumulate_grad_batches = accumulate_grad_batches
        self.save_dir = save_dir

    def _get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def _train_epoch(self):
        total_loss = 0.0

        self.model.train()

        for step, (inputs, attention_masks, labels) in enumerate(tqdm(self.train_loader)):
            inputs = inputs.to(self.device)
            attention_masks = attention_masks.to(self.device)
            labels = labels.to(self.device)

            batch_size = inputs.size(0)

            outputs = self.model(input_ids=inputs, attention_mask=attention_masks, labels=labels)
            loss = outputs.loss

            loss = loss.mean()
            loss = loss / self.accumulate_grad_batches

            self.optimizer.zero_grad()
            loss.backward()
            total_loss += loss.item()

            if step % self.accumulate_grad_batches == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                self.optimizer.step()
                self.scheduler.step()

            if step % self.log_every == 0:
                train_mean_loss = total_loss / (step + 1)
                wandb.log({"lr": self._get_lr(),
                           "train_loss": train_mean_loss,
                           "batch_size": batch_size})

            if step % self.save_every == 0:
                if not os.path.exists(self.save_dir):
                    os.mkdir(self.save_dir)
                date_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
                self.model.save_pretrained(os.path.join(self.save_dir, date_time))

    def _validate(self, epoch):
        valid_loss = 0.0

        self.model.eval()

        with torch.no_grad():
            for step, (inputs, attention_masks, labels) in enumerate(tqdm(self.valid_loader)):
                inputs = inputs.to(self.device)
                attention_masks = attention_masks.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(input_ids=inputs, attention_mask=attention_masks, labels=labels)
                loss = outputs.loss

                valid_loss += loss.item()

        valid_mean_loss = valid_loss / len(self.valid_loader)

        wandb.log({"valid_loss": valid_mean_loss})

    def fit(self):
        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch} start..")
            self._train_epoch()

            logger.info(f"Validate..")
            self._validate(epoch)
