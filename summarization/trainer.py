# MIT License
# code by Soohwan Kim @sooftware

import wandb
import time
import os
import torch
import logging
from rouge import Rouge
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SummarizationTrainer:
    def __init__(
            self,
            model,
            train_loader,
            valid_loader,
            test_loader,
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
        super(SummarizationTrainer, self).__init__()
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
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
        self.rouge = Rouge()
        self.save_dir = save_dir

    def _get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def _train_epoch(self):
        total_loss = 0.0

        self.model.train()

        for step, (input_ids, attention_masks, targets) in enumerate(tqdm(self.train_loader)):
            input_ids = input_ids.to(self.device)
            attention_masks = attention_masks.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_masks, labels=targets)

            loss = outputs.loss
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
                           "batch_size": input_ids.size(0)})

            if step % self.save_every == 0:
                if not os.path.exists(self.save_dir):
                    os.mkdir(self.save_dir)
                date_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())

                os.mkdir(os.path.join(self.save_dir, date_time))
                torch.save(self.model, os.path.join(self.save_dir, date_time, "model.pt"))

    def _validate(self, epoch):
        total_loss = 0.0
        predictions = list()
        targets = list()

        self.model.eval()

        with torch.no_grad():
            for step, (input_ids, attention_masks, targets) in enumerate(tqdm(self.valid_loader)):
                input_ids = input_ids.to(self.device)
                attention_masks = attention_masks.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_masks, labels=targets)
                loss = outputs.loss

                total_loss += loss.item()

                predictions.extend(self.tokenizer.batch_decode(outputs.logits.max(-1)[1]))
                targets.extend(self.tokenizer.batch_decode(targets))

        rouge_score = self.rouge.get_scores(predictions, targets, avg=True)
        val_mean_loss = total_loss / len(self.valid_loader)

        wandb.log({"valid_loss": val_mean_loss,
                   "valid_rouge_1_f": rouge_score["rouge-1"][0]["f"],
                   "valid_rouge_1_p": rouge_score["rouge-1"][0]["p"],
                   "valid_rouge_1_r": rouge_score["rouge-1"][0]["r"],
                   "valid_rouge_2_f": rouge_score["rouge-2"][0]["f"],
                   "valid_rouge_2_p": rouge_score["rouge-2"][0]["p"],
                   "valid_rouge_2_r": rouge_score["rouge-2"][0]["r"],
                   "valid_rouge_l_f": rouge_score["rouge-l"][0]["f"],
                   "valid_rouge_l_p": rouge_score["rouge-l"][0]["p"],
                   "valid_rouge_l_r": rouge_score["rouge-l"][0]["r"]})

    def test(self):
        predictions = list()
        targets = list()

        self.model.eval()

        with torch.no_grad():
            for step, (input_ids, attention_masks, targets) in enumerate(tqdm(self.test_loader)):
                input_ids = input_ids.to(self.device)
                attention_masks = attention_masks.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model.generate(input_ids=input_ids, attention_mask=attention_masks)

                predictions.extend(self.tokenizer.batch_decode(outputs, skip_special_tokens=True))
                targets.extend(self.tokenizer.batch_decode(targets, skip_special_tokens=True))

        rouge_score = self.rouge.get_scores(predictions, targets, avg=True)

        wandb.log({"test_rouge_1_f": rouge_score["rouge-1"][0]["f"],
                   "test_rouge_1_p": rouge_score["rouge-1"][0]["p"],
                   "test_rouge_1_r": rouge_score["rouge-1"][0]["r"],
                   "test_rouge_2_f": rouge_score["rouge-2"][0]["f"],
                   "test_rouge_2_p": rouge_score["rouge-2"][0]["p"],
                   "test_rouge_2_r": rouge_score["rouge-2"][0]["r"],
                   "test_rouge_l_f": rouge_score["rouge-l"][0]["f"],
                   "test_rouge_l_p": rouge_score["rouge-l"][0]["p"],
                   "test_rouge_l_r": rouge_score["rouge-l"][0]["r"]})

    def fit(self):
        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch} start..")
            self._train_epoch()

            logger.info(f"Validate..")
            self._validate(epoch)
