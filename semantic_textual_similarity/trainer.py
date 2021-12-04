# MIT License
# code by Soohwan Kim @sooftware

import wandb
import time
import os
import torch
import logging
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger(__name__)


class SemanticTextualSimilarityTrainer:
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
        super(SemanticTextualSimilarityTrainer, self).__init__()
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
        self.save_dir = save_dir

    def _get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def _train_epoch(self):
        total_loss = 0.0
        total_pearson_corr = 0.0
        total_spearman_corr = 0.0

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

            preds = outputs.logits.view(-1)

            pearson_corr = pearsonr(preds.tolist(), targets.tolist())[0]
            spearman_corr = spearmanr(preds.tolist(), targets.tolist())[0]

            total_pearson_corr += pearson_corr
            total_spearman_corr += spearman_corr

            if step % self.log_every == 0:
                train_mean_loss = total_loss / (step + 1)
                train_pearson_corr = total_pearson_corr / (step + 1)
                train_spearman_corr = total_spearman_corr / (step + 1)
                wandb.log({"lr": self._get_lr(),
                           "train_loss": train_mean_loss,
                           "train_pearson_corr": train_pearson_corr,
                           "train_spearman_corr": train_spearman_corr,
                           "batch_size": input_ids.size(0)})

            if step % self.save_every == 0:
                if not os.path.exists(self.save_dir):
                    os.mkdir(self.save_dir)
                date_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())

                os.mkdir(os.path.join(self.save_dir, date_time))
                torch.save(self.model, os.path.join(self.save_dir, date_time, "model.pt"))

    def _validate(self, epoch):
        total_loss = 0.0
        total_pearson_corr = 0.0
        total_spearman_corr = 0.0

        self.model.eval()

        with torch.no_grad():
            for step, (input_ids, attention_masks, targets) in enumerate(tqdm(self.valid_loader)):
                input_ids = input_ids.to(self.device)
                attention_masks = attention_masks.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_masks, labels=targets)

                total_loss += outputs.loss.item()

                preds = outputs.logits.view(-1)

                pearson_corr = pearsonr(preds.tolist(), targets.tolist())[0]
                spearman_corr = spearmanr(preds.tolist(), targets.tolist())[0]

                total_pearson_corr += pearson_corr
                total_spearman_corr += spearman_corr

        valid_mean_loss = total_loss / len(self.valid_loader)
        valid_pearson_corr = total_pearson_corr / len(self.valid_loader)
        valid_spearman_corr = total_spearman_corr / len(self.valid_loader)

        wandb.log({"valid_loss": valid_mean_loss,
                   "valid_pearson_corr": valid_pearson_corr,
                   "valid_spearman_corr": valid_spearman_corr})

    def test(self):
        total_pearson_corr = 0.0
        total_spearman_corr = 0.0

        self.model.eval()

        with torch.no_grad():
            for step, (input_ids, attention_masks, targets) in enumerate(tqdm(self.valid_loader)):
                input_ids = input_ids.to(self.device)
                attention_masks = attention_masks.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_masks)

                preds = outputs.logits.view(-1)

                pearson_corr = pearsonr(preds.tolist(), targets.tolist())[0]
                spearman_corr = spearmanr(preds.tolist(), targets.tolist())[0]

                total_pearson_corr += pearson_corr
                total_spearman_corr += spearman_corr

        test_pearson_corr = total_pearson_corr / len(self.test_loader)
        test_spearman_corr = total_spearman_corr / len(self.test_loader)

        wandb.log({"test_pearson_corr": test_pearson_corr,
                   "test_spearman_corr": test_spearman_corr})

    def fit(self):
        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch} start..")
            self._train_epoch()

            logger.info(f"Validate..")
            self._validate(epoch)
