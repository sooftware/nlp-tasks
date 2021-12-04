# MIT License
# code by Soohwan Kim @sooftware

import wandb
import time
import os
import torch
import logging
import pandas as pd
import nltk
from tqdm import tqdm

logger = logging.getLogger(__name__)


class MachineTranslationTrainer:
    def __init__(
            self,
            model,
            train_loader,
            valid_loader,
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
        super(MachineTranslationTrainer, self).__init__()
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.log_every = log_every
        self.save_every = save_every
        self.num_epochs = num_epochs
        self.tokenizer = tokenizer
        self.sos_id = tokenizer.PieceToId('<s>')
        self.eos_id = tokenizer.PieceToId('</s>')
        self.device = device
        self.gradient_clip_val = gradient_clip_val
        self.accumulate_grad_batches = accumulate_grad_batches
        self.save_dir = save_dir

    def _get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def _save_results(self, pred_texts, target_texts, save_name):
        pd.DataFrame({
            "predictions": pred_texts,
            "targets": target_texts,
        }).to_csv(save_name, index=False)

    def _train_epoch(self):
        total_loss = 0.0
        total_bleus = list()

        self.model.train()

        for step, (sources, targets, source_lengths, target_lengths) in enumerate(tqdm(self.train_loader)):
            bleus = list()

            sources = sources.to(self.device)
            source_lengths = source_lengths.to(self.device)
            targets = targets.to(self.device)
            target_lengths = target_lengths.to(self.device)

            batch_size = sources.size(0)

            del_eos_targets = targets[targets != self.eos_id].view(batch_size, -1)
            del_sos_targets = targets[:, 1:]

            outputs = self.model(sources, source_lengths, del_eos_targets, target_lengths)

            loss = self.criterion(
                outputs.contiguous().view(-1, outputs.size(-1)), del_sos_targets.contiguous().view(-1)
            )
            loss = loss / self.accumulate_grad_batches

            self.optimizer.zero_grad()
            loss.backward()
            total_loss += loss.item()

            if step % self.accumulate_grad_batches == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                self.optimizer.step()
                self.scheduler.step()

            for i in range(batch_size):
                target_text = self.tokenizer.DecodeIds(targets[i].tolist())
                pred_text = outputs[i].max(dim=1)[1]
                pred_text = self.tokenizer.DecodeIds(pred_text.tolist())
                bleu = nltk.translate.bleu_score.sentence_bleu([target_text.split()], pred_text.split())
                bleus.append(bleu)

            bleus = sum(bleus) / len(bleus)
            total_bleus.append(bleus)

            if step % self.log_every == 0:
                train_mean_loss = total_loss / (step + 1)
                total_mean_bleus = sum(total_bleus) / len(total_bleus)
                wandb.log({"lr": self._get_lr(),
                           "train_loss": train_mean_loss,
                           "train_bleu": total_mean_bleus,
                           "batch_size": batch_size})

            if step % self.save_every == 0:
                if not os.path.exists(self.save_dir):
                    os.mkdir(self.save_dir)
                date_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())

                os.mkdir(os.path.join(self.save_dir, date_time))

                torch.save(self.model, os.path.join(self.save_dir, date_time, "model.pt"))

    def _validate(self, epoch):
        total_bleus = list()

        pred_texts = list()
        target_texts = list()

        self.model.eval()

        with torch.no_grad():
            for step, (sources, targets, source_lengths, target_lengths) in enumerate(tqdm(self.valid_loader)):
                sources = sources.to(self.device)
                targets = targets.to(self.device)

                batch_size = sources.size(0)

                outputs = self.model(sources, source_lengths)

                bleus = list()
                for i in range(batch_size):
                    target_text = self.tokenizer.DecodePieces(targets[i].tolist())
                    pred_text = outputs[i].max(dim=1)[1]
                    pred_text = self.tokenizer.DecodePieces(pred_text.tolist())
                    bleu = nltk.translate.bleu_score.sentence_bleu([target_text.split()], pred_text.split())
                    bleus.append(bleu)

                    pred_texts.append(pred_text)
                    target_texts.append(target_text)

                bleus = sum(bleus) / len(bleus)
                total_bleus.append(bleus)

        total_mean_bleus = sum(total_bleus) / len(total_bleus)

        wandb.log({"valid_bleu": total_mean_bleus})

        self._save_results(pred_texts, target_texts, f"val_{epoch}.csv")

    def fit(self):
        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch} start..")
            self._train_epoch()

            logger.info(f"Validate..")
            self._validate(epoch)
