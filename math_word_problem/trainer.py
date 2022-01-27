# MIT License
# code by Soohwan Kim @sooftware

import wandb
import os
import torch
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import code_to_result

logger = logging.getLogger(__name__)


class Trainer:
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
            sep_token: str,
            log_every: int = 20,
            save_dir: str = 'ckpt',
    ) -> None:
        super(Trainer, self).__init__()
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.log_every = log_every
        self.num_epochs = num_epochs
        self.tokenizer = tokenizer
        self.eos_token = tokenizer.eos_token
        self.sep_token = sep_token
        self.device = device
        self.gradient_clip_val = gradient_clip_val
        self.accumulate_grad_batches = accumulate_grad_batches
        self.save_dir = save_dir

    def _get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def _save_result(self, inputs, codes, predicts, answers, epoch):
        pd.DataFrame({
            "inputs": inputs,
            "generate_codes": codes,
            "predicts": predicts,
            "answers": answers,
        }).to_csv(f'{epoch}_result.csv', index=False)

    def _train_epoch(self):
        total_loss = 0.0

        self.model.train()

        for step, (inputs, attention_masks, labels, answers) in enumerate(tqdm(self.train_loader)):
            inputs = inputs.to(self.device)
            attention_masks = attention_masks.to(self.device)
            labels = labels.to(self.device)

            batch_size = inputs.size(0)

            outputs = self.model(input_ids=inputs, attention_mask=attention_masks, labels=labels)
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
                           "batch_size": batch_size})

    def _get_accuracy(self, predicts, answers):
        accuracy = 0.0
        num = len(predicts)

        for pred, answer in zip(predicts, answers):
            if pred == answer:
                accuracy += 1

        accuracy /= num

        return accuracy

    def _validate(self, epoch):
        all_inputs = list()
        all_codes = list()
        all_predicts = list()
        all_answers = list()

        self.model.eval()

        with torch.no_grad():
            for step, (input_ids, attention_masks, labels, answers) in enumerate(tqdm(self.valid_loader)):
                input_ids = input_ids.to(self.device)
                attention_masks = attention_masks.to(self.device)

                generated = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_masks,
                    max_length=300,
                )
                generated = self.tokenizer.batch_decode(generated, skip_special_tokens=False)

                for g, a in zip(generated, answers):
                    splited = g.split(self.sep_token)
                    input_var, code = splited[0], splited[1].split(self.eos_token)[0]
                    code = code.replace('<pad>', '').replace('<s>', '').replace('<unk>', '').strip()
                    try:
                        predict = code_to_result(code)
                    except:
                        predict = None

                    all_inputs.append(input_var)
                    all_codes.append(code)
                    if predict is not None:
                        all_predicts.append(predict)
                    else:
                        all_predicts.append(np.nan)
                    all_answers.append(a)

        accuracy = self._get_accuracy(all_predicts, all_answers)
        wandb.log({"valid_accuracy": accuracy})
        self._save_result(all_inputs, all_codes, all_predicts, all_answers, epoch)

        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        self.model.save_pretrained(os.path.join(self.save_dir, f"epoch_{epoch}"))

    def fit(self):
        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch} start..")
            self._train_epoch()

            logger.info(f"Validate..")
            self._validate(epoch)

        self.model.save_pretrained(os.path.join(self.save_dir, "LAST_CHECKPOINT"))
