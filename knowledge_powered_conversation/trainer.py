# MIT License
# code by Soohwan Kim @sooftware

import wandb
import numpy as np
import torch
import os
import time
from tqdm import tqdm

from knowledge_powered_conversation.utils import f1_score


class Trainer:
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
        super(Trainer, self).__init__()
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

        self.model.train()

        for step, (inputs, encoder_attention_masks, labels, decoder_attention_masks) in enumerate(
                tqdm(self.train_loader)):
            inputs = inputs.to(self.device)
            encoder_attention_masks = encoder_attention_masks.to(self.device)
            labels = labels.to(self.device)
            decoder_attention_masks = decoder_attention_masks.to(self.device)

            batch_size = inputs.size(0)

            outputs = self.model(input_ids=inputs, attention_mask=encoder_attention_masks, labels=labels,
                                 decoder_attention_mask=decoder_attention_masks)
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

            if step % self.save_every == 0 and step != 0:
                if not os.path.exists(self.save_dir):
                    os.mkdir(self.save_dir)
                date_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
                self.model.module.save_pretrained(os.path.join(self.save_dir, date_time))

    def _validate(self, epoch):
        valid_loss = 0.0

        f1_scores = list()

        self.model.eval()

        with torch.no_grad():
            for step, (inputs, encoder_attention_masks, labels, decoder_attention_masks) in enumerate(
                    tqdm(self.valid_loader)):
                inputs = inputs.to(self.device)
                encoder_attention_masks = encoder_attention_masks.to(self.device)
                labels = labels.to(self.device)
                decoder_attention_masks = decoder_attention_masks.to(self.device)

                outputs = self.model(input_ids=inputs,
                                     attention_mask=encoder_attention_masks,
                                     labels=labels,
                                     decoder_attention_mask=decoder_attention_masks)
                loss = outputs.loss
                loss = loss.mean()

                generated = self.model.module.generate(input_ids=inputs,
                                                       num_beams=5,
                                                       no_repeat_ngram_size=3,
                                                       early_stopping=True)
                y_preds = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
                y_trues = self.tokenizer.batch_decode(labels)

                valid_loss += loss.item()

                score = f1_score(y_trues, y_preds)
                f1_scores.extend(score)

        valid_mean_loss = valid_loss / len(self.valid_loader)
        valid_ppl = np.exp(valid_mean_loss)
        f1_score_ = np.mean(f1_scores)

        wandb.log({"valid_loss": valid_mean_loss,
                   "valid_ppl": valid_ppl,
                   "valid_f1_score": f1_score_})

    def _test(self, epoch):
        valid_loss = 0.0

        f1_scores = list()

        self.model.eval()

        with torch.no_grad():
            for step, (inputs, encoder_attention_masks, labels, decoder_attention_masks) in enumerate(tqdm(self.test_loader)):
                inputs = inputs.to(self.device)
                encoder_attention_masks = encoder_attention_masks.to(self.device)
                labels = labels.to(self.device)
                decoder_attention_masks = decoder_attention_masks.to(self.device)

                outputs = self.model(input_ids=inputs,
                                     attention_mask=encoder_attention_masks,
                                     labels=labels,
                                     decoder_attention_mask=decoder_attention_masks)
                loss = outputs.loss
                loss = loss.mean()

                generated = self.model.module.generate(input_ids=inputs,
                                                       num_beams=5,
                                                       no_repeat_ngram_size=3,
                                                       early_stopping=True).tolist()
                y_preds = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
                y_trues = self.tokenizer.batch_decode(labels)

                valid_loss += loss.item()

                f1_scores.extend(f1_score(y_trues, y_preds))

        valid_mean_loss = valid_loss / len(self.valid_loader)
        valid_ppl = np.exp(valid_mean_loss)
        f1_score_ = np.mean(f1_scores)

        wandb.log({"test_loss": valid_mean_loss,
                   "test_ppl": valid_ppl,
                   "test_f1_score": f1_score_})

    def fit(self):
        for epoch in range(self.num_epochs):
            self._train_epoch()
            self._validate(epoch)
            self._test(epoch)
