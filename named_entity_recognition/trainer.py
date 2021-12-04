# MIT License
# code by Soohwan Kim @sooftware

import wandb
import time
import os
import torch
import logging
from tqdm import tqdm
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)


class NamedEntityRecognitionTrainer:
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
        super(NamedEntityRecognitionTrainer, self).__init__()
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.log_every = log_every
        self.save_every = save_every
        self.num_epochs = num_epochs
        self.tokenizer = tokenizer
        self.device = device
        self.gradient_clip_val = gradient_clip_val
        self.accumulate_grad_batches = accumulate_grad_batches
        self.save_dir = save_dir
        self.class_names = [
            '-', 'AFW', 'ANM', 'CVL', 'DAT', 'EVT', 'FLD', 'LOC', 'MAT', 'NUM', 'ORG', 'PER', 'PLT', 'TIM', 'TRM',
        ]

    def _get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def _train_epoch(self):
        all_predicts, all_labels = list(), list()
        total_loss, count = 0.0, 0

        self.model.train()

        for step, batch in enumerate(tqdm(self.train_loader)):
            input_ids, labels, lengths = batch

            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(input_ids=input_ids, labels=labels)

            loss = outputs.loss
            loss = loss / self.accumulate_grad_batches

            for idx, (predict, target) in enumerate(zip(outputs.logits.argmax(dim=2), labels)):
                all_predicts.extend(predict.tolist()[:lengths[idx]])
                all_labels.extend(target.tolist()[:lengths[idx]])

            self.optimizer.zero_grad()
            loss.backward()
            total_loss += loss.item()

            if step % self.accumulate_grad_batches == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                self.optimizer.step()
                self.scheduler.step()

            if step % self.log_every == 0:
                train_mean_loss = total_loss / (step + 1)
                train_f1_score = f1_score(all_labels, all_predicts, average='macro')
                wandb.log(
                    {
                        "lr": self._get_lr(),
                        "train_loss": train_mean_loss,
                        "batch_size": input_ids.size(0),
                        "train_f1_score": train_f1_score,
                        "train_confusion_matrix": wandb.plot.confusion_matrix(
                            probs=None,
                            y_true=all_labels,
                            preds=all_predicts,
                            class_names=self.class_names,
                        )
                     }
                )

            if step % self.save_every == 0:
                if not os.path.exists(self.save_dir):
                    os.mkdir(self.save_dir)
                date_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
                self.model.module.save_pretrained(os.path.join(self.save_dir, date_time))

    def _validate(self, epoch):
        all_predicts, all_labels = list(), list()
        valid_loss, count = 0.0, 0

        self.model.eval()

        with torch.no_grad():
            for batch in tqdm(self.valid_loader):
                input_ids, labels, lengths = batch

                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs.loss

                for idx, (predict, target) in enumerate(zip(outputs.logits.argmax(dim=2), labels)):
                    all_predicts.extend(predict.tolist()[:lengths[idx]])
                    all_labels.extend(target.tolist()[:lengths[idx]])

                valid_loss += loss.item()

        valid_mean_loss = valid_loss / len(self.valid_loader)
        valid_f1_score = f1_score(all_labels, all_predicts, average='macro')

        wandb.log(
            {
                "valid_mean_loss": valid_mean_loss,
                "valid_f1_score": valid_f1_score,
                "valid_confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=all_labels,
                    preds=all_predicts,
                    class_names=self.class_names,
                )
            }
        )

    def fit(self):
        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch} start..")
            self._train_epoch()

            logger.info(f"Validate..")
            self._validate(epoch)
