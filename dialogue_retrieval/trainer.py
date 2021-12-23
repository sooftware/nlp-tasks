# MIT License
# code by Soohwan Kim @sooftware

import wandb
import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import label_ranking_average_precision_score
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from typing import Dict

from utils import mkdir


class Trainer:
    def __init__(
            self,
            model: nn.Module,
            data_loaders: Dict[str, DataLoader],
            optimizer: nn.Module,
            scheduler: LambdaLR,
            num_epochs: int,
            accumulate_grad_batches: int,
            max_grad_norm: int,
            print_every: int,
            eval_every: int,
            output_dir: str,
            device: torch.device,
    ) -> None:
        self.model = model
        self.data_loaders = data_loaders
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.accumulate_grad_batches = accumulate_grad_batches
        self.max_grad_norm = max_grad_norm
        self.print_every = print_every
        self.eval_every = eval_every
        self.global_step = 1
        self.best_eval_loss = 100.0
        self.output_dir = output_dir
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.device = device

    def _train_epoch(self, epoch):
        train_loss = 0
        update_steps = 0

        with tqdm(total=len(self.data_loaders["train"]) // self.accumulate_grad_batches) as bar:
            for step, batch in enumerate(self.data_loaders["train"]):
                self.model.train()
                self.optimizer.zero_grad()

                contexts, context_masks, responses, response_masks, labels = batch

                loss = self.model(
                    contexts.to(self.device),
                    context_masks.to(self.device),
                    responses.to(self.device),
                    response_masks.to(self.device),
                    labels.to(self.device),
                )

                loss = loss / self.accumulate_grad_batches
                loss.backward()

                train_loss += loss.item()

                if (step + 1) % self.accumulate_grad_batches == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    update_steps += 1
                    self.optimizer.step()
                    self.scheduler.step()
                    self.model.zero_grad()
                    self.global_step += 1

                    if update_steps and update_steps % self.print_every == 0:
                        bar.update(min(self.print_every, update_steps))
                        wandb.log({"train_loss": train_loss / update_steps})

                    if self.global_step and self.global_step % self.eval_every == 0:
                        eval_loss = self._evaluate()

                        if eval_loss < self.best_eval_loss:
                            self.best_eval_loss = eval_loss
                            save_dir = os.path.join(self.output_dir, f"epoch_{epoch}_step_{step}")
                            mkdir(save_dir)
                            torch.save(self.model.module.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))

    def _evaluate(self):
        mrr = list()
        eval_loss, eval_hit_times = 0, 0
        num_eval_examples = 0
        r10 = r2 = r1 = r5 = 0

        self.model.eval()

        for step, batch in enumerate(self.data_loaders["valid"]):
            contexts, context_masks, responses, response_masks, labels = batch

            with torch.no_grad():
                logits = self.model(
                    contexts.to(self.device),
                    context_masks.to(self.device),
                    responses.to(self.device),
                    response_masks.to(self.device),
                )
                loss = self.criterion(logits, torch.argmax(labels.to(self.device), 1))

            r2_indices = torch.topk(logits, 2)[1]
            r5_indices = torch.topk(logits, 5)[1]
            r10_indices = torch.topk(logits, 10)[1]

            r1 += (logits.argmax(-1) == 0).sum().item()
            r2 += ((r2_indices == 0).sum(-1)).sum().item()
            r5 += ((r5_indices == 0).sum(-1)).sum().item()
            r10 += ((r10_indices == 0).sum(-1)).sum().item()

            logits = logits.data.cpu().numpy()
            for logit in logits:
                target = np.zeros(len(logit))
                target[0] = 1
                mrr.append(label_ranking_average_precision_score([target], [logit]))

            eval_loss += loss.item()
            num_eval_examples += labels.size(0)

        eval_loss = eval_loss / len(self.data_loaders["valid"])

        wandb.log({
            'eval_loss': eval_loss,
            'R1': r1 / num_eval_examples,
            'R2': r2 / num_eval_examples,
            'R5': r5 / num_eval_examples,
            'R10': r10 / num_eval_examples,
            'MRR': np.mean(mrr),
            'global_step': self.global_step,
        })
        return eval_loss

    def fit(self):
        for epoch in range(self.num_epochs):
            self._train_epoch(epoch)
            self._evaluate()
