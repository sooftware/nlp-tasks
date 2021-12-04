# MIT License
# code by Soohwan Kim @sooftware

import wandb
import time
import os
import torch
import logging
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import label_ranking_average_precision_score

logger = logging.getLogger(__name__)


class DialogueRetrievalTrainer:
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
        super(DialogueRetrievalTrainer, self).__init__()
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

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def _train_epoch(self):
        total_loss = 0.0

        self.model.train()

        for step, batch in enumerate(tqdm(self.train_loader)):
            contexts, context_attention_masks, responses, response_attention_masks = batch
            batch_size = contexts.size(0)

            contexts = contexts.to(self.device)
            context_attention_masks = context_attention_masks.to(self.device)
            responses = responses.to(self.device)
            response_attention_masks = response_attention_masks.to(self.device)

            scores = self.model(contexts=contexts,
                                context_attention_masks=context_attention_masks,
                                responses=responses,
                                response_attention_masks=response_attention_masks)

            mask = torch.eye(batch_size).to(self.device)  
            loss = F.log_softmax(scores, dim=-1) * mask
            loss = (-loss.sum(dim=1)).mean()

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
                wandb.log({"lr": self.get_lr(),
                           "train_loss": train_mean_loss,
                           "batch_size": batch_size})

            if step % self.save_every == 0:
                if not os.path.exists(self.save_dir):
                    os.mkdir(self.save_dir)
                date_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())

                os.mkdir(os.path.join(self.save_dir, date_time))
                torch.save(self.model, os.path.join(self.save_dir, date_time, "model.pt"))

    def _validate(self, epoch):
        r10 = r2 = r1 = r5 = 0
        num_total_examples = 0
        mrr = list()

        self.model.eval()

        with torch.no_grad():
            for step, batch in enumerate(tqdm(self.valid_loader)):
                contexts, context_attention_masks, responses, response_attention_masks, labels = batch

                contexts = contexts.to(self.device)
                context_attention_masks = context_attention_masks.to(self.device)
                responses = responses.to(self.device)
                response_attention_masks = response_attention_masks.to(self.device)

                scores = self.model(contexts=contexts,
                                    context_attention_masks=context_attention_masks,
                                    responses=responses,
                                    response_attention_masks=response_attention_masks)

                r2_indices = torch.topk(scores, 2)[1]
                r5_indices = torch.topk(scores, 5)[1]
                r10_indices = torch.topk(scores, 10)[1]
                r1 += (scores.argmax(-1) == 0).sum().item()
                r2 += ((r2_indices == 0).sum(-1)).sum().item()
                r5 += ((r5_indices == 0).sum(-1)).sum().item()
                r10 += ((r10_indices == 0).sum(-1)).sum().item()

                logits = logits.data.cpu().numpy()
                for logit in logits:
                    y_true = np.zeros(len(logit))
                    y_true[0] = 1
                    mrr.append(label_ranking_average_precision_score([y_true], [logit]))

                num_total_examples += contexts.size(0)

        wandb.log({"R1": r1 / num_total_examples,
                   "R2": r2 / num_total_examples,
                   "R5": r5 / num_total_examples,
                   "R10": r10 / num_total_examples,
                   "MRR": np.mean(mrr)})

    def fit(self):
        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch} start..")
            self._train_epoch()

            logger.info(f"Validate..")
            self._validate(epoch)
