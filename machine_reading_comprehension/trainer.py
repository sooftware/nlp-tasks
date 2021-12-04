# MIT License
# code by Soohwan Kim @sooftware

import wandb
import time
import os
import torch
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class MachineReadingComprehensionTrainer:
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
        super(MachineReadingComprehensionTrainer, self).__init__()
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.log_every = log_every
        self.save_every = save_every
        self.num_epochs = num_epochs
        self.tokenizer = tokenizer
        self.sos_id = tokenizer.sos_token_id
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
        count = 0
        exact_match_score = 0.0

        self.model.train()

        for step, (inputs, attention_masks, start_positions, end_positions, answers) \
                in enumerate(tqdm(self.train_loader)):
            inputs = inputs.to(self.device)
            attention_masks = attention_masks.to(self.device)
            start_positions = start_positions.to(self.device)
            end_positions = end_positions.to(self.device)

            batch_size = inputs.size(0)

            outputs = self.model(input_ids=inputs,
                                 attention_mask=attention_masks,
                                 start_positions=start_positions,
                                 end_positions=end_positions)

            loss = outputs.loss
            loss = loss / self.accumulate_grad_batches

            self.optimizer.zero_grad()
            loss.backward()
            total_loss += loss.item()

            if step % self.accumulate_grad_batches == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                self.optimizer.step()
                self.scheduler.step()

            pred_starts = torch.argmax(outputs.start_logits, dim=1)
            answer_ends = torch.argmax(outputs.end_logits, dim=1)

            for pred_start, pred_end, target_start, target_end \
                    in zip(pred_starts, answer_ends, start_positions, end_positions):
                count += 1
                if pred_start.item() == target_start.item() and pred_end.item() == target_end.item():
                    exact_match_score += 1.0

            if step % self.log_every == 0:
                train_mean_loss = total_loss / (step + 1)
                exact_match_mean_score = exact_match_score / count
                wandb.log({"lr": self._get_lr(),
                           "train_loss": train_mean_loss,
                           "batch_size": batch_size,
                           "train_em_score": exact_match_mean_score * 100})

            if step % self.save_every == 0:
                if not os.path.exists(self.save_dir):
                    os.mkdir(self.save_dir)
                date_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
                self.model.save_pretrained(os.path.join(self.save_dir, date_time))

    def _validate(self, epoch):
        valid_loss = 0.0
        exact_match_score = 0.0
        count = 0

        self.model.eval()

        with torch.no_grad():
            for inputs, attention_masks, start_positions, end_positions, answers in tqdm(self.valid_loader):
                inputs = inputs.to(self.device)
                attention_masks = attention_masks.to(self.device)
                start_positions = start_positions.to(self.device)
                end_positions = end_positions.to(self.device)

                outputs = self.model(input_ids=inputs,
                                     attention_mask=attention_masks,
                                     start_positions=start_positions,
                                     end_positions=end_positions)
                loss = outputs.loss

                pred_starts = torch.argmax(outputs.start_logits, dim=1)
                answer_ends = torch.argmax(outputs.end_logits, dim=1)

                for pred_start, pred_end, target_start, target_end \
                        in zip(pred_starts, answer_ends, start_positions, end_positions):
                    count += 1
                    if pred_start.item() == target_start.item() and pred_end.item() == target_end.item():
                        exact_match_score += 1.0

                valid_loss += loss.item()

        valid_mean_loss = valid_loss / len(self.valid_loader)
        exact_match_score /= count

        wandb.log({"valid_loss": valid_mean_loss,
                   "valid_em_score": exact_match_score * 100})

    def fit(self):
        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch} start..")
            self._train_epoch()

            logger.info(f"Validate..")
            self._validate(epoch)
