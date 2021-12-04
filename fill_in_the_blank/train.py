# MIT License
# code by Soohwan Kim @sooftware

import argparse
import wandb
import torch
import numpy as np
import random
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    BertConfig,
    TrainingArguments,
    Trainer,
)

from dataset import FillMaskDataset


def _get_parser():
    parser = argparse.ArgumentParser(description='NLP Tasks - Fill in the Blank')
    # Basic arguments
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    # Model arguments
    parser.add_argument('--pretrain_tokenizer_name', type=str, default='bert-base-uncased')
    # Training arguments
    parser.add_argument('--valid_ratio', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-05)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--log_every', type=int, default=20)
    parser.add_argument('--gradient_clip_val', type=float, default=1.0)
    parser.add_argument('--save_every', type=int, default=10_000)
    return parser


def main():
    wandb.init(project="NLP Tasks - Fill in the Blank")

    parser = _get_parser()
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = load_dataset('wikicorpus', 'raw_en')
    dataset = dataset["train"]

    num_total_datas = len(dataset)
    num_valid_datas = int(num_total_datas * args.valid_ratio)

    train_datas = dataset[:-num_valid_datas]
    valid_datas = dataset[-num_valid_datas:]

    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_tokenizer_name)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    train_dataset = FillMaskDataset(train_datas, tokenizer)
    valid_dataset = FillMaskDataset(valid_datas, tokenizer)

    model = nn.DataParallel(BertForMaskedLM(BertConfig(vocab_size=len(tokenizer)))).to(device)

    per_gpu_batch_size = args.batch_size // torch.cuda.device_count()

    training_args = TrainingArguments(output_dir=args.save_dir,
                                      overwrite_output_dir=True,
                                      num_train_epochs=args.num_epochs,
                                      per_gpu_train_batch_size=per_gpu_batch_size,
                                      save_steps=args.save_every,
                                      report_to='wandb',
                                      prediction_loss_only=True)

    trainer = Trainer(model=model,
                      args=training_args,
                      data_collator=data_collator,
                      train_dataset=train_dataset,
                      eval_dataset=valid_dataset)
    trainer.train()
    trainer.evaluate(valid_dataset)
