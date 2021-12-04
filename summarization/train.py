# MIT License
# code by Soohwan Kim @sooftware

import argparse
import torch
import torch.nn as nn
import numpy as np
import random
import wandb
from datasets import load_dataset
from functools import partial
from torch.utils.data import DataLoader
from transformers import BartForConditionalGeneration, AutoTokenizer

from dataset import XSumDataset, collate_fn
from trainer import SummarizationTrainer


def _get_parser():
    parser = argparse.ArgumentParser(description='NLP Tasks - Summarization')
    # Basic arguments
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    # Model arguments
    parser.add_argument('--pretrain_model_name', type=str, default='facebook/bart-base')
    # Training arguments
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-05)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--log_every', type=int, default=20)
    parser.add_argument('--gradient_clip_val', type=float, default=1.0)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--save_every', type=int, default=10_000)
    return parser


def main():
    data_loaders = dict()

    wandb.init(project="NLP Tasks - Summarization")

    parser = _get_parser()
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    datasets = load_dataset('xsum')

    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_model_name)
    model = nn.DataParallel(
        BartForConditionalGeneration.from_pretrained(args.pretrain_model_name)
    ).to(device)

    for dataset_name in datasets.keys():
        dataset = XSumDataset(datasets[dataset_name], tokenizer)
        data_loaders[dataset_name] = DataLoader(dataset=dataset,
                                                num_workers=args.num_workers,
                                                collate_fn=partial(collate_fn, pad_token_id=tokenizer.pad_token_id),
                                                batch_size=args.batch_size,
                                                drop_last=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer,
                                                    max_lr=args.lr,
                                                    epochs=args.num_epochs,
                                                    steps_per_epoch=len(data_loaders["train"]),
                                                    anneal_strategy='linear')

    trainer = SummarizationTrainer(
        model=model,
        train_loader=data_loaders["train"],
        valid_loader=data_loaders["validation"],
        test_loader=data_loaders["test"],
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.num_epochs,
        device=device,
        tokenizer=tokenizer,
        gradient_clip_val=args.gradient_clip_val,
        log_every=args.log_every,
        save_every=args.save_every,
        save_dir=args.save_dir,
        accumulate_grad_batches=args.accumulate_grad_batches,
    )
    trainer.fit()
    trainer.test()


if __name__ == '__main__':
    main()
