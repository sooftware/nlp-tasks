# MIT License
# code by Soohwan Kim @sooftware

import argparse
import torch
import torch.nn as nn
import numpy as np
import random
import wandb
from datasets import load_dataset
from torch.utils.data import DataLoader
from functools import partial
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config

from dataset import LanguageModelDataset, collate_fn
from trainer import LanguageModelingTrainer


def _get_parser():
    parser = argparse.ArgumentParser(description='NLP Tasks - Language Modeling')
    # Basic arguments
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    # Model arguments
    parser.add_argument('--pretrain_tokenizer_name', type=str, default='gpt2-medium')
    # Training arguments
    parser.add_argument('--valid_ratio', type=float, default=0.1)
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
    data_loaders, datas = dict(), dict()

    wandb.init(project="NLP Tasks - Language Modeling")

    parser = _get_parser()
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = load_dataset('openwebtext')
    dataset = dataset["train"]

    num_total_datas = len(dataset)
    num_valid_datas = int(num_total_datas * args.valid_ratio)

    datas["train"] = dataset[:-num_valid_datas]
    datas["valid"] = dataset[-num_valid_datas:]

    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_model_name)
    model = nn.DataParallel(GPT2LMHeadModel(GPT2Config(vocab_size=len(tokenizer)))).to(device)
    wandb.watch(model)

    for data_name in datas.keys():
        dataset = LanguageModelDataset(datas[data_name], tokenizer)
        data_loaders[data_name] = DataLoader(dataset=dataset,
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

    LanguageModelingTrainer(
        model=model,
        train_loader=data_loaders["train"],
        valid_loader=data_loaders["valid"],
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
    ).fit()


if __name__ == '__main__':
    main()

