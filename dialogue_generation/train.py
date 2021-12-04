# MIT License
# code by Soohwan Kim @sooftware

import argparse
import torch
import torch.nn as nn
import numpy as np
import random
import wandb
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, AutoTokenizer
from functools import partial

from dataset import DialogueDataset, collate_fn, load_dataset
from trainer import DialogueGenerationTrainer


def _get_parser():
    parser = argparse.ArgumentParser(description='NLP Tasks - Dialogue Generation')
    # Basic arguments
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--sep_token', type=str, default='<sep>')
    # Model arguments
    parser.add_argument('--pretrain_model_name', type=str, default='skt/kogpt2-base-v2')
    # Training arguments
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--valid_ratio', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=3e-05)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--log_every', type=int, default=20)
    parser.add_argument('--gradient_clip_val', type=float, default=1.0)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--save_every', type=int, default=10_000)
    return parser


def main():
    data_loaders = dict()

    wandb.init(project="NLP Tasks - Dialogue Generation")

    parser = _get_parser()
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_model_name,
                                              bos_token='</s>',
                                              eos_token='</s>',
                                              unk_token='<unk>',
                                              pad_token='<pad>',
                                              mask_token='<mask>')
    tokenizer.add_tokens([args.sep_token])

    model = GPT2LMHeadModel.from_pretrained(args.pretrain_model_name)
    model.resize_token_embeddings(len(tokenizer))
    model = nn.DataParallel(model).to(device)

    datasets = load_dataset(args.data_dir, args.sep_token, args.eos_token, args.valid_ratio)

    for dataset_name in datasets.keys():
        dataset = DialogueDataset(datasets[dataset_name], tokenizer)
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

    DialogueGenerationTrainer(
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
