# MIT License
# code by Soohwan Kim @sooftware

import argparse
import torch
import torch.nn as nn
import numpy as np
import random
import wandb
from functools import partial
from torch.utils.data import DataLoader

from dataset import load_datas, TranslationDataset, collate_fn
from utils import train_tokenizer
from model import Transformer
from trainer import MachineTranslationTrainer


def _get_parser():
    parser = argparse.ArgumentParser(description='NLP Tasks - Machine Translation')
    # Basic arguments
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    # Model arguments
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_encoder_layers', type=int, default=6)
    parser.add_argument('--num_decoder_layers', type=int, default=6)
    parser.add_argument('--feed_forward_dim', type=int, default=2048)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--dropout_p', type=float, default=0.1)
    # Tokenizer arguments
    parser.add_argument('--source_vocab_size', type=int, default=8000)
    parser.add_argument('--target_vocab_size', type=int, default=32000)
    parser.add_argument('--source_tokenizer_model_name', type=str, default="source")
    parser.add_argument('--target_tokenizer_model_name', type=str, default="target")
    parser.add_argument('--source_tokenizer_model_type', type=str, default="unigram")
    parser.add_argument('--target_tokenizer_model_type', type=str, default="unigram")
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
    wandb.init(project="NLP Tasks - Machine Translation")

    parser = _get_parser()
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_sources, train_targets, valid_sources, valid_targets = load_datas(args.data_dir, args.valid_ratio)

    source_tokenizer = train_tokenizer(texts=train_sources,
                                       vocab_size=args.source_vocab_size,
                                       model_name=args.source_tokenizer_model_name,
                                       model_type=args.source_tokenizer_model_type)
    target_tokenizer = train_tokenizer(texts=train_targets,
                                       vocab_size=args.target_vocab_size,
                                       model_name=args.target_tokenizer_model_name,
                                       model_type=args.target_tokenizer_model_type)

    train_dataset = TranslationDataset(sources=train_sources,
                                       targets=train_targets,
                                       source_tokenizer=source_tokenizer,
                                       target_tokenizer=target_tokenizer)
    valid_dataset = TranslationDataset(sources=valid_sources,
                                       targets=valid_targets,
                                       source_tokenizer=source_tokenizer,
                                       target_tokenizer=target_tokenizer)

    train_loader = DataLoader(dataset=train_dataset,
                              num_workers=args.num_workers,
                              collate_fn=partial(collate_fn, pad_token_id=source_tokenizer.pad_id()),
                              batch_size=args.batch_size,
                              drop_last=False)
    valid_loader = DataLoader(dataset=valid_dataset,
                              num_workers=args.num_workers,
                              collate_fn=partial(collate_fn, pad_token_id=target_tokenizer.pad_id()),
                              batch_size=args.batch_size,
                              drop_last=False)

    model = nn.DataParallel(Transformer(args.source_vocab_size, args.target_vocab_size))
    wandb.watch(model)

    criterion = nn.CrossEntropyLoss(ignore_index=target_tokenizer.PieceToId('<pad>'), reduction='mean')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer,
                                                    max_lr=args.lr,
                                                    epochs=args.num_epochs,
                                                    steps_per_epoch=len(train_loader),
                                                    anneal_strategy='linear')

    MachineTranslationTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.num_epochs,
        device=device,
        tokenizer=target_tokenizer,
        gradient_clip_val=args.gradient_clip_val,
        log_every=args.log_every,
        save_every=args.save_every,
        save_dir=args.save_dir,
        accumulate_grad_batches=args.accumulate_grad_batches,
    ).fit()


if __name__ == '__main__':
    main()

