# MIT License
# code by Soohwan Kim @sooftware

import argparse
import torch
import numpy as np
import random
import wandb
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from functools import partial

from dataset import DialogueRetrievalDataset, collate_fn
from model import PolyEncoderModel
from trainer import DialogueRetrievalTrainer


def _get_parser():
    parser = argparse.ArgumentParser(description='NLP Tasks - Dialogue Retrieval')
    # Basic arguments
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--train_data_path', type=str, required=True)
    parser.add_argument('--valid_data_path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    # Model arguments
    parser.add_argument('--pretrain_model_name_or_path', type=str, default='tunib/electra-ko-base')
    parser.add_argument('--num_poly_codes', type=int, default=64)
    # Training arguments
    parser.add_argument('--valid_sample_count', type=float, default=1000)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-05)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--log_every', type=int, default=20)
    parser.add_argument('--gradient_clip_val', type=float, default=1.0)
    parser.add_argument('--save_every', type=int, default=10_000)
    parser.add_argument('--accumulate_grad_batches', type=int, default=4)
    return parser


def main():
    data_loaders, datas = dict(), dict()

    wandb.init(project="NLP Tasks - Dialogue Retrieval")

    parser = _get_parser()
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_model_name_or_path)
    model = nn.DataParallel(PolyEncoderModel(args.pretrain_model_name_or_path, args.num_poly_codes)).to(device)

    for dataset_name in ("train", "valid"):
        dataset = DialogueRetrievalDataset(args.train_data_path if dataset_name == 'train' else args.valid_data_path,
                                           tokenizer=tokenizer,
                                           sample_cnt=None if dataset_name == 'train' else args.valid_sample_count)
        data_loaders[dataset_name] = DataLoader(dataset=dataset,
                                                num_workers=args.num_workers,
                                                collate_fn=partial(
                                                    collate_fn,
                                                    pad_token_id=tokenizer.pad_token_id,
                                                    is_training=True if dataset_name == 'train' else False,
                                                ),
                                                batch_size=args.batch_size,
                                                drop_last=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer,
                                                    max_lr=args.lr,
                                                    epochs=args.num_epochs,
                                                    steps_per_epoch=len(data_loaders["train"]),
                                                    anneal_strategy='linear')

    DialogueRetrievalTrainer(
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
