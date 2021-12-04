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
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from dataset import GlueMNLIDataset, collate_fn
from trainer import NaturalLanguageInferenceTrainer


def _get_parser():
    parser = argparse.ArgumentParser(description='NLP Tasks - Natural Language Inference')
    # Basic arguments
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    # Model arguments
    parser.add_argument('--pretrain_model_name', type=str, default='tunib/electra-ko-en-small')
    parser.add_argument('--num_labels', type=int, default=1)
    parser.add_argument('--dropout_p', type=float, default=0.1)
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
    wandb.init(project="NLP Tasks - Natural Language Inference")

    parser = _get_parser()
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    datasets = load_dataset('glue', 'mnli')

    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_model_name)
    model = nn.DataParallel(
        AutoModelForSequenceClassification.from_pretrained(args.pretrain_model_name, num_labels=3)
    ).to(device)

    train_dataset = GlueMNLIDataset(datasets["train"], tokenizer)
    valid_dataset = GlueMNLIDataset((datasets["validation_matched"], datasets["validation_mismatched"]), tokenizer)
    test_dataset = GlueMNLIDataset((datasets["test_matched"], datasets["test_mismatched"]), tokenizer)

    train_loader = DataLoader(dataset=train_dataset,
                              num_workers=args.num_workers,
                              collate_fn=partial(collate_fn, pad_token_id=tokenizer.pad_token_id),
                              batch_size=args.batch_size,
                              drop_last=False)
    valid_loader = DataLoader(dataset=valid_dataset,
                              num_workers=args.num_workers,
                              collate_fn=partial(collate_fn, pad_token_id=tokenizer.pad_token_id),
                              batch_size=args.batch_size,
                              drop_last=False)
    test_loader = DataLoader(dataset=test_dataset,
                             num_workers=args.num_workers,
                             collate_fn=partial(collate_fn, pad_token_id=tokenizer.pad_token_id),
                             batch_size=args.batch_size,
                             drop_last=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer,
                                                    max_lr=args.lr,
                                                    epochs=args.num_epochs,
                                                    steps_per_epoch=len(train_loader),
                                                    anneal_strategy='linear')

    trainer = NaturalLanguageInferenceTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
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
