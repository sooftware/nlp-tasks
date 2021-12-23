# MIT License
# code by Soohwan Kim @sooftware

import os
import wandb
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup

from dataset import RetrievalDataset
from model import PolyEncoderModel
from utils import set_seed, mkdir
from trainer import Trainer


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_model_name_or_path", default='klue/roberta-base', type=str)
    parser.add_argument("--num_poly_codes", default=64, type=int, help="Number of m of polyencoder")
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--train_data_path", default='data/train.txt', type=str)
    parser.add_argument("--valid_data_path", default='data/valid.txt', type=str)
    parser.add_argument("--project_name", default='dialogue-retrieval', type=str)
    parser.add_argument("--run_name", default='kaki', type=str)
    parser.add_argument("--max_context_length", default=512, type=int)
    parser.add_argument("--max_response_length", default=128, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=10, type=int, help="Total batch size for eval.")
    parser.add_argument("--print_every", default=100, type=int, help="Log frequency")
    parser.add_argument("--lr", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--warmup_steps", default=100, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_epochs", default=10, type=int, help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=12345, help="random seed for initialization")
    parser.add_argument('--accumulate_grad_batches', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    return parser.parse_args()


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    data_loaders = dict()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = _get_args()

    mkdir(args.output_dir)

    set_seed(args.seed)

    wandb.init(project=args.project_name, name=args.run_name)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_model_name_or_path, use_auth_token=True)

    for stage in ("train", "valid"):
        dataset = RetrievalDataset(
            args.train_data_path if stage == "train" else args.valid_data_path,
            tokenizer,
            args.max_context_length,
            args.max_response_length,
            sample_cnt=None if stage == "train" else 1000,
        )
        data_loaders[stage] = DataLoader(
            dataset,
            batch_size=args.train_batch_size if stage == "train" else args.eval_batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=True if stage == "train" else False,
            num_workers=args.num_workers,
        )

    steps_per_epoch = len(data_loaders["train"]) // args.accumulate_grad_batches * args.num_epochs

    model = nn.DataParallel(
        PolyEncoderModel(
            pretrain_model_name_or_path=args.pretrain_model_name_or_path,
            num_poly_codes=args.num_poly_codes,
        )
    ).to(device)

    optimizer = AdamW(
        model.parameters(), lr=args.lr, eps=args.adam_epsilon, weight_decay=args.weight_decay
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=steps_per_epoch
    )

    print_every = args.print_every // args.accumulate_grad_batches
    eval_every = min(len(data_loaders["train"]) // 2, 1000)
    eval_every = eval_every // args.accumulate_grad_batches

    trainer = Trainer(
        model=model,
        data_loaders=data_loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.num_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        max_grad_norm=args.max_grad_norm,
        print_every=args.print_every,
        eval_every=eval_every,
        output_dir=args.output_dir,
        device=device,
    )
    trainer.fit()
