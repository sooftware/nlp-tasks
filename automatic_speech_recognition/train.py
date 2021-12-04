# MIT License
# code by Soohwan Kim @sooftware

import argparse
import torch
import torch.nn as nn
import numpy as np
import random
import wandb
from torch.utils.data import DataLoader
from functools import partial

from preprocessor import KsponSpeechPreprocessor
from dataset import SpeechToTextDataset, collate_fn
from model import ListenAttendSpell
from trainer import AutomaticSpeechRecognitionTrainer
from tokenizer import KsponSpeechTokenizer


def _get_parser():
    parser = argparse.ArgumentParser(description='NLP Tasks - Automatic Speech Recognition')
    # Basic arguments
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--test_manifest_path', type=str, required=True)
    parser.add_argument('--test_data_dir', type=str, required=True)
    parser.add_argument('--preprocess_mode', type=str, default='phonetic')
    parser.add_argument('--vocab_path', type=str, default='aihub_labels.csv')
    parser.add_argument('--manifest_file_path', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    # Training arguments
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_mels', type=int, default=40)
    parser.add_argument('--vocab_size', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-05)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--log_every', type=int, default=20)
    parser.add_argument('--gradient_clip_val', type=float, default=1.0)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--save_every', type=int, default=10_000)
    return parser


def main():
    dataset, data_loader = dict(), dict()

    wandb.init(project="NLP Tasks - Automatic Speech Recognition")

    parser = _get_parser()
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    audio_paths, transcripts = KsponSpeechPreprocessor(dataset_path=args.data_dir,
                                                       preprocess_mode=args.preprocess_mode,
                                                       test_manifest_path=args.test_manifest_path,
                                                       vocab_path=args.vocab_path,
                                                       vocab_size=args.vocab_size).setup(args.manifest_file_path)

    tokenizer = KsponSpeechTokenizer(args.vocab_path,
                                     pad_token='<pad>',
                                     sos_token='<sos>',
                                     eos_token='<eos>',
                                     blank_token='<blank>')

    for stage in ("train", "valid", "test"):
        dataset[stage] = SpeechToTextDataset(data_dir=args.data_dir if stage != "test" else args.test_data_dir,
                                             audio_paths=audio_paths[stage],
                                             transcripts=transcripts[stage],
                                             tokenizer=tokenizer,
                                             num_mels=args.num_mels)
        data_loader[stage] = DataLoader(dataset=dataset[stage],
                                        num_workers=args.num_workers,
                                        collate_fn=partial(collate_fn, pad_token_id=tokenizer.pad_id),
                                        batch_size=args.batch_size,
                                        drop_last=False)

    model = nn.DataParallel(ListenAttendSpell(input_dim=args.num_mels, num_classes=args.vocab_size)).to(device)
    wandb.watch(model)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id, reduction='mean')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer,
                                                    max_lr=args.lr,
                                                    epochs=args.num_epochs,
                                                    steps_per_epoch=len(data_loader["train"]),
                                                    anneal_strategy='linear')

    trainer = AutomaticSpeechRecognitionTrainer(
        model=model,
        train_loader=data_loader["train"],
        valid_loader=data_loader["valid"],
        test_loader=data_loader["test"],
        criterion=criterion,
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

