# MIT License
# code by Soohwan Kim @sooftware

import os
import pandas as pd
import random
import torch
from torch.utils.data import Dataset


def load_datas(data_dir: str, valid_ratio: float):
    if not os.path.exists(data_dir):
        raise FileNotFoundError

    sources, targets = list(), list()

    for file in os.listdir(data_dir):
        if not file.endswith('xlsx'):
            continue

        df = pd.read_excel(os.path.join(data_dir, file))

        if '한국어' in df.columns:
            sources.extend(list(df['한국어']))
            try:
                targets.extend(list(df['영어 검수']))
            except:
                targets.extend(list(df['영어검수']))
        else:
            sources.extend(list(df['원문']))
            try:
                targets.extend(list(df['Review']))
            except:
                try:
                    targets.extend(list(df['REVIEW']))
                except:
                    targets.extend(list(df['번역문']))

    zipped = list(zip(sources, targets))
    random.shuffle(zipped)
    sources, targets = list(zip(*zipped))

    num_total_datas = len(sources)
    num_valid_datas = int(num_total_datas * valid_ratio)

    train_sources = sources[:-num_valid_datas]
    train_targets = targets[:-num_valid_datas]

    valid_sources = sources[-num_valid_datas:]
    valid_targets = targets[-num_valid_datas:]

    return train_sources, train_targets, valid_sources, valid_targets


def collate_fn(batch, pad_token_id):
    r"""
    Functions that pad to the maximum sequence length

    Args:
        batch (tuple): tuple contains input and target tensors

    Returns:
        seqs (torch.FloatTensor): tensor contains input sequences.
        target (torch.IntTensor): tensor contains target sequences.
        seq_lengths (torch.IntTensor): tensor contains input sequence lengths
        target_lengths (torch.IntTensor): tensor contains target sequence lengths
    """
    def seq_length_(p):
        return len(p[0])

    def target_length_(p):
        return len(p[1])

    seq_lengths = [len(s[0]) for s in batch]
    target_lengths = [len(s[1]) - 1 for s in batch]

    max_seq_sample = max(batch, key=seq_length_)[0]
    max_target_sample = max(batch, key=target_length_)[1]

    max_seq_size = len(max_seq_sample)
    max_target_size = len(max_target_sample)

    batch_size = len(batch)

    seqs = torch.zeros(batch_size, max_seq_size)

    targets = torch.zeros(batch_size, max_target_size).fill_(pad_token_id).to(torch.long)

    for x in range(batch_size):
        sample = batch[x]
        tensor = torch.tensor(sample[0])
        target = torch.tensor(sample[1])
        seq_length = len(tensor)

        seqs[x].narrow(0, 0, seq_length).copy_(tensor)
        targets[x].narrow(0, 0, len(target)).copy_(torch.LongTensor(target))

    seqs = seqs.long()
    targets = targets.long()
    seq_lengths = torch.LongTensor(seq_lengths)
    target_lengths = torch.LongTensor(target_lengths)

    return seqs, targets, seq_lengths, target_lengths


class TranslationDataset(Dataset):
    def __init__(self, sources, targets, source_tokenizer, target_tokenizer):
        super(TranslationDataset, self).__init__()
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.sos_id = target_tokenizer.PieceToId('<s>')
        self.eos_id = target_tokenizer.PieceToId('</s>')
        self.sources = list()
        self.targets = list()

        for source, target in zip(sources, targets):
            source = source_tokenizer.EncodeAsIds(source)
            self.sources.append(source)

            target = target_tokenizer.EncodeAsIds(target)
            target = [self.sos_id] + target + [self.eos_id]
            self.targets.append(target)

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        return self.sources[idx], self.targets[idx]
