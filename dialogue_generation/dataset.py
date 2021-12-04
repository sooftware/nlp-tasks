# MIT License
# code by Soohwan Kim @sooftware

import os
import json
import random
import re
import torch
from tqdm import tqdm
from torch.utils.data import Dataset


def collate_fn(batch, pad_token_id):
    def seq_length_(p):
        return len(p[0])

    max_seq_sample = max(batch, key=seq_length_)[0]
    max_seq_size = len(max_seq_sample)

    batch_size = len(batch)

    input_ids = torch.zeros(batch_size, max_seq_size).fill_(pad_token_id).long()
    attention_masks = torch.zeros(batch_size, max_seq_size).fill_(pad_token_id).long()
    labels = torch.zeros(batch_size, max_seq_size).fill_(pad_token_id).long()

    for idx in range(batch_size):
        sample = batch[idx]
        sample_input_ids = sample[0]
        sample_attention_masks = sample[1]
        sample_labels = sample[2]

        input_ids[idx].narrow(0, 0, len(sample_input_ids)).copy_(torch.LongTensor(sample_input_ids))
        attention_masks[idx].narrow(0, 0, len(sample_attention_masks)).copy_(torch.LongTensor(sample_attention_masks))
        labels[idx].narrow(0, 0, len(sample_labels)).copy_(torch.LongTensor(sample_labels))

    return input_ids, attention_masks, labels


def load_dataset(data_dir, sep_token: str, eos_token: str, val_ratio: float = 0.05):
    lines = list()

    if not os.path.exists(data_dir):
        raise FileNotFoundError

    for file in os.listdir(data_dir):
        if file.endswith('json'):
            with open(os.path.join(data_dir, file), encoding='utf-8-sig') as json_file:
                json_data = json.load(json_file, strict=False)

                for idx, episode in tqdm(enumerate(json_data['data']), desc=file):
                    last_id = None
                    line = str()

                    if episode['header']['dialogueInfo']['numberOfParticipants'] != 2:
                        continue

                    for utterance in episode['body']:
                        if last_id is None:
                            line += utterance['utterance']
                        elif last_id == utterance['participantID']:
                            line += ' ' + utterance['utterance']
                        else:
                            line += sep_token + utterance['utterance']

                        last_id = utterance['participantID']
                    line += eos_token
                    line = re.sub(' +', ' ', line)
                    lines.append(line)

    random.shuffle(lines)

    num_total_datas = len(lines)
    num_val_datas = int(num_total_datas * val_ratio)

    valid_datas = lines[-num_val_datas:]
    train_datas = lines[:-num_val_datas]
    return {
        "train": train_datas,
        "valid": valid_datas
    }


class DialogueDataset(Dataset):
    def __init__(self, datas, tokenizer):
        super(DialogueDataset, self).__init__()
        self.tokenizer = tokenizer
        self.input_ids = list()
        self.attention_masks = list()
        self.labels = list()

        for data in tqdm(datas, desc='Tokenizing..'):
            encoding_dict = self.tokenizer(data, truncation=True, max_length=tokenizer.model_max_length)
            self.input_ids.append(encoding_dict['input_ids'])
            self.attention_masks.append(encoding_dict['attention_mask'])
            self.labels.append(encoding_dict['input_ids'])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_masks[idx], self.labels[idx]
