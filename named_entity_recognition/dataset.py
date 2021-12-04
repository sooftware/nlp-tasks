# MIT License
# code by Soohwan Kim @sooftware

import wget
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

NER_TAGS = {
    "-": 0,
    "AFW": 1,
    "ANM": 2,
    "CVL": 3,
    "DAT": 4,
    "EVT": 5,
    "FLD": 6,
    "LOC": 7,
    "MAT": 8,
    "NUM": 9,
    "ORG": 10,
    "PER": 11,
    "PLT": 12,
    "TIM": 13,
    "TRM": 14,
}


def load_dataset(tokenizer, valid_ratio: float = 0.05):
    """
    Dataset Format:
         1       비토리오     PER_B
         2       양일         DAT_B
         3       만에         -
         4       영사관       ORG_B
         5       감호         CVL_B
         6       용퇴,        -
         7       항룡         -
         8       압력설       -
         9       의심만       -
         10      가율         -

         1       이           -
         ...
         ...
    """
    dataset = {
        "input_ids": list(),
        "labels": list(),
    }

    wget.download("https://raw.githubusercontent.com/naver/nlp-challenge/master/missions/ner/data/train/train_data")

    with open('train_data') as f:
        words = list()
        ner_tags = list()

        for line in tqdm(f.readlines()):
            line = line.strip()
            if line == '':
                tokens = list()
                labels = list()

                for word, ner_tag in zip(words, ner_tags):
                    word_tokens = tokenizer.encode(word)[1:-1]
                    word_labels = [NER_TAGS[ner_tag] for _ in range(len(word_tokens))]

                    tokens.extend(word_tokens)
                    labels.extend(word_labels)

                tokens = tokens[:tokenizer.model_max_length]
                labels = labels[:tokenizer.model_max_length]

                dataset["input_ids"].append(torch.LongTensor(tokens))
                dataset["labels"].append(torch.LongTensor(labels))

                words.clear()
                ner_tags.clear()
                continue

            _, word, ner_tag = line.split('\t')

            words.append(word)
            ner_tags.append(ner_tag.split('_')[0])

    num_total_datas = len(dataset["input_ids"])
    num_valid_datas = int(num_total_datas * valid_ratio)

    valid_input_ids = dataset["input_ids"][-num_valid_datas:]
    valid_labels = dataset["labels"][-num_valid_datas:]

    train_input_ids = dataset["input_ids"][:-num_valid_datas]
    train_labels = dataset["labels"][:-num_valid_datas]

    return {
        "train": {
            "input_ids": train_input_ids,
            "labels": train_labels
        },
        "valid": {
            "input_ids": valid_input_ids,
            "labels": valid_labels
        },
    }


class NaverNERDataset(Dataset):
    def __init__(self, data_source, tokenizer):
        super(NaverNERDataset, self).__init__()
        self.tokenizer = tokenizer
        self.data_source = data_source

    def __len__(self):
        return len(self.data_source["input_ids"])

    def __getitem__(self, idx):
        return self.data_source["input_ids"][idx], self.data_source["labels"][idx]


def collate_fn(batch, pad_token_id):
    def seq_length_(p):
        return len(p[0])

    lengths = [len(s[0]) for s in batch]

    max_seq_sample = max(batch, key=seq_length_)[0]
    max_seq_size = len(max_seq_sample)

    batch_size = len(batch)

    input_ids = torch.zeros(batch_size, max_seq_size).fill_(pad_token_id).long()
    labels = torch.zeros(batch_size, max_seq_size).fill_(pad_token_id).long()

    for idx in range(batch_size):
        sample = batch[idx]
        sample_input_ids = sample[0]
        sample_labels = sample[1]

        input_ids[idx].narrow(0, 0, len(sample_input_ids)).copy_(torch.LongTensor(sample_input_ids))
        labels[idx].narrow(0, 0, len(sample_labels)).copy_(torch.LongTensor(sample_labels))

    return input_ids, labels, lengths
