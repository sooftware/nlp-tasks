# MIT License
# code by Soohwan Kim @sooftware

import pandas as pd
import torch
import wget
import random
from torch.utils.data import Dataset


def load_dataset(valid_ratio: float = 0.05):
    try:
        wget.download('https://raw.githubusercontent.com/tunib-ai/KMWP/main/data/train.csv')
    except:
        raise ValueError("URL not valid")

    df = pd.read_csv('train.csv')

    problems = df['problem']
    codes = df['code']
    answers = df['answer']

    zipped = list(zip(problems, codes, answers))
    random.shuffle(zipped)
    problems, codes, answers = zip(*zipped)

    num_valids = int(len(problems) * valid_ratio)

    train_problems = problems[:-num_valids]
    train_codes = codes[:-num_valids]
    train_answers = answers[:-num_valids]

    valid_problems = problems[-num_valids:]
    valid_codes = codes[-num_valids:]
    valid_answers = answers[-num_valids:]

    return {
        "train": {
            "examples": train_problems,
            "codes": train_codes,
            "answers": train_answers,
        },
        "valid": {
            "examples": valid_problems,
            "codes": valid_codes,
            "answers": valid_answers,
        },
    }


def collate_fn(batch, pad_token_id):
    def seq_length_(p):
        return len(p[0])

    max_seq_sample = max(batch, key=seq_length_)[0]
    max_seq_size = len(max_seq_sample)

    batch_size = len(batch)

    input_ids = torch.zeros(batch_size, max_seq_size).fill_(pad_token_id).long()
    attention_masks = torch.zeros(batch_size, max_seq_size).fill_(pad_token_id).long()
    labels = torch.zeros(batch_size, max_seq_size).long()
    answers = list()

    for idx in range(batch_size):
        sample = batch[idx]
        sample_input_ids = sample[0]
        sample_attention_masks = sample[1]
        sample_labels = sample[2]
        sample_answer = sample[3]

        input_ids[idx].narrow(0, 0, len(sample_input_ids)).copy_(torch.LongTensor(sample_input_ids))
        attention_masks[idx].narrow(0, 0, len(sample_attention_masks)).copy_(torch.LongTensor(sample_attention_masks))
        labels[idx].narrow(0, 0, len(sample_labels)).copy_(torch.LongTensor(sample_labels))
        answers.append(sample_answer)

    return input_ids, attention_masks, labels, answers


class MathWordProblemDataset(Dataset):
    def __init__(self, data_sources, tokenizer, sep_token: str = "<unused0>", is_training: bool = True):
        super(MathWordProblemDataset, self).__init__()
        self.input_ids = list()
        self.attention_masks = list()
        self.labels = list()
        self.answers = list()
        self.is_training = is_training
        self.eos_token = tokenizer.eos_token

        if is_training:
            for example, code, answer in zip(data_sources["examples"], data_sources["codes"], data_sources["answers"]):
                inputs = example + sep_token + code + self.eos_token

                encoding_dict = tokenizer(inputs, truncation=True, padding=True, max_length=1024)
                self.input_ids.append(encoding_dict["input_ids"])
                self.attention_masks.append(encoding_dict["attention_mask"])
                self.labels.append(encoding_dict["input_ids"])
                self.answers.append(answer)
        else:
            for example, answer in zip(data_sources["examples"], data_sources["answers"]):
                inputs = example + sep_token
                encoding_dict = tokenizer(inputs, truncation=True, padding=True, max_length=1024)
                self.input_ids.append(encoding_dict["input_ids"])
                self.attention_masks.append(encoding_dict["attention_mask"])
                self.labels.append(encoding_dict["input_ids"])
                self.answers.append(answer)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_masks[idx], self.labels[idx], self.answers[idx]
