# MIT License
# code by Soohwan Kim @sooftware

import torch
from torch.utils.data import Dataset


def collate_fn(batch, pad_token_id):
    def seq_length_(p):
        return len(p[0])

    labels = list()
    max_seq_sample = max(batch, key=seq_length_)[0]
    max_seq_size = len(max_seq_sample)

    batch_size = len(batch)

    input_ids = torch.zeros(batch_size, max_seq_size).fill_(pad_token_id).long()
    attention_masks = torch.zeros(batch_size, max_seq_size).fill_(pad_token_id).long()

    for idx in range(batch_size):
        sample = batch[idx]
        sample_input_ids = sample[0]
        sample_attention_masks = sample[1]
        sample_labels = sample[2]

        input_ids[idx].narrow(0, 0, len(sample_input_ids)).copy_(torch.LongTensor(sample_input_ids))
        attention_masks[idx].narrow(0, 0, len(sample_attention_masks)).copy_(torch.LongTensor(sample_attention_masks))
        labels.append(sample_labels)

    labels = torch.FloatTensor(labels)

    return input_ids, attention_masks, labels


class GlueSTSDataset(Dataset):
    def __init__(self, data_sources, tokenizer):
        super(GlueSTSDataset, self).__init__()
        self.inputs = list()
        self.attention_masks = list()
        self.targets = list()

        for data in data_sources:
            encoding_dict = tokenizer(data["sentence1"], data["sentence2"])

            self.inputs.append(encoding_dict["input_ids"])
            self.attention_masks.append(encoding_dict["attention_mask"])
            self.targets.append(data["label"])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.attention_masks[idx], self.targets[idx]
