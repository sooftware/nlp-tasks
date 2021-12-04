# MIT License
# code by Soohwan Kim @sooftware

import torch
from torch.utils.data import Dataset


def collate_fn(batch, pad_token_id):
    def seq_length_(p):
        return len(p[0])

    def target_length_(p):
        return len(p[2])

    max_seq_sample = max(batch, key=seq_length_)[0]
    max_target_sample = max(batch, key=target_length_)[2]

    max_seq_size = len(max_seq_sample)
    max_target_size = len(max_target_sample)

    batch_size = len(batch)

    input_ids = torch.zeros(batch_size, max_seq_size).fill_(pad_token_id).long()
    attention_masks = torch.zeros(batch_size, max_seq_size).fill_(pad_token_id).long()
    targets = torch.zeros(batch_size, max_target_size).fill_(pad_token_id).long()

    for idx in range(batch_size):
        sample = batch[idx]
        sample_input_ids = sample[0]
        sample_attention_masks = sample[1]
        sample_labels = sample[2]

        input_ids[idx].narrow(0, 0, len(sample_input_ids)).copy_(torch.LongTensor(sample_input_ids))
        attention_masks[idx].narrow(0, 0, len(sample_attention_masks)).copy_(torch.LongTensor(sample_attention_masks))
        targets[idx].narrow(0, 0, len(sample_labels)).copy_(torch.LongTensor(sample_labels))

    return input_ids, attention_masks, targets


class XSumDataset(Dataset):
    def __init__(self, data_sources, tokenizer):
        super(XSumDataset, self).__init__()
        self.inputs = list()
        self.attention_masks = list()
        self.targets = list()

        for data in data_sources:
            input_encoding_dict = tokenizer(data["document"], max_length=1024, truncation=True)
            target_encoding_dict = tokenizer(data["summary"], max_length=1024, truncation=True)

            self.inputs.append(input_encoding_dict["input_ids"])
            self.attention_masks.append(input_encoding_dict["attention_mask"])
            self.targets.append(target_encoding_dict["input_ids"])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.attention_masks[idx], self.targets[idx]
