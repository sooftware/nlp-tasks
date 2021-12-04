# MIT License
# code by Soohwan Kim @sooftware

import torch
from torch.utils.data import Dataset


def collate_fn(batch, pad_token_id):
    def seq_length_(p):
        return len(p[0])

    max_seq_sample = max(batch, key=seq_length_)[0]
    max_seq_size = len(max_seq_sample)

    batch_size = len(batch)

    input_ids = torch.zeros(batch_size, max_seq_size).fill_(pad_token_id).long()
    attention_masks = torch.zeros(batch_size, max_seq_size).fill_(pad_token_id).long()
    labels = torch.zeros(batch_size, max_seq_size).long()

    for idx in range(batch_size):
        sample = batch[idx]
        sample_input_ids = sample[0]
        sample_attention_masks = sample[1]
        sample_labels = sample[2]

        input_ids[idx].narrow(0, 0, len(sample_input_ids)).copy_(torch.LongTensor(sample_input_ids))
        attention_masks[idx].narrow(0, 0, len(sample_attention_masks)).copy_(torch.LongTensor(sample_attention_masks))
        labels[idx].narrow(0, 0, len(sample_labels)).copy_(torch.LongTensor(sample_labels))

    return input_ids, attention_masks, labels


class LanguageModelDataset(Dataset):
    def __init__(self, texts, tokenizer):
        super(LanguageModelDataset, self).__init__()
        self.tokenizer = tokenizer
        self.texts = list()
        self.attention_masks = list()
        self.labels = list()

        for text in texts:
            encoding_dict = self.tokenizer(text["text"], max_length=tokenizer.model_max_length, truncation=True)
            self.texts.append(encoding_dict['input_ids'])
            self.attention_masks.append(encoding_dict['attention_mask'])
            self.labels.append(encoding_dict['input_ids'])

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.attention_masks[idx], self.labels[idx]
