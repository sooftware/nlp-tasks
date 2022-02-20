# MIT License
# code by Soohwan Kim @sooftware

import torch
from torch.utils.data import Dataset
from tqdm import trange


def load_dataset(data_dir):
    train_texts = list()
    valid_texts = list()

    with open(f'{data_dir}/train.txt', encoding='utf-8') as f:
        for line in f.readlines():
            train_texts.append(line)

    with open(f'{data_dir}/valid.txt', encoding='utf-8') as f:
        for line in f.readlines():
            valid_texts.append(line)

    return {
        "train": train_texts,
        "valid": valid_texts,
    }


def collate_fn(batch, pad_token_id):
    def seq_length_(p):
        return len(p[0])

    def target_length_(p):
        return len(p[2])

    max_seq_sample = max(batch, key=seq_length_)[0]
    max_seq_size = len(max_seq_sample)
    max_target_sample = max(batch, key=target_length_)[2]
    max_target_size = len(max_target_sample)

    batch_size = len(batch)

    input_ids = torch.zeros(batch_size, max_seq_size).fill_(pad_token_id).long()
    encoder_attention_masks = torch.zeros(batch_size, max_seq_size).fill_(pad_token_id).long()
    labels = torch.zeros(batch_size, max_target_size).long()
    decoder_attention_masks = torch.zeros(batch_size, max_target_size).fill_(pad_token_id).long()

    for idx in range(batch_size):
        sample = batch[idx]
        sample_input_ids = sample[0]
        sample_encoder_attention_masks = sample[1]
        sample_labels = sample[2]
        sample_decoder_attention_masks = sample[3]

        input_ids[idx].narrow(0, 0, len(sample_input_ids)).copy_(torch.LongTensor(sample_input_ids))
        encoder_attention_masks[idx].narrow(0, 0, len(sample_encoder_attention_masks)).copy_(torch.LongTensor(
            sample_encoder_attention_masks))
        labels[idx].narrow(0, 0, len(sample_labels)).copy_(torch.LongTensor(sample_labels))
        decoder_attention_masks[idx].narrow(0, 0, len(sample_decoder_attention_masks)).copy_(torch.LongTensor(
            sample_decoder_attention_masks))

    return input_ids, encoder_attention_masks, labels, decoder_attention_masks


class KnowledgePoweredConversationDataset(Dataset):
    def __init__(self, split_dataset, tokenizer):
        super(KnowledgePoweredConversationDataset, self).__init__()
        self.tokenizer = tokenizer
        self.input_ids = list()
        self.encodier_attention_masks = list()
        self.lm_labels = list()
        self.decoder_attention_masks = list()

        histories = list(split_dataset['history'])
        gold_knowledges = list(split_dataset['gold_knowledge'])
        targets = list(split_dataset['target'])

        for idx in trange(len(split_dataset)):
            history = histories[idx].replace('<start_conversation>', '').replace('</s>', '<sep>')
            gold_knowledge = gold_knowledges[idx]

            input_text = f"<knowledge>{gold_knowledge}{history}"
            target_text = targets[idx]

            encoder_dict = self.tokenizer(input_text, padding=True, truncation=True)
            decoder_dict = self.tokenizer(target_text, padding=True, truncation=True)

            self.input_ids.append(encoder_dict['input_ids'])
            self.encodier_attention_masks.append(encoder_dict['attention_mask'])

            self.lm_labels.append(decoder_dict['input_ids'])
            self.decoder_attention_masks.append(decoder_dict['attention_mask'])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return(
            self.input_ids[idx], self.encodier_attention_masks[idx], self.lm_labels[idx], self.decoder_attention_masks[idx]
        )