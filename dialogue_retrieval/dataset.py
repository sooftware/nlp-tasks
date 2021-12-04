# MIT License
# code by Soohwan Kim @sooftware

import os
import json
import random
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def load_dataset(data_dir, sep_token, valid_ratio: float = 0.05):
    episodes = list()
    contexts = list()
    responses = list()

    if not os.path.exists(data_dir):
        raise FileNotFoundError

    for file in os.listdir(data_dir):
        if file.endswith('json'):
            with open(os.path.join(data_dir, file), encoding='utf-8-sig') as json_file:
                json_data = json.load(json_file, strict=False)

                for idx, episode in enumerate(json_data['data']):
                    last_id = None

                    if episode['header']['dialogueInfo']['numberOfParticipants'] != 2:
                        continue

                    utterances = list()

                    for utterance in episode['body']:
                        if last_id is None:
                            utterances.append(utterance['utterance'])
                        elif last_id == utterance['participantID']:
                            utterances[-1] += utterance['utterance']
                        else:
                            utterances.append(utterance['utterance'])

                        last_id = utterance['participantID']

                    episodes.append(utterances)

    random.shuffle(episodes)
    # Logic of choosing about half of a series of conversations.
    for episode in tqdm(episodes, desc='Preprocess episode..'):
        num_utterances = len(episode)

        sample_num_turns = random.sample([i for i in range(2, num_utterances + 1)], round(num_utterances / 2))

        for s in sample_num_turns:
            input_ = sep_token.join(episode[:s - 1])
            target = episode[s - 1]
            contexts.append(input_)
            responses.append(target)

    num_total_datas = len(contexts)
    num_valid_datas = int(num_total_datas * valid_ratio)

    valid_contexts = contexts[-num_valid_datas:]
    valid_responses = responses[-num_valid_datas:]

    train_contexts = contexts[:-num_valid_datas]
    train_responses = responses[:-num_valid_datas]

    return {
        "train": {
            "contexts": train_contexts,
            "responses": train_responses
        },
        "valid": {
            "contexts": valid_contexts,
            "responses": valid_responses
        },
    }


class DialogueRetrievalDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        super(DialogueRetrievalDataset, self).__init__()
        self.tokenizer = tokenizer
        self.contexts = list()
        self.context_attention_masks = list()
        self.responses = list()
        self.response_attention_masks = list()

        for context, response in tqdm(zip(dataset["contexts"], dataset["responses"]), desc='Tokenizing..'):
            context = self.tokenizer(context, truncation=True, max_length=tokenizer.model_max_length)
            response = self.tokenizer(response, truncation=True, max_length=tokenizer.model_max_length)

            self.contexts.append(context['input_ids'])
            self.context_attention_masks.append(context['attention_mask'])

            self.responses.append(response['input_ids'])
            self.response_attention_masks.append(response['attention_mask'])

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        return (
            self.contexts[idx],
            self.context_attention_masks[idx],
            self.responses[idx],
            self.response_attention_masks[idx],
        )


def collate_fn(batch, pad_token_id, is_training: bool = True):
    def seq_length_(p):
        return len(p[0])

    def target_length_(p):
        return len(p[2])

    max_seq_sample = max(batch, key=seq_length_)[0]
    max_target_sample = max(batch, key=target_length_)[2]

    max_seq_size = len(max_seq_sample)
    max_target_size = len(max_target_sample)

    batch_size = len(batch)

    contexts = torch.zeros(batch_size, max_seq_size).fill_(pad_token_id).long()
    context_attention_masks = torch.zeros(batch_size, max_seq_size).fill_(pad_token_id).long()
    responses = torch.zeros(batch_size, max_target_size).fill_(pad_token_id).long()
    response_attention_masks = torch.zeros(batch_size, max_target_size).fill_(pad_token_id).long()

    for x in range(batch_size):
        sample = batch[x]
        sample_context = torch.tensor(sample[0])
        sample_context_attention_mask = torch.tensor(sample[1])
        sample_response = torch.tensor(sample[2])
        sample_response_attention_mask = torch.tensor(sample[3])

        context_length = len(sample_context)
        response_length = len(sample_response)

        contexts[x].narrow(0, 0, context_length).copy_(sample_context)
        context_attention_masks[x].narrow(0, 0, context_length).copy_(sample_context_attention_mask)
        responses[x].narrow(0, 0, response_length).copy_(sample_response)
        response_attention_masks[x].narrow(0, 0, response_length).copy_(sample_response_attention_mask)

    if not is_training:
        # Repeat responses for batch_size times to simulate test phase.
        # Every context is paired with batch_size responses
        #
        # B x T => B x B x T (B: batch size, T: sequence length)
        #
        # Example::
        #
        # >>> responses = torch.arange(36).view(3, 12)
        # tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        #         [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        #         [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]])
        #
        # >>> responses = responses.transpose(0, 1).expand(batch_size, batch_size, dim)
        # tensor([[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        #          [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        #          [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]],
        #
        #         [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        #          [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        #          [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]],
        #
        #         [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        #          [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        #          [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]]])
        #
        labels = responses
        responses = responses.unsqueeze(1).transpose(0, 1)
        responses = responses.expand(batch_size, batch_size, max_target_size)
        response_attention_masks = response_attention_masks.unsqueeze(1).transpose(0, 1)
        response_attention_masks = response_attention_masks.expand(batch_size, batch_size, max_target_size)
        return (
            contexts.long(),
            context_attention_masks.long(),
            responses.long(),
            response_attention_masks.long(),
            labels.long(),
        )

    return (
        contexts.long(),
        context_attention_masks.long(),
        responses.unsqueeze(1).long(),
        response_attention_masks.unsqueeze(1).long(),
    )
