# MIT License
# code by Soohwan Kim @sooftware

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast


class ResponseParser(object):
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, texts):
        input_ids_list, segment_ids_list, input_masks_list, contexts_masks_list = [], [], [], []
        for text in texts:
            tokenized_dict = self.tokenizer.encode_plus(text, max_length=self.max_length, pad_to_max_length=True)
            input_ids, input_masks = tokenized_dict['input_ids'], tokenized_dict['attention_mask']
            input_ids_list.append(input_ids)
            input_masks_list.append(input_masks)

        return input_ids_list, input_masks_list


class ContextParser(object):
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.cls_id = self.tokenizer.cls_token_id
        self.sep_id = self.tokenizer.sep_token_id
        self.sep_token = self.tokenizer.sep_token
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, texts):
        context = self.sep_token.join(texts)
        tokenized_dict = self.tokenizer.encode_plus(context, truncation=True, padding=True)
        input_ids, input_masks = tokenized_dict['input_ids'], tokenized_dict['attention_mask']
        input_ids = input_ids[-self.max_length:]
        input_ids[0] = self.cls_id
        input_masks = input_masks[-self.max_length:]
        input_ids += [self.pad_id] * (self.max_length - len(input_ids))
        input_masks += [0] * (self.max_length - len(input_masks))
        assert len(input_ids) == self.max_length
        assert len(input_masks) == self.max_length

        return input_ids, input_masks


class RetrievalDataset(Dataset):
    def __init__(
            self,
            file_path: str,
            tokenizer: PreTrainedTokenizerFast,
            context_max_length: int,
            response_max_length: int,
            sample_cnt: int = None,
    ) -> None:
        self.context_parser = ContextParser(tokenizer, context_max_length)
        self.response_parser = ResponseParser(tokenizer, response_max_length)
        self.data_source = list()
        negative_responses = list()

        with open(file_path, encoding='utf-8') as f:
            group = {
                'context': None,
                'responses': [],
                'labels': []
            }

            for line in f:
                split = line.strip('\n').split('\t')
                label, context, response = int(split[0]), split[1:-1], split[-1]

                if label == 1 and len(group['responses']) > 0:
                    self.data_source.append(group)
                    group = {
                        'context': None,
                        'responses': [],
                        'labels': []
                    }
                    if sample_cnt is not None and len(self.data_source) >= sample_cnt:
                        break
                else:
                    negative_responses.append(response)

                group['responses'].append(response)
                group['labels'].append(label)
                group['context'] = context

            if len(group['responses']) > 0:
                self.data_source.append(group)

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, index):
        group = self.data_source[index]
        context, responses, labels = group['context'], group['responses'], group['labels']

        transformed_context = self.context_parser(context)
        transformed_responses = self.response_parser(responses)

        return (
            transformed_context,
            transformed_responses,
            labels,
        )

    def collate_fn(self, batch):
        contexts, context_masks, responses, response_masks, labels = list(), list(), list(), list(), list()

        for sample in batch:
            (context, context_mask), (response, response_mask) = sample[:2]

            contexts.append(context)
            context_masks.append(context_mask)

            responses.append(response)
            response_masks.append(response_mask)

            labels.append(sample[-1])

        contexts = torch.LongTensor(contexts)
        context_masks = torch.LongTensor(context_masks)
        responses = torch.LongTensor(responses)
        response_masks = torch.LongTensor(response_masks)
        labels = torch.LongTensor(labels)

        return (
            contexts,
            context_masks,
            responses,
            response_masks,
            labels,
        )
