import torch
from torch.utils.data import Dataset


def collate_fn(batch, pad_token_id):
    def seq_length_(p):
        return len(p[0])

    max_context_sample = max(batch, key=seq_length_)[0]
    max_context_size = len(max_context_sample)

    batch_size = len(batch)

    contexts = torch.zeros(batch_size, max_context_size).fill_(pad_token_id).long()
    context_masks = torch.zeros(batch_size, max_context_size).fill_(pad_token_id).long()
    responses = list()
    response_masks = list()
    labels = list()

    for idx in range(batch_size):
        sample = batch[idx]
        context = sample[0]
        context_mask = sample[1]
        response = sample[2]
        response_mask = sample[3]
        label = sample[4]

        contexts[idx].narrow(0, 0, len(context)).copy_(torch.LongTensor(context))
        context_masks[idx].narrow(0, 0, len(context_mask)).copy_(torch.LongTensor(context_mask))
        responses.append(response)
        response_masks.append(response_mask)
        labels.append(label)

    responses = torch.tensor(responses)
    response_masks = torch.tensor(response_masks)
    labels = torch.tensor(labels).long()

    return (
        contexts,
        context_masks,
        responses,
        response_masks,
        labels,
    )


class ResponseTransformer(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, texts):
        input_ids_list, input_masks_list = list(), list()

        for text in texts:
            tokenized_dict = self.tokenizer(text, max_length=128, padding='max_length')
            input_ids, input_masks = tokenized_dict['input_ids'], tokenized_dict['attention_mask']
            input_ids_list.append(input_ids)
            input_masks_list.append(input_masks)

        return input_ids_list, input_masks_list


class ContextTransformer(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.sep_token = self.tokenizer.sep_token

    def __call__(self, texts):
        # another option is to use [SEP], but here we follow the discussion at:
        # https://github.com/facebookresearch/ParlAI/issues/2306#issuecomment-599180186
        context = self.sep_token.join(texts)
        tokenized_dict = self.tokenizer(context)
        input_ids, input_masks = tokenized_dict['input_ids'], tokenized_dict['attention_mask']
        return input_ids, input_masks


class DialogueRetrievalDataset(Dataset):
    def __init__(self, file_path, tokenizer, sample_cnt=None):
        self.context_transform = ContextTransformer(tokenizer)
        self.response_transform = ResponseTransformer(tokenizer)
        self.data_source = list()
        neg_responses = list()

        with open(file_path, encoding='utf-8') as f:
            group = {
                'context': None,
                'responses': list(),
                'labels': list(),
            }
            for line in f:
                split = line.strip('\n').split('\t')
                label, context, response = int(split[0]), split[1:-1], split[-1]
                if int(label) == 1 and len(group['responses']) > 0:
                    self.data_source.append(group)
                    group = {
                        'context': None,
                        'responses': list(),
                        'labels': list(),
                    }
                    if sample_cnt is not None and len(self.data_source) >= sample_cnt:
                        break
                else:
                    neg_responses.append(response)

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
        context, context_mask = self.context_transform(context)
        responses, response_masks = self.response_transform(responses)
        return context, context_mask, responses, response_masks, labels
