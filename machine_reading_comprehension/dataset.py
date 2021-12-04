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
    start_positions = torch.zeros(batch_size, 1).long()
    end_positions = torch.zeros(batch_size, 1).long()

    for idx in range(batch_size):
        sample = batch[idx]
        sample_input_ids = sample[0]
        sample_attention_masks = sample[1]
        sample_start_positions = sample[2]
        sample_end_positions = sample[2]

        input_ids[idx].narrow(0, 0, len(sample_input_ids)).copy_(torch.LongTensor(sample_input_ids))
        attention_masks[idx].narrow(0, 0, len(sample_attention_masks)).copy_(torch.LongTensor(sample_attention_masks))
        start_positions[idx].narrow(0, 0, len(sample_start_positions)).copy_(torch.LongTensor(sample_start_positions))
        end_positions[idx].narrow(0, 0, len(sample_end_positions)).copy_(torch.LongTensor(sample_end_positions))

    return input_ids, attention_masks, start_positions, end_positions


class SquadDataset(Dataset):
    def __init__(self, datas, tokenizer):
        super(SquadDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = tokenizer.model_max_length

        questions, contexts, answers = self._read_squad(datas)
        answers = self._add_end_index(answers, contexts)
        encoding_dicts = self._tokenize(contexts, questions)
        self.encoding_dicts = self._add_token_positions(encoding_dicts, answers)

    def _read_squad(self, datas):
        questions = list()
        contexts = list()
        answers = list()

        for data in datas:
            questions.append(data["question"])
            contexts.append(data["context"])
            answers.append(data["answers"])

        return questions, contexts, answers

    def _add_end_index(self, answers, contexts):
        for answer, context in zip(answers, contexts):
            target_text = answer['text']
            start_index = answer['answer_start']
            end_index = start_index + len(target_text)

            # sometimes squad answers are off by a character or two – fix this
            if context[start_index:end_index] == target_text:
                answer['answer_end'] = end_index
            elif context[start_index - 1:end_index - 1] == target_text:
                answer['answer_start'] = start_index - 1
                answer['answer_end'] = end_index - 1
            elif context[start_index - 2:end_index - 2] == target_text:
                answer['answer_start'] = start_index - 2
                answer['answer_end'] = end_index - 2

        return answers

    def _tokenize(self, contexts, questions):
        encoding_dicts = list()
        for context, question in zip(contexts, questions):
            encoding_dicts.append(self.tokenizer(question, context, truncation=True, max_length=self.max_length))
        return encoding_dicts

    def _add_token_positions(self, encoding_dicts, answers):
        start_positions = []
        end_positions = []
        for i in range(len(answers)):
            start_positions.append(encoding_dicts.char_to_token(i, answers[i]['answer_start']))
            end_positions.append(encoding_dicts.char_to_token(i, answers[i]['answer_end'] - 1))

            # if start position is None, the answer passage has been truncated
            if start_positions[-1] is None:
                start_positions[-1] = self.tokenizer.model_max_length
            if end_positions[-1] is None:
                end_positions[-1] = self.tokenizer.model_max_length

        encoding_dicts.update({'start_positions': start_positions, 'end_positions': end_positions})
        return encoding_dicts

    def __len__(self):
        return len(self.encoding_dicts)

    def __getitem__(self, idx):
        return (
            self.encoding_dicts[idx]["input_ids"],
            self.encoding_dicts[idx]["attention_mask"],
            self.encoding_dicts[idx]["start_positions"],
            self.encoding_dicts[idx]["end_positions"],
        )
