# MIT License
# code by Soohwan Kim @sooftware

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ElectraModel, AutoConfig


class PolyEncoderModel(nn.Module):
    def __init__(
            self,
            pretrain_model_name_or_path,
            num_poly_codes: int,
    ) -> None:
        super().__init__()
        self.model = ElectraModel.from_pretrained(pretrain_model_name_or_path)
        self.config = AutoConfig.from_pretrained(pretrain_model_name_or_path)
        self.num_poly_codes = num_poly_codes
        self.poly_code_ids = torch.arange(self.num_poly_codes).long().unsqueeze(0)
        self.poly_code_embeddings = nn.Embedding(num_poly_codes, self.config.hidden_size)
        torch.nn.init.normal_(self.poly_code_embeddings.weight, self.config.hidden_size ** -0.5)

    def _get_batch_size(self, tensor):
        return tensor.size(0)

    def _dot_attention(self, query, key, value):
        attn_weights = torch.matmul(query, key.transpose(2, 1))
        attn_weights = F.softmax(attn_weights, -1)
        return torch.matmul(attn_weights, value)

    def encode_context(self, input_ids, attention_masks):
        assert len(input_ids.size()) == 2
        assert len(attention_masks.size()) == 2

        batch_size = self._get_batch_size(input_ids)

        context_outputs = self.model(input_ids, attention_masks)[0]

        poly_code_ids = self.poly_code_ids.expand(batch_size, self.num_poly_codes).to(input_ids.device)
        poly_codes = self.poly_code_embeddings(poly_code_ids)

        return self._dot_attention(poly_codes, context_outputs, context_outputs)

    def encode_response(self, input_ids, attention_masks):
        batch_size, num_responses, seq_length = input_ids.size()

        input_ids = input_ids.view(-1, seq_length)
        attention_masks = attention_masks.view(-1, seq_length)

        candidate_embeddings = self.model(input_ids, attention_masks)[0][:, 0, :]
        return candidate_embeddings.view(batch_size, num_responses, -1)

    def _get_num_responses(self, responses_input_ids):
        return responses_input_ids.size(1)

    def get_scores(self, embeddings, candidate_embeddings, is_training: bool = True):
        batch_size, _, dim = candidate_embeddings.size()

        if is_training:
            candidate_embeddings = candidate_embeddings.transpose(0, 1)
            candidate_embeddings = candidate_embeddings.expand(batch_size, batch_size, dim)

            context_embeddings = self._dot_attention(candidate_embeddings, embeddings, embeddings)
            context_embeddings = context_embeddings.squeeze()

        else:
            context_embeddings = self._dot_attention(candidate_embeddings, embeddings, embeddings)

        scores = (context_embeddings * candidate_embeddings).sum(-1)
        return scores

    def forward(
            self,
            contexts,
            context_attention_masks,
            responses,
            response_attention_masks,
    ):
        num_responses = self._get_num_responses(responses)
        is_training = True if num_responses == 1 else False

        embeddings = self.encode_context(contexts, context_attention_masks)
        candidate_embeddings = self.encode_response(responses, response_attention_masks)

        return self.get_scores(embeddings, candidate_embeddings, is_training)
