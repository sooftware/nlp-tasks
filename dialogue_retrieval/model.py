# MIT License
# code by Soohwan Kim @sooftware

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class PolyEncoderModel(nn.Module):
    def __init__(self, pretrain_model_name_or_path, num_poly_codes: int = 64):
        super().__init__()
        self.model = AutoModel.from_pretrained(pretrain_model_name_or_path)
        self.num_poly_codes = num_poly_codes
        self.poly_code_embeddings = nn.Embedding(self.num_poly_codes, self.model.config.hidden_size)
        torch.nn.init.normal_(self.poly_code_embeddings.weight, self.model.config.hidden_size ** -0.5)

    def dot_attention(self, query, key, value):
        attn = torch.matmul(query, key.transpose(2, 1))
        attn = F.softmax(attn, -1)
        return torch.matmul(attn, value)

    def encode_context(self, input_ids, attention_masks):
        batch_size = input_ids.size(0)

        context_outputs = self.model(input_ids, attention_masks)[0]
        poly_code_ids = torch.arange(self.num_poly_codes, dtype=torch.long).to(input_ids.device)
        poly_code_ids = poly_code_ids.unsqueeze(0).expand(batch_size, self.num_poly_codes)
        poly_codes = self.poly_code_embeddings(poly_code_ids)
        return self.dot_attention(poly_codes, context_outputs, context_outputs)

    def encode_response(self, input_ids, attention_masks):
        batch_size, num_responses, seq_length = input_ids.shape

        responses_input_ids = input_ids.view(-1, seq_length)
        responses_input_masks = attention_masks.view(-1, seq_length)

        candidate_embeddings = self.model(responses_input_ids, responses_input_masks)[0][:, 0, :]
        return candidate_embeddings.view(batch_size, num_responses, -1)

    def get_score(self, contexts, responses):
        contexts = self.dot_attention(responses, contexts, contexts)
        score = (contexts * responses).sum(-1)
        return score

    def get_loss(self, contexts, responses):
        batch_size = contexts.size(0)

        responses = responses.permute(1, 0, 2)
        responses = responses.expand(batch_size, batch_size, responses.size(2))

        contexts = self.dot_attention(responses, contexts, contexts).squeeze()
        dot_product = (contexts * responses).sum(-1)

        mask = torch.eye(batch_size).to(contexts.device)

        loss = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss.sum(dim=1)).mean()

        return loss

    def forward(
            self,
            context_input_ids,
            context_input_masks,
            responses_input_ids,
            responses_input_masks,
            labels=None,
    ):
        is_training = True if labels is not None else False

        if is_training:
            responses_input_ids = responses_input_ids[:, 0, :].unsqueeze(1)
            responses_input_masks = responses_input_masks[:, 0, :].unsqueeze(1)

        contexts = self.encode_context(context_input_ids, context_input_masks)
        responses = self.encode_response(responses_input_ids, responses_input_masks)

        if is_training:
            output = self.get_loss(contexts, responses)
        else:
            output = self.get_score(contexts, responses)

        return output
