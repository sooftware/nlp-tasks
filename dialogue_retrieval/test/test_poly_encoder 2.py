import unittest
import torch
from transformers import AutoTokenizer, AutoConfig

from ..model import PolyEncoder


class TestPolyEncoder(unittest.TestCase):
    def test_forward_train(self):
        tokenizer = AutoTokenizer.from_pretrained("tunib/electra-ko-en-small")
        model = PolyEncoder(pretrain_model_name_or_path='tunib/electra-ko-en-small', num_poly_codes=16)

        contexts = tokenizer("poly encoder test", return_tensors="pt")
        responses = tokenizer("response test", return_tensors="pt")

        context_input_ids = contexts["input_ids"].repeat([3, 1])
        context_mask = contexts["attention_mask"].repeat([3, 1])

        response_input_ids = responses["input_ids"].repeat([3, 1]).unsqueeze(1)
        response_mask = responses["attention_mask"].repeat([3, 1]).unsqueeze(1)

        scores = model(context_input_ids, context_mask, response_input_ids, response_mask)

        assert len(scores.size()) == 2, "Shape Error"

    def test_forward_inference(self):
        tokenizer = AutoTokenizer.from_pretrained("tunib/electra-ko-en-small")
        model = PolyEncoder(pretrain_model_name_or_path='tunib/electra-ko-en-small', num_poly_codes=16)

        contexts = tokenizer("poly encoder test", return_tensors="pt")
        responses = tokenizer("response test", return_tensors="pt")

        context_input_ids = contexts["input_ids"].repeat([3, 1])
        context_mask = contexts["attention_mask"].repeat([3, 1])

        response_input_ids = responses["input_ids"].unsqueeze(1).repeat([3, 5, 1])
        response_mask = responses["attention_mask"].unsqueeze(1).repeat([3, 5, 1])

        scores = model(context_input_ids, context_mask, response_input_ids, response_mask)

        assert len(scores.size()) == 2, "Shape Error"

    def test_encode_context(self):
        B, T, N = 3, 10, 16

        config = AutoConfig.from_pretrained('tunib/electra-ko-en-small')
        D = config.hidden_size

        model = PolyEncoder(pretrain_model_name_or_path='tunib/electra-ko-en-small', num_poly_codes=N)

        contexts = torch.ones(B, T).long()
        attention_masks = torch.zeros(B, T).long()

        encode_context_outputs = model.encode_context(contexts, attention_masks)
        assert list(encode_context_outputs.size()) == [B, N, D]

    def test_encode_response(self):
        B, R, T = 3, 1, 10

        config = AutoConfig.from_pretrained('tunib/electra-ko-en-small')
        D = config.hidden_size

        model = PolyEncoder(pretrain_model_name_or_path='tunib/electra-ko-en-small', num_poly_codes=N)

        responses = torch.ones(B, R, T).long()
        attention_masks = torch.zeros(B, R, T).long()

        encode_response_outputs = model.encode_response(responses, attention_masks)
        assert list(encode_response_outputs.size()) == [B, R, D]


if __name__ == '__main__':
    unittest.main()
