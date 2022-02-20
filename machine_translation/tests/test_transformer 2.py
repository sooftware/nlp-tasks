import unittest
import torch

from machine_translation.model import Transformer


class TestTransformer(unittest.TestCase):
    def test_transformer_inference(self):
        BATCH_SIZE = 3
        SEQ_LENGTH = 10
        MAX_LENGTH = 32
        TARGET_VOCAB_SIZE = 320

        model = Transformer(80, 320, d_model=64, num_encoder_layers=2, num_decoder_layers=2, max_length=MAX_LENGTH)
        inputs = torch.zeros(BATCH_SIZE, SEQ_LENGTH).long()
        input_lengths = torch.LongTensor([10, 9, 8])

        outputs = model.forward(inputs, input_lengths)

        assert len(outputs.size()) == 3
        assert outputs.size(0) == 3
        assert outputs.size(1) == MAX_LENGTH - 1
        assert outputs.size(2) == TARGET_VOCAB_SIZE
        print("test_transformer_inference Success ==")


if __name__ == '__main__':
    unittest.main()
