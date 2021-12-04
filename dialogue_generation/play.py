# MIT License
# code by Soohwan Kim @sooftware

import argparse
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel
from typing import List, Tuple


class Chatbot:
    def __init__(
        self,
        pretrain_model_name_or_path,
        tokenizer_name,
        sep_token,
        device,
        num_beams: int = 3,
        length_penalty: float = 1.2,
        no_repeat_ngram_size: int = 4,
        repetition_penalty: float = 1.2,
    ):
        super(Chatbot, self).__init__()
        print("Loading model ...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = GPT2LMHeadModel.from_pretrained(pretrain_model_name_or_path).to(device)

        self.sep_token = sep_token
        self.sep_token_id = self.tokenizer.vocab[sep_token]
        self.num_beams = num_beams
        self.max_length = self.tokenizer.model_max_length
        self.length_penalty = length_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.repetition_penalty = repetition_penalty
        self.device = device

    def __call__(
        self,
        user_response: str,
        histories: List[str],
    ) -> Tuple[str, bool]:
        is_finish = False

        histories.append(user_response)

        input_string = self.sep_token.join(histories)
        input_string += self.sep_token

        inputs = self.tokenizer(input_string, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=self.max_length,
            num_beams=self.num_beams,
            length_penalty=self.length_penalty,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            repetition_penalty=self.repetition_penalty,
            eos_token_id=self.sep_token_id,
            early_stopping=True,
        )
        bot_response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        bot_response = bot_response.split(self.sep_token)[-2]

        bot_response.replace("<unk>", "")
        histories.append(bot_response)

        if "</s>" in bot_response:
            is_finish = True
            histories.clear()

            eos_idx = bot_response.index("</s>")
            bot_response = bot_response[:eos_idx]

        elif "<pad>" in bot_response:
            is_finish = True
            histories.clear()

            eos_idx = bot_response.index("<pad>")
            bot_response = bot_response[:eos_idx]

        return bot_response, is_finish


def _get_parser():
    parser = argparse.ArgumentParser(description='NLP Tasks - Dialogue Retrieval')
    # Basic arguments
    parser.add_argument('--pretrain_model_name_or_path', type=str, required=True)
    parser.add_argument('--pretrain_tokenizer_name', type=str, required=True)
    parser.add_argument('--sep_token', type=str, required=True)
    return parser


if __name__ == "__main__":
    histories = list()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = _get_parser()
    args = parser.parse_args()

    chatbot = Chatbot(args.pretrain_model_name_or_path,
                      args.pretrain_tokenizer_name,
                      args.sep_token,
                      device)

    while True:
        user_response = input("YOUR TURN: ")
        bot_response, is_finish = chatbot(user_response, histories)
        print(f"BOT: {bot_response}")
