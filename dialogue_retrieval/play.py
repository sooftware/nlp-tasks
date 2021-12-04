# MIT License
# code by Soohwan Kim @sooftware

import argparse
import pickle
import torch
import torch.nn as nn
import numpy as np
import random
from transformers import AutoTokenizer
from typing import List


class Chatbot:
    def __init__(
        self,
        checkpoint_path: str,
        tokenizer_name: str,
        candidate_vectors_path: str,
        candidates_path: str,
        batch_size: int,
        device: torch.device,
    ):
        super(Chatbot, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        model = torch.load(checkpoint_path)
        if isinstance(model, nn.DataParallel):
            self.model = model.module.to(device)
        else:
            self.model = model.to(device)

        self.model.eval()
        self.sep_token = self.tokenizer.sep_token
        self.sep_token_id = self.tokenizer.vocab[self.sep_token]
        self.device = device
        self.batch_size = batch_size
        candidate_dict = self.prepare_candidate_vectors(
            candidate_vectors_path=candidate_vectors_path,
            candidates_path=candidates_path,
            batch_size=batch_size,
        )
        self.vectors = candidate_dict["vector"]
        self.texts = candidate_dict["text"]

    def __call__(
            self,
            user_response: str,
            histories: List[str],
    ) -> str:
        scores = list()

        if len(histories) > 10 and random.random() <= 0.15:
            histories.clear()

        histories.append(user_response)

        input_string = self.sep_token.join(histories)
        inputs = self.tokenizer(input_string, return_tensors="pt").to(self.device)

        with torch.no_grad():
            embeddings = self.model.encode_context(inputs["input_ids"], inputs["attention_mask"])

            for idx in range(0, len(self.vectors), self.batch_size):
                vectors = torch.tensor(self.vectors[idx:idx + self.batch_size]).to(self.device)
                score = self.model.get_scores(embeddings, vectors, is_training=False)
                score = score.squeeze().tolist()
                scores.extend(score)

        bot_response = self.texts[np.argmax(scores)]
        histories.append(bot_response)

        return bot_response

    def prepare_candidate_vectors(
            self,
            candidate_vectors_path: str = None,
            candidates_path: str = None,
            batch_size: int = 128,
    ):
        assert candidate_vectors_path is not None or candidates_path is not None

        if candidate_vectors_path is None:
            candidate_dict = {
                "text": list(),
                "vector": list(),
            }

            with open(candidates_path, encoding='utf-8-sig') as fr:
                candidates = fr.readlines()
                with torch.no_grad():
                    for idx in range(0, len(candidates), batch_size):
                        batch = candidates[idx:idx + batch_size]
                        inputs = self.tokenizer(
                            batch,
                            return_tensors="pt",
                            max_length=self.tokenizer.model_max_length,
                            truncation=True,
                            padding=True,
                        ).to(self.device)

                        candidate_vecs = self.model.encode_response(
                            input_ids=inputs["input_ids"].unsqueeze(1),
                            attention_masks=inputs["attention_mask"].unsqueeze(1),
                        )

                        for batch_idx in range(batch_size):
                            if batch_idx >= len(candidate_vecs):
                                break
                            vector = candidate_vecs[batch_idx].tolist()
                            candidate_dict["vector"].append(vector)
                            candidate_dict["text"].append(candidates[idx + batch_idx])

                        with open(f"{candidate_vectors_path}", 'wb') as fw:
                            pickle.dump(candidate_dict, fw)

        else:
            with open(candidate_vectors_path, "rb") as f:
                candidate_dict = pickle.load(f)

        return candidate_dict


def _get_parser():
    parser = argparse.ArgumentParser(description='NLP Tasks - Dialogue Retrieval')
    # Basic arguments
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--tokenizer_name', type=str, required=True)
    parser.add_argument('--candidate_vectors_path', type=str, required=False, default=None)
    parser.add_argument('--candidates_path', type=str, required=False, default=None)
    parser.add_argument('--batch_size', type=int, default=128)
    return parser


def main():
    histories = list()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = _get_parser()
    args = parser.parse_args()

    chatbot = Chatbot(args.checkpoint_path,
                      args.tokenizer_name,
                      args.candidate_vectors_path,
                      args.candidates_path,
                      args.batch_size,
                      device)

    while True:
        user_response = input("YOUR TURN: ")
        bot_response = chatbot(user_response, histories)
        print(f"BOT: {bot_response}")


if __name__ == '__main__':
    main()
