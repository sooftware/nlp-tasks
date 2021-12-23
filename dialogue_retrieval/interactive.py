# MIT License
# code by Soohwan Kim @sooftware

import argparse
import pickle
import torch
import numpy as np
import os
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import List

from model import PolyEncoderModel
from dataset import ContextParser


class RetrievalBot:
    def __init__(
        self,
        checkpoint_path: str,
        tokenizer_name: str,
        candidate_vectors_path: str,
        candidates_path: str,
        batch_size: int,
        num_poly_codes: int,
        device: torch.device,
    ):
        super(RetrievalBot, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_auth_token=True)

        model = PolyEncoderModel(tokenizer_name, num_poly_codes)
        print("Load state dict..")
        success = model.load_state_dict(torch.load(checkpoint_path))
        print(success)
        self.model = model.to(device)
        self.model.eval()

        self.context_parser = ContextParser(self.tokenizer, 509)
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
        histories.append(user_response)

        context = self.sep_token.join(histories)
        context_dict = self.tokenizer(context, return_tensors="pt")
        context, context_mask = context_dict["input_ids"].to(self.device), context_dict["attention_mask"].to(self.device)

        if len(context) >= 512:
            for _ in range(3):
                del histories[0]

        print(self.tokenizer.batch_decode(context))

        with torch.no_grad():
            embeddings = self.model.encode_context(context, context_mask)

            for idx in tqdm(range(0, len(self.vectors), self.batch_size)):
                vectors = torch.tensor(self.vectors[idx:idx + self.batch_size]).to(self.device)
                score = self.model.get_score(embeddings, vectors)
                score = score.squeeze().tolist()
                scores.extend(score)

        index = -1

        while True:
            sorted_index = np.argsort(scores)
            bot_response = self.texts[sorted_index[index]]

            if bot_response in histories:
                index -= 1
            else:
                histories.append((bot_response))
                break

        return bot_response

    def prepare_candidate_vectors(
            self,
            candidate_vectors_path: str = None,
            candidates_path: str = None,
            batch_size: int = 128,
    ):
        assert candidate_vectors_path is not None or candidates_path is not None

        if candidates_path is not None:
            candidate_dict = {
                "text": list(),
                "vector": list(),
            }

            with open(candidates_path, encoding='utf-8-sig') as fr:
                candidates = fr.readlines()
                with torch.no_grad():
                    for idx in tqdm(range(0, len(candidates), batch_size)):
                        batch = candidates[idx:idx + batch_size]
                        inputs = self.tokenizer(
                            batch,
                            return_tensors="pt",
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
    parser.add_argument('--num_poly_codes', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=128)
    return parser


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    histories = list()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = _get_parser()
    args = parser.parse_args()

    chatbot = RetrievalBot(args.checkpoint_path,
                           args.tokenizer_name,
                           args.candidate_vectors_path,
                           args.candidates_path,
                           args.batch_size,
                           args.num_poly_codes,
                           device)

    while True:
        user_response = input("YOUR TURN: ")
        bot_response = chatbot(user_response, histories)
        print(f"BOT: {bot_response}")


if __name__ == '__main__':
    main()
