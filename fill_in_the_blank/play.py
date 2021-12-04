# MIT License
# code by Soohwan Kim @sooftware

import argparse
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM


def _get_parser():
    parser = argparse.ArgumentParser(description='NLP Tasks - Fill in the Blank')
    # Basic arguments
    parser.add_argument('--pretrain_model_name_or_path', type=str, required=True)
    parser.add_argument('--pretrain_tokenizer_name', type=str, required=True)
    parser.add_argument('--mask_token', type=str, default='[MASK]')
    return parser


def main():
    parser = _get_parser()
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_tokenizer_name)
    model = AutoModelForMaskedLM.from_pretrained(args.pretrain_model_name_or_path)

    unmasker = pipeline("fill-mask", tokenizer=tokenizer, model=model)

    tokenizer_mask_token = tokenizer.mask_token

    while True:
        inputs = input("INPUT: ")
        inputs = inputs.replace(args.mask_token, tokenizer_mask_token)

        print("Output:\n", unmasker(inputs))


if __name__ == '__main__':
    main()
