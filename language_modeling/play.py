# MIT License
# code by Soohwan Kim @sooftware

import argparse
from transformers import AutoTokenizer, GPT2LMHeadModel


def _get_parser():
    parser = argparse.ArgumentParser(description='NLP Tasks - Language Modeling')
    # Basic arguments
    parser.add_argument('--pretrain_model_name_or_path', type=str, required=True)
    parser.add_argument('--pretrain_tokenizer_name', type=str, required=True)
    return parser


def main():
    parser = _get_parser()
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_tokenizer_name)
    model = GPT2LMHeadModel.from_pretrained(args.pretrain_model_name_or_path)

    while True:
        inputs = input("ENTER PROMPT: ")

        print("Output:")
        encoding_dict = tokenizer(inputs, return_tensors="pt")
        outputs = model.generate(input_ids=encoding_dict["input_ids"],
                                 attention_mask=encoding_dict["attention_mask"],
                                 num_beams=10,
                                 length_penalty=0.6,
                                 no_repeat_ngram_size=4,
                                 repetition_penalty=1.2)
        outputs = tokenizer.decode(outputs[0])
        print(outputs)


if __name__ == '__main__':
    main()
