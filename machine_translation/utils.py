# MIT License
# code by Soohwan Kim @sooftware

import sentencepiece as spm


def train_tokenizer(texts: list, vocab_size: int, model_name: str, model_type: str = 'bpe'):
    with open(f'{model_name}_{model_type}.txt', 'w', encoding="utf-8") as f:
        for text in texts:
            f.write(f'{text}\n')

    spm.SentencePieceTrainer.Train(
        f"--input={model_name}_{model_type}.txt "
        f"--model_prefix={model_name} "
        f"--vocab_size={vocab_size} "
        f"--model_type={model_type} "
        f"--pad_id=0 "
        f"--bos_id=1 "
        f"--eos_id=2 "
        f"--unk_id=3 "
    )
    sp = spm.SentencePieceProcessor()
    sp.Load(f"{model_name}.model")
    return sp
