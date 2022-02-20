# MIT License
# code by Soohwan Kim @sooftware

import re
import torch
from konlpy.tag import Mecab
from itertools import chain
from collections import Counter

NUM_KNOWLEDGE = 10
KNOWLEDGE = "<knowledge>"
CONVERSATION = "<conversation>"

m = torch.distributions.geometric.Geometric(torch.tensor([0.5]))


re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')
mecab_tokenizer = Mecab().morphs


def normalize_answer(s):
    """
    From Parlai, lower text and remove punctuation, articles and extra whitespace.
    """
    s = s.lower()
    s = re_punc.sub(' ', s)
    s = s.strip()
    # TODO: this could almost certainly be faster with a regex \s+ -> ' '
    s = ' '.join(s.split())
    return s


def f1_score(golds, preds):
    global mecab_tokenizer

    f1_scores = list()

    for gold_str, pred_str in zip(golds, preds):
        g_tokens = mecab_tokenizer(normalize_answer(gold_str))
        p_tokens = mecab_tokenizer(normalize_answer(pred_str).replace('<s>', ''))

        common = Counter(g_tokens) & Counter(p_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            f1_scores.append(0)
            continue
        precision = 1.0 * num_same / len(p_tokens)
        recall = 1.0 * num_same / len(g_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        f1_scores.append(f1)
    return f1_scores


def format_turn(example, sep_token="</s>"):
    gold_knowledge = example["dialog"][-1]["checked_sentence_value"]
    last_dialog = example["dialog"][-1]
    speaker = last_dialog["speaker"]
    background_knowledge = map(lambda x: x["values"], last_dialog["retrieved_passages"])
    background_knowledge = list(chain(*background_knowledge))[:NUM_KNOWLEDGE]
    background_knowledge = [s for s in background_knowledge if s != gold_knowledge]
    background_knowledge = background_knowledge[:(NUM_KNOWLEDGE-1)]
    assert gold_knowledge not in background_knowledge
    if gold_knowledge == "no_passages_used" or gold_knowledge == "":
        gold_knowledge_idx = -100
    else:
        gold_knowledge_idx = min(int(m.sample().item()), NUM_KNOWLEDGE-2)
        background_knowledge.insert(gold_knowledge_idx, gold_knowledge)
    num_kp = NUM_KNOWLEDGE - len(background_knowledge)
    background_knowledge += [""]*num_kp
    background_knowledge = list(map(lambda x: f"{KNOWLEDGE}{x}", background_knowledge))
    assert len(background_knowledge) == NUM_KNOWLEDGE, num_kp
    dialog_history = ["<start_of_conversation>"] + list(map(lambda x: x["text"], example["dialog"]))
    next_utterance = dialog_history.pop()
    dialog_history = CONVERSATION + sep_token.join(list(reversed(dialog_history)))
    return {
        "history": dialog_history,
        "target": next_utterance,
        "background": background_knowledge,
        "background_target": gold_knowledge_idx,
        "gold_knowledge": gold_knowledge,
        "speaker": speaker
    }
