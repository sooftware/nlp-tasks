# MIT License
# code by Soohwan Kim @sooftware

import torch
import Levenshtein as Lev
from typing import Tuple


class ErrorRate(object):
    r"""
    Provides inteface of error rate calcuation.
    Note:
        Do not use this class directly, use one of the sub classes.
    """

    def __init__(self, tokenizer) -> None:
        self.total_dist = 0.0
        self.total_length = 0.0
        self.tokenizer = tokenizer

    def __call__(self, targets, y_hats):
        r"""
        Calculating error rate.
        Args:
            targets (torch.Tensor): set of ground truth
            y_hats (torch.Tensor): predicted y values (y_hat) by the model
        Returns:
            - **cer**: character error rate
        """
        dist, length = self._get_distance(targets, y_hats)
        self.total_dist += dist
        self.total_length += length
        return self.total_dist / self.total_length

    def _get_distance(self, targets: torch.Tensor, y_hats: torch.Tensor) -> Tuple[float, int]:
        r"""
        Provides total character distance between targets & y_hats
        Args: targets, y_hats
            targets (torch.Tensor): set of ground truth
            y_hats (torch.Tensor): predicted y values (y_hat) by the model
        Returns: total_dist, total_length
            - **total_dist**: total distance between targets & y_hats
            - **total_length**: total length of targets sequence
        """
        total_dist = 0
        total_length = 0

        for (target, y_hat) in zip(targets, y_hats):
            s1 = self.tokenizer.decode(target)
            s2 = self.tokenizer.decode(y_hat)

            dist, length = self.metric(s1, s2)

            total_dist += dist
            total_length += length

        return total_dist, total_length

    def metric(self, *args, **kwargs) -> Tuple[float, int]:
        raise NotImplementedError


class CharacterErrorRate(ErrorRate):
    r"""
    Computes the Character Error Rate, defined as the edit distance between the
    two provided sentences after tokenizing to characters.
    """
    def __init__(self, tokenizer):
        super(CharacterErrorRate, self).__init__(tokenizer)

    def metric(self, s1: str, s2: str) -> Tuple[float, int]:
        r"""
        Computes the Character Error Rate, defined as the edit distance between the
        two provided sentences after tokenizing to characters.
        Args: s1, s2
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        Returns: dist, length
            - **dist**: distance between target & y_hat
            - **length**: length of target sequence
        """
        s1 = s1.replace(' ', '')
        s2 = s2.replace(' ', '')

        # if '_' in sentence, means subword-unit, delete '_'
        if '_' in s1:
            s1 = s1.replace('_', '')

        if '_' in s2:
            s2 = s2.replace('_', '')

        dist = Lev.distance(s2, s1)
        length = len(s1.replace(' ', ''))

        return dist, length
