import torch
import torch.nn.Functional as F


class IterMeter(object):
    """Keeps track of total iterations"""

    def __init__(self):
        self.val = 0

    def step(self):
        self.val += 1

    def get(self):
        return self.val


def avg_wer(wer_scores, combined_ref_len):
    return float(sum(wer_scores)) / float(combined_ref_len)


def _levenstein_distance(ref, hyp):
    pass


def word_errors():
    pass


def char_errors():
    pass


def wer():
    pass


def cer():
    pass
