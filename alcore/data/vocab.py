from ..utils import *

from collections import Counter
from itertools import chain
import copy


def calc_token_freq(series):
    """Function to quickly compute all token frequencies in a corpus. Assumes the text has already been tokenized.

    Args:
        series (iterable): An iterable (typically a pandas series) containing the document corpus

    Returns:
        A Counter object
    """
    return Counter(chain.from_iterable(map(str.split, series)))


class Vocab:
    """Basic vocabulary that creates forward and backward mappings between the tokens themselves (i.e., strings) and
    their assigned indices. Also allows for a maximum vocabulary size.
    """
    def __init__(self, text, vocab_size=None, unk_idx:int=0, pad_idx:int=1):
        """
        Args:
            text (iterable): Typically a pandas series containing the document corpus
            vocab_size (int): Number of tokens to include in the vocabulary. The top `vocab_size` tokens by frequency
                will be included.
            unk_idx (int): The vocabulary index corresponding to the "unknown" token
            pad_idx (int): The vocabulary index corresponding to the "padding" token
        """
        self.unk = "__UNK__"
        self.pad = "__PAD__"
        self.unk_idx:int = unk_idx
        self.pad_idx:int = pad_idx

        token_counts = calc_token_freq(text)

        # if a max vocab size is specified, create the vocab this way
        if vocab_size is not None:
            toks = [x[0] for x in token_counts.most_common(vocab_size)]
        # if no max vocab size is specified, create it this way
        else:
            toks = list(token_counts.keys())

        # insert "unknown" and "padding" tokens into the vocabulary
        toks = [None, None] + toks
        toks[unk_idx] = self.unk
        toks[pad_idx] = self.pad

        # create forward and backward mappings
        self.w2i = DictWithDefaults({w: i for i, w in enumerate(toks)}, default=unk_idx)
        self.i2w = {i: w for w, i in self.w2i.items()}

        # dict with word frequencies in the same order as the core mappings
        self.w2freq = DictWithDefaults({w:token_counts[w] for w,_ in self.w2i.items()}, default=0)

        self.w2proba = self._compute_prob_dist()

    def __len__(self):
        return len(self.w2i)

    def _compute_prob_dist(self):
        """
        Computes the empirical discrete probability distribution over the vocabulary, based on
        frequencies of occurrence in the text

        Returns:
            (dict): Dictionary where keys are words and values are probabilities
        """
        bg_freq = copy.deepcopy(self.w2freq)
        Z = sum(bg_freq.values())
        for key in bg_freq:
            bg_freq[key] = bg_freq[key] / Z
        return bg_freq
