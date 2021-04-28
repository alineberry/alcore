import torch
import torch.nn.functional as F
from torch import nn


class FCLayer(nn.Module):
    """Standard base level fully-connected layer. Has the option to chain together batch norm, dropout, linear,
    and activation function (in that order). The presence of batch norm, dropout, and activation function is optional.
    """
    def __init__(self, in_sz, out_sz, bn=False, dropout=0., actn=nn.ReLU()):
        """
        Args:
            in_sz (int): The layer's input size.
            out_sz (int): The layer's output size.
            bn (bool): Whether to use batch norm.
            dropout (float): Dropout probability. If set to 0, a dropout layer is not added.
            actn (func): Activation function to use.
        """
        super().__init__()
        sublayers = []
        if bn: sublayers.append(nn.BatchNorm1d(in_sz))
        if dropout != 0: sublayers.append(nn.Dropout(dropout))
        sublayers.append(nn.Linear(in_sz, out_sz))
        if actn is not None: sublayers.append(actn)
        self.sublayers = nn.Sequential(*sublayers)

    def forward(self, x):
        return self.sublayers(x)


class WordEmbAggregator(nn.Module):
    """Module that encodes a variable length document with a bag of words methodology.  Word vectors are extracted
    from an embedding matrix and either summed or averaged to produce a single vector representation of the full
    document.
    """
    def __init__(self, word_emb, word_emb_drop=0., aggregation='sum', freeze_word_emb=False):
        """
        Args:
            word_emb (nn.Embedding): Embedding matrix for the vocabulary.
            aggregation (str): Method used to aggregate individual word vectors into a single document vector.
                Currently, only 'sum' is supported.
            freeze_word_emb (bool): Whether or not to freeze the word embedding weights during backprop.
        """
        super().__init__()

        # ensure supported aggregation mode
        assert aggregation in ['sum'], f'Invalid argument "{aggregation}" passed for the `aggregation` argument'

        self.word_emb = word_emb
        self.word_emb.weight.requires_grad = not freeze_word_emb
        self.word_emb_drop = nn.Dropout(word_emb_drop)

        # size of data being output from this aggregator. used by higher level/downstream module.
        self.output_size = self.word_emb.weight.shape[1]

    def forward(self, x_tokens, x_lens):
        token_feats = self.word_emb(x_tokens)  # (batch_size, max_seq_len, embed_size)
        token_feats = token_feats.sum(dim=1)  # (batch_size, embed_size)
        token_feats = self.word_emb_drop(token_feats)
        # token_feats = token_feats.type(torch.cuda.FloatTensor)  # TODO: is this line required?
        return token_feats


def trunc_normal_(x, mean=0., std=1.):
    "Truncated normal initialization."
    # From https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/12
    return x.normal_().fmod_(2).mul_(std).add_(mean)


def embedding(ni, nf):
    "Create an embedding layer."
    emb = nn.Embedding(ni, nf)
    # See https://arxiv.org/abs/1711.09160
    with torch.no_grad(): trunc_normal_(emb.weight, std=0.01)
    return emb

