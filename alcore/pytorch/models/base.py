import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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


class VarLenLSTM(nn.Module):
    """A generic LSTM module which efficiently handles batches of variable length (ie padded) sequences using
    packing and unpacking
    """
    def __init__(self, input_size, hidden_size, num_layers=1):
        """
        Args:
            input_size (int): Dimensionality of LSTM input vector
            hidden_size (int): Dimensionality of LSTM hidden state and cell state
        """
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, num_layers=num_layers)

    def forward(self, x, x_lens, h_0, c_0):
        """
        Most of the code here handles the packing and unpacking that's required to efficiently
        perform LSTM calculations for variable length sequences.
        Args:
            x: Distributed input tensors, ready for direct input to LSTM
            x_lens: Sequence lengths of examples in the batch
            h_0: Tensor to initialize LSTM hidden state
            c_0: Tensor to initialize LSTM cell state
        Returns:
            out_padded, (h_n, c_n)
            It returns the same as the underlying PyTorch LSTM; see PyTorch docs.
        """
        max_seq_len = x.size(1)
        sorted_lens, idx = x_lens.sort(dim=0, descending=True)
        x_sorted = x[idx]
        x_packed = pack_padded_sequence(x_sorted, lengths=sorted_lens, batch_first=True)
        out_packed, (h_n, c_n) = self.lstm(x_packed, (h_0, c_0))
        out_padded, _ = pad_packed_sequence(out_packed, batch_first=True, total_length=max_seq_len)
        _, reverse_idx = idx.sort(dim=0, descending=False)
        out_padded = out_padded[reverse_idx]
        h_n = h_n[:, reverse_idx]
        c_n = c_n[:, reverse_idx]
        return out_padded, (h_n, c_n)