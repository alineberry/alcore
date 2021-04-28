from ...utils import *
from .base import *

import torch
import torch.nn.functional as F
from torch import nn


def default_tabular_layers(in_sz, out_sz, n_layers=1):
    gap = in_sz - out_sz
    increments = n_layers + 1
    layers = []
    for i in range(1,increments):
        prop = 1 - i/increments
        layer_size = int(gap * prop)
        layers.append(layer_size)
    return layers


def emb_sz_rule(n_cat, max_sz=300): return int(min(max_sz, round(1.6 * n_cat**0.56)))


# TODO: make sure continuous data is getting normalized on input
class TabularModel(nn.Module):
    """General purpose tabular model that handles continuous and categorical data. Can be used to model a variety of
    ouputs:
    - Unnormalized class probabilities | binary or multiclass classification setting | output can be fed into
    `nn.CrossEntropyLoss`.
    - Unbounded continuous target variable | regression settings | output can be fed into `nn.MSELoss`
    - Bounded continuous target variable | regression settings | output can be fed into `nn.MSELoss` | uses a sigmoid
    to set upper and lower limits on the output; can make it easier for the model
    - Binary class probability | binary classification setting | by setting `out_sz=1` and `y_range=[0,1]`,
    it will output a class probability.

    source: fastai.tabular.models.TabularModel
    """
    def __init__(self, emb_szs, n_cont, out_sz, layers, ps=None, emb_drop=0., y_range=None, use_bn=True,
                 bn_final=False):
        """
        Args:
            emb_szs (list[tuple[int,int])): Definition of the embedding sizes to use for categorical data. The length
                of `emb_szs` should be equal to the number of categorical variables. The elements of `emb_szs` are
                tuples like `(cardinality, embedding dimension)`, and these should be ordered in accordance with the
                order of categorical variables in the data loader.
            n_cont (int): Number of continuous variables
            out_sz (int): Output size of this model.
            layers (list[int]): List of hidden layer sizes defining the architecture of the model
            ps (list[float]): List of dropout probabilities corresponding to `layers`
            emb_drop (float): Dropout probability to use on the embedding layers
            y_range (list[float]): Upper and lower bounds set on the model's output
            use_bn (bool): Whether or not to use batch norm in the hidden layers
            bn_final (bool): Whether or not to use batch norm in the final layer
        """
        super().__init__()
        ps = [0]*len(layers) if ps is None else ps
        ps = listify(ps, layers)
        self.embeds = nn.ModuleList([embedding(ni, nf) for ni,nf in emb_szs])
        self.emb_drop = nn.Dropout(emb_drop)
        self.bn_cont = nn.BatchNorm1d(n_cont)
        n_emb = sum(e.embedding_dim for e in self.embeds)
        self.n_emb,self.n_cont,self.y_range = n_emb,n_cont,y_range
        sizes = self.get_sizes(layers, out_sz)
        actns = [nn.ReLU(inplace=True) for _ in range(len(sizes)-2)] + [None]
        layers = []
        for i,(n_in,n_out,dp,act) in enumerate(zip(sizes[:-1],sizes[1:],[0.]+ps,actns)):
            layers.append(FCLayer(n_in, n_out, bn=use_bn and i!=0, dropout=dp, actn=act))
        if bn_final: layers.append(nn.BatchNorm1d(sizes[-1]))
        self.layers = nn.Sequential(*layers)

    def get_sizes(self, layers, out_sz):
        return [self.n_emb + self.n_cont] + layers + [out_sz]

    def forward(self, x_cat, x_cont):
        if self.n_emb != 0:
            x = [e(x_cat[:,i]) for i,e in enumerate(self.embeds)]
            x = torch.cat(x, 1)
            x = self.emb_drop(x)
        if self.n_cont != 0:
            x_cont = self.bn_cont(x_cont)
            x = torch.cat([x, x_cont], 1) if self.n_emb != 0 else x_cont
        x = self.layers(x)
        if self.y_range is not None:
            x = (self.y_range[1]-self.y_range[0]) * torch.sigmoid(x) + self.y_range[0]
        return x


class TabularEncoder(nn.Module):
    def __init__(self, emb_szs, n_cont, emb_drop):
        super().__init__()
        self.embeds = nn.ModuleList([embedding(ni, nf) for ni,nf in emb_szs])
        self.emb_drop = nn.Dropout(emb_drop)
        n_emb = sum(e.embedding_dim for e in self.embeds)
        self.n_emb,self.n_cont = n_emb,n_cont
        self.bn_cont = nn.BatchNorm1d(n_cont)
        self.output_size = self.n_emb + self.n_cont

    def forward(self, x_cat, x_cont):
        if self.n_emb != 0:
            x = [e(x_cat[:,i]) for i,e in enumerate(self.embeds)]
            x = torch.cat(x, 1)
            x = self.emb_drop(x)
        if self.n_cont != 0:
            x_cont = self.bn_cont(x_cont)
            x = torch.cat([x, x_cont], 1) if self.n_emb != 0 else x_cont
        return x

