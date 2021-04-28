import torch
import torch.nn.functional as F
from torch import nn
from .base import FCLayer
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


class VAEBaseEncoder(nn.Module):
    """Standard encoder module for variational autoencoders with tabular input and
    independent Gaussian variational posterior.
    """

    def __init__(self, data_size, hidden_sizes, dropouts, latent_size, bn=False):
        """
        Args:
            data_size (int): Dimensionality of the input data.
            hidden_sizes (list[int]): Sizes of hidden layers (not including the
                input layer or the latent layer).
            latent_size (int): Dimensionality of the latent space.
            bn (bool): Whether or not to use batch norm in the hidden layers.
        """
        super().__init__()

        self.data_size = data_size
        self.latent_size = latent_size
        self.output_size = self.latent_size

        # construct the encoder
        encoder_szs = [data_size] + hidden_sizes
        encoder_layers = []
        for in_sz, out_sz, p in zip(encoder_szs[:-1], encoder_szs[1:], dropouts):
            fc_layer = FCLayer(in_sz, out_sz, bn=bn, dropout=p)
            encoder_layers.append(fc_layer)
        self.encoder = nn.Sequential(*encoder_layers)
        self.encoder_mu = nn.Linear(encoder_szs[-1], latent_size)
        self.encoder_logvar = nn.Linear(encoder_szs[-1], latent_size)

    def encode(self, x):
        return self.encoder(x)

    def gaussian_param_projection(self, x):
        """Outputs parameters of variational posterior"""
        return self.encoder_mu(x), self.encoder_logvar(x)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.encode(x)
        mu, logvar = self.gaussian_param_projection(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
