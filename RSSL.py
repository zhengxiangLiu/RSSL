import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.distributions.normal import Normal
from RKNLayer import RKNLayer
from RKNCell import RKNCell
from typing import Iterable, Tuple, List, Union
import numpy as np
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from torch.distributions import kl_divergence,Independent
from torch_geometric.nn import GATConv
class TransformerBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(TransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(hidden_size, num_layers)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x,y, pad_mask=None):
        # self-attention
        stock_data = torch.transpose(x, 2, 1)
        relation_data = torch.matmul(stock_data, y)
        relation_data = F.normalize(relation_data, p=2.0, dim=1)
        relation_data = torch.transpose(relation_data, 2, 1)
        x = relation_data + x
        attn_output, _ = self.self_attn(x, x, x, attn_mask=None, key_padding_mask=pad_mask)
        x = self.norm1(x + self.dropout1(attn_output))
        ff_output = self.fc2(F.relu(self.fc1(x)))
        x = self.norm2(x + self.dropout2(ff_output))
        return x

def reparameterize(mu, var):
    """
    Samples z from a multivariate Gaussian with diagonal covariance matrix using the
    reparameterization trick.
    """
    d = Normal(torch.Tensor([0.]).cuda(), torch.Tensor([1.]).cuda())
    r = d.sample(mu.size()).squeeze(-1)
    return r * var.float() + mu.float()

class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, cell_config):
        super(TransformerModel, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.encoder1=Encoder(hidden_size, hidden_size, hidden_size)
        self._rkn_layer = RKNLayer(latent_obs_dim=hidden_size, cell_config=cell_config).cuda()

        self.line1 = torch.nn.Linear(hidden_size, 1)
        self.output_in_dim=hidden_size
        self.output_out_dim=hidden_size
        self.output = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, self.output_in_dim // 2),
            torch.nn.ReLU(inplace=False),
            torch.nn.Linear(self.output_in_dim // 2, self.output_in_dim // 4),
            torch.nn.ReLU(inplace=False),
            torch.nn.Linear(self.output_in_dim // 4, self.output_out_dim),
        )
        log_ic_init = var_activation_inverse(10.0)
        self._initial_mean = torch.zeros(1, hidden_size*2).cuda()
        self._log_icu = torch.nn.Parameter(log_ic_init * torch.ones(1, hidden_size).cuda())
        self._log_icl = torch.nn.Parameter(log_ic_init * torch.ones(1, hidden_size).cuda())
        self._ics = torch.zeros(1, hidden_size).cuda()
        self.transformer_blocks = nn.ModuleList([TransformerBlock(hidden_size, hidden_size, num_heads, 0.1) for _ in range(num_layers)])

        self.linmean=nn.Linear(hidden_size*2,hidden_size)
        self.linvar = nn.Linear(hidden_size*3,hidden_size)


    def forward(self, inputs,relation_data,pad_mask=None):
        x = inputs
        x = torch.transpose(x, 1, 0)
        get_all_out=[]
        encoder_outputs = self.encoder(x)
        for transformer_block in self.transformer_blocks:
            encoder_outputs = transformer_block(encoder_outputs,relation_data, pad_mask)
        encoder_outputs = torch.transpose(encoder_outputs, 1, 0)
        mu, log_var = self.encoder1(encoder_outputs)

        post_mean, post_cov = self._rkn_layer(mu, log_var, self._initial_mean,
                                              [var_activation(self._log_icu), var_activation(self._log_icl), self._ics])
        z_mean=self.linmean(post_mean)
        z_var=self.linvar(torch.cat(post_cov, dim=-1))
        z_mean = F.normalize(z_mean, p=2.0, dim=1)
        z_var = F.relu(F.normalize(z_var, p=2.0, dim=1))
        all_hiddens=reparameterize(z_mean,z_var)
        output = []
        all_hiddens = torch.transpose(all_hiddens, 1, 0)
        pred=self.line1(all_hiddens[-1])
        return pred


class TemporalGATModel(nn.Module):
    def __init__(self, num_features, num_relations, num_classes, num_heads, hidden_units):
        super(TemporalGATModel, self).__init__()
        self.num_features = num_features
        self.num_relations = num_relations
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.hidden_units = hidden_units

        self.conv1 = GATConv(num_features, hidden_units, heads=num_heads)
        self.conv2 = GATConv(hidden_units * num_heads, num_features, heads=1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=0.6, training=self.training)
        edge_weight = data.edge_attr
        x = self.conv1(x, edge_index)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)

        return x


def var_activation_inverse(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        inverse of elu+1, numpy only, for initialization
        :param x: input
        :return:
        """
        return np.log(np.exp(x) - 1.0)

def var_activation(x: torch.Tensor) -> torch.Tensor:
        """
        elu + 1 activation faction to ensure positive covariances
        :param x: input
        :return: exp(x) if x < 0 else x + 1
        """
        return torch.log(torch.exp(x) + 1.0)

def reparameterize(mu, var):
    """
    Samples z from a multivariate Gaussian with diagonal covariance matrix using the
    reparameterization trick.
    """
    d = Normal(torch.Tensor([0.]).cuda(), torch.Tensor([1.]).cuda())
    r = d.sample(mu.size()).squeeze(-1)
    return r * var.float() + mu.float()



class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, z_dim):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.z_dim = z_dim

        modules = nn.ModuleList()
        for _ in range(3):
            modules.append(nn.Sequential(
                nn.Linear(hidden_dims, hidden_dims),
                nn.LeakyReLU(),
            ))
            hidden_dims = hidden_dims

        self.hidden = modules
        self.mu = nn.Linear(hidden_dims, z_dim)
        self.log_var = nn.Linear(hidden_dims, z_dim)

    def forward(self, x):
        for layer in self.hidden:
            x = layer(x)
        mu = self.mu(x)
        log_var = F.relu(self.log_var(x))
        return mu, log_var

