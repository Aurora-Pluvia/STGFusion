# 导入模块
from trainer import *
import os
from STGFusion import run
import matplotlib.pyplot as plt
from pathlib import Path
import scanpy as sc
from operator import truediv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn import GCNConv, SGConv, GATConv, JumpingKnowledge, APPNP, MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch import FloatTensor
import scipy.sparse
import numpy as np
from optimized_decoder import SparseInnerProductDecoder
import torch_sparse
from torch_geometric.utils import degree
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import Sequential, BatchNorm, InstanceNorm
from typing import Callable, Iterable, Union, Tuple, Optional
import logging
from models import GCN
import math
import os
from multiprocessing.sharedctypes import Value
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import GCN
from torch_geometric.utils import degree
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn import GCNConv
from matplotlib import rcParams
import matplotlib.font_manager as fm

# 使用系统中已安装的中文字体
try:
    # 尝试使用微软雅黑或其他常见中文字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题# 设置matplotlib样式，确保白色背景
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'
except:
    print("设置中文字体失败，尝试使用备选方案")

# region 预定义模块
#################### 预定义以下模块

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, save_mem=True, use_bn=True):
        super(GCN, self).__init__()

        self.convs = nn.ModuleList()
        # self.convs.append(
        #     GCNConv(in_channels, hidden_channels, cached=not save_mem, normalize=not save_mem))
        self.convs.append(
            GCNConv(in_channels, hidden_channels, cached=not save_mem))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            # self.convs.append(
            #     GCNConv(hidden_channels, hidden_channels, cached=not save_mem, normalize=not save_mem))
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=not save_mem))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # self.convs.append(
        #     GCNConv(hidden_channels, out_channels, cached=not save_mem, normalize=not save_mem))
        self.convs.append(
            GCNConv(hidden_channels, out_channels, cached=not save_mem))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        # x = data.graph['node_feat']
        # edge_index=data.graph['edge_index']
        # edge_weight=data.graph['edge_weight'] if 'edge_weight' in data.graph else None
        for i, conv in enumerate(self.convs[:-1]):
            if edge_weight is None:
                x = conv(x, edge_index)
            else:
                x = conv(x, edge_index, edge_weight)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


def full_attention_conv(qs, ks, vs, output_attn=False):
    # normalize input
    qs = qs / torch.norm(qs, p=2)  # [N, H, M]
    ks = ks / torch.norm(ks, p=2)  # [L, H, M]
    N = qs.shape[0]

    # numerator
    kvs = torch.einsum("lhm,lhd->hmd", ks, vs)
    attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs)  # [N, H, D]
    attention_num += N * vs

    # denominator
    all_ones = torch.ones([ks.shape[0]]).to(ks.device)
    ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
    attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum)  # [N, H]

    # attentive aggregated results
    attention_normalizer = torch.unsqueeze(
        attention_normalizer, len(attention_normalizer.shape))  # [N, H, 1]
    attention_normalizer += torch.ones_like(attention_normalizer) * N
    attn_output = attention_num / attention_normalizer  # [N, H, D]

    # compute attention for visualization if needed
    if output_attn:
        attention = torch.einsum("nhm,lhm->nlh", qs, ks).mean(dim=-1)  # [N, N]
        normalizer = attention_normalizer.squeeze(dim=-1).mean(dim=-1, keepdims=True)  # [N,1]
        attention = attention / normalizer

    if output_attn:
        return attn_output, attention
    else:
        return attn_output


class TransConvLayer(nn.Module):
    '''
    transformer with fast attention
    '''

    def __init__(self, in_channels,
                 out_channels,
                 num_heads,
                 use_weight=True):
        super().__init__()
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        if use_weight:
            self.Wv = nn.Linear(in_channels, out_channels * num_heads)

        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        if self.use_weight:
            self.Wv.reset_parameters()

    def forward(self, query_input, source_input, edge_index=None, edge_weight=None, output_attn=False):
        # feature transformation
        query = self.Wq(query_input).reshape(-1,
                                             self.num_heads, self.out_channels)
        key = self.Wk(source_input).reshape(-1,
                                            self.num_heads, self.out_channels)
        if self.use_weight:
            value = self.Wv(source_input).reshape(-1,
                                                  self.num_heads, self.out_channels)
        else:
            value = source_input.reshape(-1, 1, self.out_channels)

        # compute full attentive aggregation
        if output_attn:
            attention_output, attn = full_attention_conv(
                query, key, value, output_attn)  # [N, H, D]
        else:
            attention_output = full_attention_conv(
                query, key, value)  # [N, H, D]

        final_output = attention_output
        final_output = final_output.mean(dim=1)

        if output_attn:
            return final_output, attn
        else:
            return final_output


class TransConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, num_heads=1,
                 alpha=0.5, dropout=0.5, use_bn=True, use_residual=True, use_weight=True, use_act=False):
        super().__init__()

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(hidden_channels))
        for i in range(num_layers):
            self.convs.append(
                TransConvLayer(hidden_channels, hidden_channels, num_heads=num_heads, use_weight=use_weight))
            self.bns.append(nn.LayerNorm(hidden_channels))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.residual = use_residual
        self.alpha = alpha
        self.use_act = use_act

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        # x = data.graph['node_feat']
        # edge_index = data.graph['edge_index']
        # edge_weight = data.graph['edge_weight'] if 'edge_weight' in data.graph else None
        layer_ = []

        # input MLP layer
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        layer_.append(x)

        for i, conv in enumerate(self.convs):
            # graph convolution with full attention aggregation
            x = conv(x, x, edge_index, edge_weight)
            if self.residual:
                x = self.alpha * x + (1 - self.alpha) * layer_[i]
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_.append(x)

        return x

    def get_attentions(self, x):
        layer_, attentions = [], []
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        layer_.append(x)
        for i, conv_ in enumerate(self.convs):
            x, attn = conv_(x, x, output_attn=True)
            attentions.append(attn)
            if self.residual:
                x = self.alpha * x + (1 - self.alpha) * layer_[i]
            if self.use_bn:
                x = self.bns[i + 1](x)
            layer_.append(x)
        return torch.stack(attentions, dim=0)  # [layer num, N, N]


class SGFormer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, num_heads=1,
                 alpha=0.5, dropout=0.5, use_bn=True, use_residual=True, use_weight=True, use_graph=False,
                 use_act=False, graph_weight=0.8, gnn=None, aggregate='add'):
        super().__init__()
        self.trans_conv = TransConv(in_channels, hidden_channels, num_layers, num_heads, alpha, dropout, use_bn,
                                    use_residual, use_weight)
        self.gnn = gnn
        self.use_graph = use_graph
        self.graph_weight = graph_weight
        self.use_act = use_act

        self.aggregate = aggregate

        if aggregate == 'add':
            self.fc = nn.Linear(hidden_channels, out_channels)
        elif aggregate == 'cat':
            self.fc = nn.Linear(2 * hidden_channels, out_channels)
        else:
            raise ValueError(f'Invalid aggregate type:{aggregate}')

        self.params1 = list(self.trans_conv.parameters())
        self.params2 = list(self.gnn.parameters()) if self.gnn is not None else []
        self.params2.extend(list(self.fc.parameters()))

    def forward(self, x, edge_index, edge_weight=None):
        x1 = self.trans_conv(x, edge_index, edge_weight=None)
        if self.use_graph:
            x2 = self.gnn(x, edge_index, edge_weight)
            # print('x1:', x1.shape)
            # print('x2:', x2.shape)
            if self.aggregate == 'add':
                x = self.graph_weight * x2 + (1 - self.graph_weight) * x1
            else:
                x = torch.cat((x1, x2), dim=1)
        else:
            x = x1
        x = self.fc(x)
        return x

    def get_attentions(self, x):
        attns = self.trans_conv.get_attentions(x)  # [layer num, N, N]
        return attns

    def reset_parameters(self):
        self.trans_conv.reset_parameters()
        if self.use_graph:
            self.gnn.reset_parameters()


def buildNetwork(
        in_features,
        out_features,
        activate="relu",
        p_drop=0.0
):
    net = []
    net.append(nn.Linear(in_features, out_features))
    net.append(BatchNorm(out_features, momentum=0.01, eps=0.001))
    if activate == "relu":
        net.append(nn.ELU())
    elif activate == "sigmoid":
        net.append(nn.Sigmoid())
    if p_drop > 0:
        net.append(nn.Dropout(p_drop))
    return nn.Sequential(*net)


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(
            self,
            dropout,
            act=torch.sigmoid,
    ):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(
            self,
            z,
    ):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj


class GradientReverseLayer(torch.autograd.Function):
    """Layer that reverses and scales gradients before
    passing them up to earlier ops in the computation graph
    during backpropogation.
    """

    @staticmethod
    def forward(ctx, x, weight):
        """
        Perform a no-op forward pass that stores a weight for later
        gradient scaling during backprop.
        Parameters
        ----------
        x : torch.FloatTensor
            [Batch, Features]
        weight : float
            weight for scaling gradients during backpropogation.
            stored in the "context" ctx variable.
        Notes
        -----
        We subclass `Function` and use only @staticmethod as specified
        in the newstyle pytorch autograd functions.
        https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function
        We define a "context" ctx of the class that will hold any values
        passed during forward for use in the backward pass.
        `x.view_as(x)` and `*1` are necessary so that `GradReverse`
        is actually called
        `torch.autograd` tries to optimize backprop and
        excludes no-ops, so we have to trick it :)
        """
        # store the weight we'll use in backward in the context
        ctx.weight = weight
        return x.view_as(x) * 1.0

    @staticmethod
    def backward(ctx, grad_output):
        """Return gradients
        Returns
        -------
        rev_grad : torch.FloatTensor
            reversed gradients scaled by `weight` passed in `.forward()`
        None : None
            a dummy "gradient" required since we passed a weight float
            in `.forward()`.
        """
        # here scale the gradient and multiply by -1
        # to reverse the gradients
        return (grad_output * -1 * ctx.weight), None


class AdversarialNetwork(nn.Module):
    """Build a Graph Convolutional Adversarial Network
       for semi-supervised Domain Adaptation. """

    def __init__(
            self,
            model,
            n_domains: int = 2,
            weight: float = 1,
            n_layers: int = 2,
    ) -> None:
        """
        Parameters
        ----------
        model : ExtractDEF
            cell type classification model.
        n_domains : int
            number of domains to adapt.
        weight : float
            weight for reversed gradients.
        n_layers : int
            number of hidden layers in the network.

        Returns
        -------
        None.
        """
        super(AdversarialNetwork, self).__init__()
        self.model = model
        self.n_domains = n_domains
        self.n_layers = n_layers
        self.weight = weight

        hidden_layers = [
                            nn.Linear(self.model.linear_encoder_hidden[-1] + self.model.conv_hidden[-1],
                                      self.model.linear_encoder_hidden[-1] + self.model.conv_hidden[-1]),
                            nn.ReLU(),
                        ] * n_layers

        self.domain_clf = nn.Sequential(
            *hidden_layers,
            nn.Linear(self.model.linear_encoder_hidden[-1] + self.model.conv_hidden[-1], self.n_domains),
        )

        return

    def set_rev_grad_weight(
            self,
            weight: float,
    ) -> None:
        """Set the weight term used after reversing gradients"""
        self.weight = weight
        return

    def target_distribution(
            self,
            target
    ):
        weight = (target ** 2) / torch.sum(target, 0)
        return (weight.t() / torch.sum(weight, 1)).t()

    def deepst_loss(
            self,
            decoded,
            x,
            preds,
            labels,
            mu,
            logvar,
            n_nodes,
            norm,
            mask=None,
            mse_weight=10,
            bce_kld_weight=0.1,
    ):
        mse_fun = torch.nn.MSELoss()
        mse_loss = mse_fun(decoded, x)

        if mask is not None:
            preds = preds * mask
            labels = labels * mask

        bce_logits_loss = norm * F.binary_cross_entropy_with_logits(preds, labels)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 / n_nodes * torch.mean(torch.sum(
            1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
        return mse_weight * mse_loss + bce_logits_loss + bce_kld_weight * KLD

    def forward(
            self,
            x: torch.FloatTensor,
            edge_index,
    ) -> torch.FloatTensor:
        """Perform a forward pass.

        Parameters
        ----------
        x : torch.FloatTensor
            [Batch, Features] input.

        Returns
        -------
        domain_pred : torch.FloatTensor
            [Batch, n_domains] logits.
        x_embed : torch.FloatTensor
            [Batch, n_hidden]
        """
        # reverse gradients and scale by a weight
        # domain_pred -> x_rev -> GradientReverseLayer -> x_embed
        #      d+     ->  d+   ->     d-      ->   d-
        z, mu, logvar, de_feat, q, feat_x, gnn_z = self.model(x, edge_index)
        x_rev = GradientReverseLayer.apply(
            z,
            self.weight,
        )
        # classify the domains
        domain_pred = self.domain_clf(x_rev)
        return z, mu, logvar, de_feat, q, feat_x, gnn_z, domain_pred


###################################下面是重定义的模型，增加了两个算法：
#########################################- GCN+Transformer
#############################################- SGFormer

class DeepST_model(nn.Module):
    def __init__(self,
                 input_dim,
                 Conv_type='GCNConv',
                 linear_encoder_hidden=[32, 20],
                 linear_decoder_hidden=[32],
                 conv_hidden=[32, 8],
                 p_drop=0.01,
                 dec_cluster_n=15,
                 alpha=0.9,
                 activate="relu",
                 ):
        super(DeepST_model, self).__init__()
        self.input_dim = input_dim
        self.Conv_type = Conv_type
        self.alpha = alpha
        self.conv_hidden = conv_hidden
        self.linear_encoder_hidden = linear_encoder_hidden
        self.linear_decoder_hidden = linear_decoder_hidden
        self.activate = activate
        self.p_drop = p_drop
        self.dec_cluster_n = dec_cluster_n

        current_encoder_dim = self.input_dim
        ### a deep autoencoder network
        self.encoder = nn.Sequential()
        for le in range(len(linear_encoder_hidden)):
            self.encoder.add_module(f'encoder_L{le}',
                                    buildNetwork(current_encoder_dim, self.linear_encoder_hidden[le], self.activate,
                                                 self.p_drop))
            current_encoder_dim = linear_encoder_hidden[le]
        current_decoder_dim = linear_encoder_hidden[-1] + conv_hidden[-1]

        self.decoder = nn.Sequential()
        for ld in range(len(linear_decoder_hidden)):
            self.decoder.add_module(f'decoder_L{ld}',
                                    buildNetwork(current_decoder_dim, self.linear_decoder_hidden[ld], self.activate,
                                                 self.p_drop))
            current_decoder_dim = self.linear_decoder_hidden[ld]
        self.decoder.add_module(f'decoder_L{len(self.linear_decoder_hidden)}', nn.Linear(self.linear_decoder_hidden[-1],
                                                                                         self.input_dim))

        #### a variational graph autoencoder based on pytorch geometric
        '''https://pytorch-geometric.readthedocs.io/en/latest/index.html'''

        # GCN layers
        if self.Conv_type == "GCNConv":
            '''https://arxiv.org/abs/1609.02907'''
            from torch_geometric.nn import GCNConv
            self.conv = Sequential('x, edge_index', [
                (GCNConv(linear_encoder_hidden[-1], conv_hidden[0] * 2), 'x, edge_index -> x1'),
                BatchNorm(conv_hidden[0] * 2),
                nn.ReLU(inplace=True),
            ])
            self.conv_mean = Sequential('x, edge_index', [
                (GCNConv(conv_hidden[0] * 2, conv_hidden[-1]), 'x, edge_index -> x1'),
            ])
            self.conv_logvar = Sequential('x, edge_index', [
                (GCNConv(conv_hidden[0] * 2, conv_hidden[-1]), 'x, edge_index -> x1'),
            ])

        ############################################
        # 这是新增的GCN+Transformer
        ############################################
        if self.Conv_type == "GCN_Transformer_Conv":
            '''https://arxiv.org/abs/1609.02907'''
            from torch_geometric.nn import GCNConv
            from torch_geometric.nn import TransformerConv
            self.conv = Sequential('x, edge_index', [
                (GCNConv(linear_encoder_hidden[-1], conv_hidden[0] * 2), 'x, edge_index -> x1'),
                BatchNorm(conv_hidden[0] * 2),
                nn.ReLU(inplace=True),
                (TransformerConv(linear_encoder_hidden[-1], conv_hidden[0] * 2), 'x, edge_index -> x1'),
                BatchNorm(conv_hidden[0] * 2),
                nn.ReLU(inplace=True),
            ])
            self.conv_mean = Sequential('x, edge_index', [
                (GCNConv(conv_hidden[0] * 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                BatchNorm(conv_hidden[-1]),
                nn.ReLU(inplace=True),
                (TransformerConv(conv_hidden[0] * 2, conv_hidden[-1]), 'x, edge_index -> x1'),
            ])
            self.conv_logvar = Sequential('x, edge_index', [
                (GCNConv(conv_hidden[0] * 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                BatchNorm(conv_hidden[-1]),
                nn.ReLU(inplace=True),
                (TransformerConv(conv_hidden[0] * 2, conv_hidden[-1]), 'x, edge_index -> x1'),
            ])

        ############################################
        # 这是新增的SGFormer
        ############################################
        elif self.Conv_type == 'SGFormer':
            gcn_1 = GCN(in_channels=linear_encoder_hidden[-1],
                        hidden_channels=conv_hidden[0] * 2,
                        out_channels=conv_hidden[0] * 2)
            self.conv = Sequential('x, edge_index', [
                (SGFormer(in_channels=linear_encoder_hidden[-1],
                          hidden_channels=conv_hidden[0] * 2,
                          out_channels=conv_hidden[0] * 2,
                          gnn=gcn_1,
                          use_graph=True), 'x, edge_index -> x1'),
                BatchNorm(conv_hidden[0] * 2),
                nn.ReLU(inplace=True),
            ])

            gcn_2 = GCN(in_channels=conv_hidden[0] * 2,
                        hidden_channels=conv_hidden[0] * 2,
                        out_channels=conv_hidden[-1])
            self.conv_mean = Sequential('x, edge_index', [
                (SGFormer(in_channels=conv_hidden[0] * 2,
                          hidden_channels=conv_hidden[-1],
                          out_channels=conv_hidden[-1],
                          gnn=gcn_2,
                          use_graph=True), 'x, edge_index -> x1'),
            ])

            gcn_3 = GCN(in_channels=conv_hidden[0] * 2,
                        hidden_channels=conv_hidden[0] * 2,
                        out_channels=conv_hidden[-1])
            self.conv_logvar = Sequential('x, edge_index', [
                (SGFormer(in_channels=conv_hidden[0] * 2,
                          hidden_channels=conv_hidden[-1],
                          out_channels=conv_hidden[-1],
                          gnn=gcn_3,
                          use_graph=True), 'x, edge_index -> x1'),
            ])

        elif self.Conv_type == "SAGEConv":
            from torch_geometric.nn import SAGEConv
            self.conv = Sequential('x, edge_index', [
                (SAGEConv(linear_encoder_hidden[-1], conv_hidden[0] * 2), 'x, edge_index -> x1'),
                BatchNorm(conv_hidden[0] * 2),
                nn.ReLU(inplace=True),
            ])
            self.conv_mean = Sequential('x, edge_index', [
                (SAGEConv(conv_hidden[0] * 2, conv_hidden[-1]), 'x, edge_index -> x1'),
            ])
            self.conv_logvar = Sequential('x, edge_index', [
                (SAGEConv(conv_hidden[0] * 2, conv_hidden[-1]), 'x, edge_index -> x1'),
            ])

        elif self.Conv_type == "GraphConv":
            from torch_geometric.nn import GraphConv
            self.conv = Sequential('x, edge_index', [
                (GraphConv(linear_encoder_hidden[-1], conv_hidden[0] * 2), 'x, edge_index -> x1'),
                BatchNorm(conv_hidden[0] * 2),
                nn.ReLU(inplace=True),
            ])
            self.conv_mean = Sequential('x, edge_index', [
                (GraphConv(conv_hidden[0] * 2, conv_hidden[-1]), 'x, edge_index -> x1'),
            ])
            self.conv_logvar = Sequential('x, edge_index', [
                (GraphConv(conv_hidden[0] * 2, conv_hidden[-1]), 'x, edge_index -> x1'),
            ])

        elif self.Conv_type == "GatedGraphConv":
            from torch_geometric.nn import GatedGraphConv
            self.conv = Sequential('x, edge_index', [
                (GatedGraphConv(linear_encoder_hidden[-1], conv_hidden[0] * 2), 'x, edge_index -> x1'),
                BatchNorm(conv_hidden[0] * 2),
                nn.ReLU(inplace=True),
            ])
            self.conv_mean = Sequential('x, edge_index', [
                (GatedGraphConv(conv_hidden[0] * 2, conv_hidden[-1]), 'x, edge_index -> x1'),
            ])
            self.conv_logvar = Sequential('x, edge_index', [
                (GatedGraphConv(conv_hidden[0] * 2, conv_hidden[-1]), 'x, edge_index -> x1'),
            ])

        elif self.Conv_type == "ResGatedGraphConv":
            from torch_geometric.nn import ResGatedGraphConv
            self.conv = Sequential('x, edge_index', [
                (ResGatedGraphConv(linear_encoder_hidden[-1], conv_hidden[0] * 2), 'x, edge_index -> x1'),
                BatchNorm(conv_hidden[0] * 2),
                nn.ReLU(inplace=True),
            ])
            self.conv_mean = Sequential('x, edge_index', [
                (ResGatedGraphConv(conv_hidden[0] * 2, conv_hidden[-1]), 'x, edge_index -> x1'),
            ])
            self.conv_logvar = Sequential('x, edge_index', [
                (ResGatedGraphConv(conv_hidden[0] * 2, conv_hidden[-1]), 'x, edge_index -> x1'),
            ])

        elif self.Conv_type == "TransformerConv":
            from torch_geometric.nn import TransformerConv
            self.conv = Sequential('x, edge_index', [
                (TransformerConv(linear_encoder_hidden[-1], conv_hidden[0] * 2), 'x, edge_index -> x1'),
                BatchNorm(conv_hidden[0] * 2),
                nn.ReLU(inplace=True),
            ])
            self.conv_mean = Sequential('x, edge_index', [
                (TransformerConv(conv_hidden[0] * 2, conv_hidden[-1]), 'x, edge_index -> x1'),
            ])
            self.conv_logvar = Sequential('x, edge_index', [
                (TransformerConv(conv_hidden[0] * 2, conv_hidden[-1]), 'x, edge_index -> x1'),
            ])

        elif self.Conv_type == "TAGConv":
            from torch_geometric.nn import TAGConv
            self.conv = Sequential('x, edge_index', [
                (TAGConv(linear_encoder_hidden[-1], conv_hidden[0] * 2), 'x, edge_index -> x1'),
                BatchNorm(conv_hidden[0] * 2),
                nn.ReLU(inplace=True),
            ])
            self.conv_mean = Sequential('x, edge_index', [
                (TAGConv(conv_hidden[0] * 2, conv_hidden[-1]), 'x, edge_index -> x1'),
            ])
            self.conv_logvar = Sequential('x, edge_index', [
                (TAGConv(conv_hidden[0] * 2, conv_hidden[-1]), 'x, edge_index -> x1'),
            ])

        elif self.Conv_type == "ARMAConv":
            from torch_geometric.nn import ARMAConv
            self.conv = Sequential('x, edge_index', [
                (ARMAConv(linear_encoder_hidden[-1], conv_hidden[0] * 2), 'x, edge_index -> x1'),
                BatchNorm(conv_hidden[0] * 2),
                nn.ReLU(inplace=True),
            ])
            self.conv_mean = Sequential('x, edge_index', [
                (ARMAConv(conv_hidden[0] * 2, conv_hidden[-1]), 'x, edge_index -> x1'),
            ])
            self.conv_logvar = Sequential('x, edge_index', [
                (ARMAConv(conv_hidden[0] * 2, conv_hidden[-1]), 'x, edge_index -> x1'),
            ])
        elif self.Conv_type == "SGConv":
            from torch_geometric.nn import SGConv
            self.conv = Sequential('x, edge_index', [
                (SGConv(linear_encoder_hidden[-1], conv_hidden[0] * 2), 'x, edge_index -> x1'),
                BatchNorm(conv_hidden[0] * 2),
                nn.ReLU(inplace=True),
            ])
            self.conv_mean = Sequential('x, edge_index', [
                (SGConv(conv_hidden[0] * 2, conv_hidden[-1]), 'x, edge_index -> x1'),
            ])
            self.conv_logvar = Sequential('x, edge_index', [
                (SGConv(conv_hidden[0] * 2, conv_hidden[-1]), 'x, edge_index -> x1'),
            ])
        elif self.Conv_type == "MFConv":
            from torch_geometric.nn import MFConv
            self.conv = Sequential('x, edge_index', [
                (MFConv(linear_encoder_hidden[-1], conv_hidden[0] * 2), 'x, edge_index -> x1'),
                BatchNorm(conv_hidden[0] * 2),
                nn.ReLU(inplace=True),
            ])
            self.conv_mean = Sequential('x, edge_index', [
                (MFConv(conv_hidden[0] * 2, conv_hidden[-1]), 'x, edge_index -> x1'),
            ])
            self.conv_logvar = Sequential('x, edge_index', [
                (MFConv(conv_hidden[0] * 2, conv_hidden[-1]), 'x, edge_index -> x1'),
            ])
        elif self.Conv_type == "RGCNConv":
            from torch_geometric.nn import RGCNConv
            self.conv = Sequential('x, edge_index', [
                (RGCNConv(linear_encoder_hidden[-1], conv_hidden[0] * 2, num_relations=3), 'x, edge_index -> x1'),
                BatchNorm(conv_hidden[0] * 2),
                nn.ReLU(inplace=True),
            ])
            self.conv_mean = Sequential('x, edge_index', [
                (RGCNConv(conv_hidden[0] * 2, conv_hidden[-1], num_relations=3), 'x, edge_index -> x1'),
            ])
            self.conv_logvar = Sequential('x, edge_index', [
                (RGCNConv(conv_hidden[0] * 2, conv_hidden[-1], num_relations=3), 'x, edge_index -> x1'),
            ])
        elif self.Conv_type == "FeaStConv":
            from torch_geometric.nn import FeaStConv
            self.conv = Sequential('x, edge_index', [
                (FeaStConv(linear_encoder_hidden[-1], conv_hidden[0] * 2), 'x, edge_index -> x1'),
                BatchNorm(conv_hidden[0] * 2),
                nn.ReLU(inplace=True),
            ])
            self.conv_mean = Sequential('x, edge_index', [
                (FeaStConv(conv_hidden[0] * 2, conv_hidden[-1]), 'x, edge_index -> x1'),
            ])
            self.conv_logvar = Sequential('x, edge_index', [
                (FeaStConv(conv_hidden[0] * 2, conv_hidden[-1]), 'x, edge_index -> x1'),
            ])
        elif self.Conv_type == "LEConv":
            from torch_geometric.nn import LEConv
            self.conv = Sequential('x, edge_index', [
                (LEConv(linear_encoder_hidden[-1], conv_hidden[0] * 2), 'x, edge_index -> x1'),
                BatchNorm(conv_hidden[0] * 2),
                nn.ReLU(inplace=True),
            ])
            self.conv_mean = Sequential('x, edge_index', [
                (LEConv(conv_hidden[0] * 2, conv_hidden[-1]), 'x, edge_index -> x1'),
            ])
            self.conv_logvar = Sequential('x, edge_index', [
                (LEConv(conv_hidden[0] * 2, conv_hidden[-1]), 'x, edge_index -> x1'),
            ])
        elif self.Conv_type == "ClusterGCNConv":
            from torch_geometric.nn import ClusterGCNConv
            self.conv = Sequential('x, edge_index', [
                (ClusterGCNConv(linear_encoder_hidden[-1], conv_hidden[0] * 2), 'x, edge_index -> x1'),
                BatchNorm(conv_hidden[0] * 2),
                nn.ReLU(inplace=True),
            ])
            self.conv_mean = Sequential('x, edge_index', [
                (ClusterGCNConv(conv_hidden[0] * 2, conv_hidden[-1]), 'x, edge_index -> x1'),
            ])
            self.conv_logvar = Sequential('x, edge_index', [
                (ClusterGCNConv(conv_hidden[0] * 2, conv_hidden[-1]), 'x, edge_index -> x1'),
            ])
        # 使用优化的稀疏内积解码器 - 降低时间复杂度从O(n²)到O(E)
        self.dc = SparseInnerProductDecoder(dropout=p_drop)
        # DEC cluster layer
        self.cluster_layer = Parameter(
            torch.Tensor(self.dec_cluster_n, self.linear_encoder_hidden[-1] + self.conv_hidden[-1]))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def encode(
            self,
            x,
            adj,
    ):
        feat_x = self.encoder(x)
        print('feat_x: ', feat_x.shape)
        print(adj)
        conv_x = self.conv(feat_x, adj)

        print('conv_x', conv_x.shape)
        # print('mean:', self.conv_mean(conv_x, adj).shape)
        # print('conv_logvar:', self.conv_logvar(conv_x, adj).shape)
        return self.conv_mean(conv_x, adj), self.conv_logvar(conv_x, adj), feat_x

    def reparameterize(
            self,
            mu,
            logvar,
    ):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def target_distribution(
            self,
            target
    ):
        weight = (target ** 2) / torch.sum(target, 0)
        return (weight.t() / torch.sum(weight, 1)).t()

    def deepst_loss(
            self,
            decoded,
            x,
            preds,
            labels,
            mu,
            logvar,
            n_nodes,
            norm,
            mask=None,
            mse_weight=10,
            bce_kld_weight=0.1,
    ):
        mse_fun = torch.nn.MSELoss()
        mse_loss = mse_fun(decoded, x)

        if mask is not None:
            preds = preds * mask
            labels = labels * mask

        bce_logits_loss = norm * F.binary_cross_entropy_with_logits(preds, labels)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 / n_nodes * torch.mean(torch.sum(
            1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
        return mse_weight * mse_loss + bce_kld_weight * (bce_logits_loss + KLD)

    def forward(
            self,
            x,
            adj,
    ):
        mu, logvar, feat_x = self.encode(x, adj)
        # print(mu.shape, 'mu')
        gnn_z = self.reparameterize(mu, logvar)
        z = torch.cat((feat_x, gnn_z), 1)
        de_feat = self.decoder(z)

        #
        q = 1.0 / ((1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha) + 1e-8)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return z, mu, logvar, de_feat, q, feat_x, gnn_z

# endregion

# region 数据预处理
##############################################以下预处理过程与原框架一样##########################################
#visium DLPFC
data_path = "../data/1.DLPFC"  #### to your path
data_name = '151673'  #### project name
save_path = "../Results/Visium/DLPFC"  #### save path
n_domains = 7  ###### the number of spatial domains.
"""
#visium HBRC
data_path = "../data/3.Human_Breast_Cancer"  #### to your path
data_name = 'HBRC'  #### project name
save_path = "../Results/Visium/HBRC"  #### save path
n_domains = 20  ###### the number of spatial domains.

# slideseq
data_path = "../data/slideseq_30923225_MouseHippocampus/usedata"  #### to your path
data_name = 'Puck_180413_7'  #### project name
save_path = "../Results/slideseq"  #### save path
n_domains = 8  ###### the number of spatial domains.

# slideseq_h5ad (Mouse Hippocampus Tissue)
data_path = "../data/6.Mouse_Hippocampus_Tissue"  #### to your path
data_name = 'Hippocampus'  #### project name
save_path = "../Results/slideseq"  #### save path
n_domains = 8  ###### the number of spatial domains.

# stereoSeq
data_path = "../data/MouseOlfactoryBulb-StereoSeq"  #### to your path
data_name = 'data2'  #### project name
save_path = "../Results/stereoSeq"  #### save path
n_domains = 7  ###### the number of spatial domains.

# stereoSeq_h5ad
data_path = "../data/MouseOlfactoryBulb-StereoSeq"  #### to your path
data_name = 'data1'  #### project name
save_path = "../Results/stereoSeq"  #### save path
n_domains = 7  ###### the number of spatial domains.
"""

deepen = run(save_path=save_path,
             task="Identify_Domain",
             pre_epochs=800,  ####  choose the number of training
             epochs=1000,  #### choose the number of training
             use_gpu=True)

###### Read in 10x Visium data, or user can read in themselves.
# Visium、MERFISH、slideSeq、slideseq_h5ad、seqFish、stereoSeq、stereoSeq_h5ad
adata = deepen._get_adata(platform="Visium", data_path=data_path, data_name=data_name)
###### Segment the Morphological Image
adata = deepen._get_image_crop(adata, data_name=data_name)

###### Data augmentation. spatial_type includes three kinds of "KDTree", "BallTree" and "LinearRegress", among which "LinearRegress"
###### is only applicable to 10x visium and the remaining omics selects the other two.
###### "use_morphological" defines whether to use morphological images.
# spatial_type="LinearRegress"、"NearestNeighbors"、"KDTree"、"BallTree"
# 内存不足时使用：use_batch_processing：是否分批，use_sparse：是否使用稀疏矩阵
adata = deepen._get_augment(adata, spatial_type="BallTree", use_morphological=True, use_batch_processing=False, batch_size = 2500, use_sparse=False)

###### 图构建。"KDTree", "BallTree", "kneighbors_graph", "Radius", adj.py
graph_dict = deepen._get_graph(adata.obsm["spatial"], distType="BallTree")

###### Enhanced data preprocessing
data = deepen._data_process(adata, pca_n_comps=200)

print("\n===== adata基本信息 =====")
print(f"adata观测属性: {list(adata.obs.columns)}")
print(f"adata变量属性: {list(adata.var.columns)}")
print(f"adata多维观测属性: {list(adata.obsm.keys())}")
# 打印adata.obs的前5行内容
print("\n===== adata.obs的前5行内容 =====")
print(adata.obs.head())

# endregion

###### 模型训练
conv_type = "SGFormer"  # GCN_Transformer_Conv/SGFormer/GCNConv

stgfusion_embed = deepen._fit(
    data=data,
    graph_dict=graph_dict,
    Conv_type=conv_type
)

###### 输出
adata.obsm["DeepST_embed"] = stgfusion_embed

###### 聚类
# adata = deepen._get_cluster_data(adata, n_domains=n_domains, priori=False)  # 无先验知识
adata = deepen._get_cluster_data(adata, n_domains=n_domains, priori=True)  # 有先验知识

print(f"adata观测属性: {list(adata.obs.columns)}")

# region 聚类结果作图
save_path += ("/ModelResult/" + conv_type)
"""sc.pl.spatial(adata, color='DeepST_refine_domain', frameon=False, spot_size=150)
plt.savefig(os.path.join(save_path, f'{data_name}_domains.pdf'), bbox_inches='tight', dpi=300)"""
fig = plt.figure(figsize=(10, 8))  # 创建一个指定大小的图形
# spot_size = 150(其他) / 250(HBRC) / 50(stereo)
sc.pl.spatial(adata, color='DeepST_refine_domain', frameon=False, spot_size=150, show=False, ax=fig.gca())
plt.tight_layout()  # 调整布局
plt.savefig(os.path.join(save_path, f'{data_name}_domains_{conv_type}.pdf'), bbox_inches='tight', dpi=300)
# 再保存一个PNG格式的图像作为备份
plt.savefig(os.path.join(save_path, f'{data_name}_domains_{conv_type}.png'), bbox_inches='tight', dpi=300)
plt.close(fig)  # 确保关闭图形，释放内存

print(f"聚类结果图已保存至: {os.path.join(save_path, f'{data_name}_domains_{conv_type}.pdf')} 和 {os.path.join(save_path, f'{data_name}_domains_{conv_type}.png')}")
# endregion

# region 计算ARI、NMI、SC、DB

# 初始化变量，确保后续代码可以访问
ari = None
nmi_score = None

# 首先尝试从metadata.tsv读取真实标签
print("\n===== 尝试从metadata.tsv读取真实标签 =====")
try:
    import pandas as pd
    metadata_file_path = os.path.join(data_path, data_name, "metadata.tsv")
    
    if os.path.exists(metadata_file_path):
        print(f"找到metadata.tsv文件: {metadata_file_path}")
        
        # 读取metadata.tsv
        metadata_df = pd.read_csv(metadata_file_path, sep="\t")
        
        print(f"\nmetadata.tsv的形状: {metadata_df.shape}")
        print(f"metadata.tsv的列名: {metadata_df.columns.tolist()}")
        print(f"\nmetadata.tsv的前5行:")
        print(metadata_df.head())
        
        # 查找可能包含真实标签的列
        possible_label_columns = ['layer_guess', 'layer', 'ground_truth', 'true_label', 'label', 'cluster', 'cell_type', 'annotation']
        label_column = None
        
        for col in possible_label_columns:
            if col in metadata_df.columns:
                label_column = col
                print(f"\n找到可能的标签列: {label_column}")
                print(f"该列的唯一值: {metadata_df[label_column].unique()}")
                print(f"该列的值计数:")
                print(metadata_df[label_column].value_counts())
                break
        
        if label_column:
            # 将标签添加到adata.obs
            # 需要确保索引对齐
            if metadata_df.index.name is None:
                # 如果没有索引列，假设第一列是索引
                metadata_df = pd.read_csv(metadata_file_path, sep="\t", index_col=0)
            
            # 对齐索引并添加到adata.obs
            adata.obs['ground_truth'] = metadata_df.loc[adata.obs.index, label_column]
            
            # 处理nan值，将其标记为'Unknown'
            adata.obs['ground_truth'] = adata.obs['ground_truth'].fillna('Unknown')
            
            print(f"\n成功从metadata.tsv加载真实标签到adata.obs['ground_truth']")
            print(f"标签类别及数量:")
            print(adata.obs['ground_truth'].value_counts())
            
            # 计算ARI
            if 'DeepST_refine_domain' in adata.obs.columns:
                from sklearn.metrics import adjusted_rand_score
                ari = adjusted_rand_score(adata.obs['ground_truth'], adata.obs['DeepST_refine_domain'])
                print(f"\nARI score: {ari:.4f}")
        else:
            print(f"\n警告: 未找到标准的标签列，请手动指定列名")
            print(f"可用的列: {metadata_df.columns.tolist()}")
    else:
        print(f"未找到metadata.tsv文件: {metadata_file_path}")
        
except Exception as e:
    print(f"读取metadata.tsv时出错: {str(e)}")
    import traceback
    traceback.print_exc()

# 读取truth.txt文件并将真实标签添加到adata中
print("\n===== 尝试从truth.txt读取真实标签 =====")
try:
    # 构建truth.txt文件的完整路径
    truth_file_path = os.path.join(data_path, data_name, "truth.txt")
    
    # 检查文件是否存在
    if os.path.exists(truth_file_path):
        # 读取truth.txt文件
        import pandas as pd
        truth_df = pd.read_csv(truth_file_path, sep="\t", header=None, index_col=0)
        
        # 如果文件没有列名，则给第二列指定一个名称
        if truth_df.shape[1] >= 1:
            truth_df.columns = ['ground_truth'] + list(truth_df.columns[1:]) if len(truth_df.columns) > 1 else ['ground_truth']
            
            truth_df['ground_truth'] = truth_df['ground_truth'].fillna(0)

            # 将真实标签添加到adata.obs中
            # 假设adata.obs的索引与truth_df的索引顺序一致
            adata.obs['ground_truth'] = truth_df['ground_truth'].values
            
            print(f"成功从truth.txt加载了{len(truth_df)}个点的真实标签")
            print(f"标签类别: {adata.obs['ground_truth'].unique()}")
            print(f"空值已被替换为0")
            
            # 如果DeepST_refine_domain列存在，计算ARI评分
            if 'DeepST_refine_domain' in adata.obs.columns:
                from sklearn.metrics import adjusted_rand_score
                ari = adjusted_rand_score(adata.obs['ground_truth'], adata.obs['DeepST_refine_domain'])
                print(f"ARI score: {ari:.4f}")
        else:
            print("警告: truth.txt文件格式不正确，无法读取真实标签")
    else:
        print(f"警告: 未找到truth.txt文件: {truth_file_path}")
except Exception as e:
    print(f"读取truth.txt文件时出错: {str(e)}")

# 只有当ground_truth存在时才打印
if 'ground_truth' in adata.obs.columns:
    print(adata.obs["ground_truth"])

# NMI、SC、DB
# 计算聚类评估指标
print("\n===== 计算聚类评估指标 =====")
from sklearn.metrics import normalized_mutual_info_score, silhouette_score, davies_bouldin_score

# 确保DeepST_refine_domain是分类数据类型
if 'DeepST_refine_domain' in adata.obs.columns:
    # 将聚类结果转换为数值型数组用于计算指标
    cluster_labels = adata.obs['DeepST_refine_domain'].astype('category').cat.codes.values
    
    # 获取嵌入表示用于计算基于距离的指标
    embed_data = adata.obsm['DeepST_embed']
    
    # 计算轮廓系数 (Silhouette Coefficient)
    # 值越接近1表示聚类效果越好
    sc_score = silhouette_score(embed_data, cluster_labels)
    print(f"轮廓系数 (Silhouette Coefficient): {sc_score:.4f}")
    
    # 计算Davies-Bouldin指数
    # 值越小表示聚类效果越好
    db_score = davies_bouldin_score(embed_data, cluster_labels)
    print(f"Davies-Bouldin指数: {db_score:.4f}")
    
    # 如果有真实标签，计算NMI
    if 'ground_truth' in adata.obs.columns:
        # 将真实标签转换为数值型数组
        true_labels = adata.obs['ground_truth'].astype('category').cat.codes.values
        
        # 计算归一化互信息 (NMI)
        # 值越接近1表示聚类结果与真实标签越一致
        nmi_score = normalized_mutual_info_score(true_labels, cluster_labels)
        print(f"归一化互信息 (NMI): {nmi_score:.4f}")
        
        # 将评估指标保存到adata.uns中
        adata.uns['clustering_metrics'] = {
            'ARI': ari if ari is not None else 0,
            'NMI': nmi_score,
            'Silhouette_Coefficient': sc_score,
            'Davies_Bouldin_Index': db_score
        }
    else:
        # 将评估指标保存到adata.uns中
        adata.uns['clustering_metrics'] = {
            'Silhouette_Coefficient': sc_score,
            'Davies_Bouldin_Index': db_score
        }
    
    # 将评估指标保存到文件（追加模式，不覆盖之前的内容）
    from datetime import datetime
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    metrics_file = os.path.join(save_path, f'{data_name}_clustering_metrics.txt')
    with open(metrics_file, 'a') as f:  # 使用 'a' 模式追加写入
        f.write(f"\n{'='*50}\n")
        f.write(f"记录时间: {current_time}\n")
        f.write(f"模型类型: {conv_type}\n")
        f.write(f"{'='*50}\n")
        if 'ground_truth' in adata.obs.columns and ari is not None:
            f.write(f"调整兰德指数 (ARI): {ari:.4f}\n")
        if 'ground_truth' in adata.obs.columns and nmi_score is not None:
            f.write(f"归一化互信息 (NMI): {nmi_score:.4f}\n")
        f.write(f"轮廓系数 (Silhouette Coefficient): {sc_score:.4f}\n")
        f.write(f"Davies-Bouldin指数: {db_score:.4f}\n")
    print(f"聚类评估指标已追加保存至: {metrics_file}")
else:
    print("警告: 未找到聚类结果列 'DeepST_refine_domain'")

# endregion

# region 计算Wilcoxon秩和检验
# 添加到您的代码中，可以放在聚类评估指标分析部分之后
print("\n===== 执行Wilcoxon秩和检验分析 =====")
from scipy.stats import wilcoxon, ranksums

# 1. 对不同空间域之间的基因表达差异进行检验
if 'DeepST_refine_domain' in adata.obs.columns:
    domains = adata.obs['DeepST_refine_domain'].unique()
    print(f"分析{len(domains)}个空间域之间的基因表达差异...")

    # 获取原始表达矩阵
    if 'counts' in adata.layers:
        count_matrix = adata.layers['counts']
    else:
        count_matrix = adata.X

    # 将稀疏矩阵转换为密集矩阵
    from scipy.sparse import issparse

    if issparse(count_matrix):
        count_matrix = count_matrix.toarray()

    # 创建存储结果的数据结构
    import pandas as pd

    wilcoxon_results = {}

    # 选择要分析的基因数量（可以是前N个高变异基因或全部基因）
    n_genes = min(500, adata.n_vars)  # 选择前500个基因或全部基因
    selected_genes = adata.var_names[:n_genes]

    # 为每对空间域计算Wilcoxon检验
    for i, domain1 in enumerate(domains):
        for j, domain2 in enumerate(domains):
            if j <= i:  # 避免重复计算
                continue

            print(f"比较域{domain1}和域{domain2}...")
            # 获取两个域的细胞/点的索引
            idx1 = adata.obs['DeepST_refine_domain'] == domain1
            idx2 = adata.obs['DeepST_refine_domain'] == domain2

            # 统计显著差异表达基因的数量
            significant_genes = 0
            p_values = []

            # 对每个基因进行检验
            for g, gene in enumerate(selected_genes):
                if g % 100 == 0:
                    print(f"  处理第{g}/{len(selected_genes)}个基因...")

                gene_idx = adata.var_names.get_loc(gene)
                expr1 = count_matrix[idx1, gene_idx]
                expr2 = count_matrix[idx2, gene_idx]

                # 执行Wilcoxon秩和检验（两个独立样本）
                try:
                    # 使用ranksums而不是wilcoxon，因为这里是两个独立样本
                    stat, p_value = ranksums(expr1, expr2)
                    p_values.append(p_value)
                    if p_value < 0.05:  # 设置显著性阈值
                        significant_genes += 1
                except Exception as e:
                    print(f"  检验基因{gene}时出错: {str(e)}")

            # 存储结果
            key = f"{domain1}_vs_{domain2}"
            wilcoxon_results[key] = {
                "significant_genes": significant_genes,
                "total_genes": len(selected_genes),
                "percentage": significant_genes / len(selected_genes) * 100,
                "median_p_value": np.median(p_values)
            }

    # 将结果保存到文件
    results_df = pd.DataFrame(wilcoxon_results).T
    results_df.index.name = "Domain_Comparison"
    wilcoxon_file = os.path.join(save_path, f'{data_name}_wilcoxon_results.csv')
    results_df.to_csv(wilcoxon_file)
    print(f"Wilcoxon检验结果已保存至: {wilcoxon_file}")

    # 可视化差异最显著的前几对空间域
    sorted_comparisons = sorted(wilcoxon_results.items(),
                                key=lambda x: x[1]['percentage'],
                                reverse=True)

    plt.figure(figsize=(12, 10))
    top_n = min(10, len(sorted_comparisons))
    comparisons = [comp[0] for comp in sorted_comparisons[:top_n]]
    percentages = [comp[1]['percentage'] for comp in sorted_comparisons[:top_n]]

    plt.bar(comparisons, percentages)  # 改为垂直柱状图
    plt.ylim(0, 50)  # 设置y轴最大值为50
    plt.ylabel('差异表达基因百分比 (%)')  # 对调x轴y轴标签
    plt.xlabel('空间域比较')
    plt.xticks(rotation=45)  # 旋转x轴标签，使文本更易读
    plt.title('Wilcoxon检验: 空间域间差异表达基因比例')
    plt.tight_layout()  # 确保标签不会被截断
    plt.savefig(os.path.join(save_path, f'{data_name}_wilcoxon_comparison.pdf'), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(save_path, f'{data_name}_wilcoxon_comparison.png'), bbox_inches='tight', dpi=300)
    print(f"Wilcoxon检验可视化结果已保存")

    # 2. 可选：计算域内基因表达与全局表达的差异
    print("\n分析每个空间域与全局基因表达的差异...")
    global_vs_domain = {}

    for domain in domains:
        print(f"分析域{domain}...")
        # 获取该域的细胞/点的索引
        idx = adata.obs['DeepST_refine_domain'] == domain

        significant_genes = 0
        p_values = []

        # 对每个基因进行检验
        for g, gene in enumerate(selected_genes):
            if g % 100 == 0:
                print(f"  处理第{g}/{len(selected_genes)}个基因...")

            gene_idx = adata.var_names.get_loc(gene)
            domain_expr = count_matrix[idx, gene_idx]
            global_expr = count_matrix[:, gene_idx]

            # 从全局表达中移除当前域的表达
            other_expr = global_expr[~idx]

            # 执行Wilcoxon秩和检验
            try:
                stat, p_value = ranksums(domain_expr, other_expr)
                p_values.append(p_value)
                if p_value < 0.05:  # 设置显著性阈值
                    significant_genes += 1
            except Exception as e:
                print(f"  检验基因{gene}时出错: {str(e)}")

        # 存储结果
        global_vs_domain[domain] = {
            "significant_genes": significant_genes,
            "total_genes": len(selected_genes),
            "percentage": significant_genes / len(selected_genes) * 100,
            "median_p_value": np.median(p_values)
        }

    # 将结果保存到文件
    global_df = pd.DataFrame(global_vs_domain).T
    global_df.index.name = "Domain"
    global_wilcoxon_file = os.path.join(save_path, f'{data_name}_global_wilcoxon_results.csv')
    global_df.to_csv(global_wilcoxon_file)
    print(f"全局vs域的Wilcoxon检验结果已保存至: {global_wilcoxon_file}")

    # 可视化每个域与全局表达的差异
    plt.figure(figsize=(10, 8))
    domains_list = list(global_vs_domain.keys())
    percentages = [global_vs_domain[d]['percentage'] for d in domains_list]

    plt.bar(domains_list, percentages)
    plt.ylim(0, 50)  # 将y轴范围设置为0-50
    plt.xlabel('空间域')
    plt.ylabel('差异表达基因百分比 (%)')
    plt.title('Wilcoxon检验: 每个空间域与全局表达的差异')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{data_name}_global_wilcoxon.pdf'), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(save_path, f'{data_name}_global_wilcoxon.png'), bbox_inches='tight', dpi=300)
    print(f"全局vs域的Wilcoxon检验可视化结果已保存")
else:
    print("警告: 未找到聚类结果列 'DeepST_refine_domain'，无法执行Wilcoxon检验")
# endregion

# region PAGA
# 将分组变量转换为分类数据类型
print("\n===== 将分组变量转换为分类数据类型 =====")
if 'ground_truth' in adata.obs.columns:
    # 将ground_truth转换为分类数据类型
    adata.obs['ground_truth'] = adata.obs['ground_truth'].astype('category')
    print("已将ground_truth转换为分类数据类型")

# 确保DeepST_refine_domain也是分类数据类型
if 'DeepST_refine_domain' in adata.obs.columns:
    adata.obs['DeepST_refine_domain'] = adata.obs['DeepST_refine_domain'].astype('category')
    print("已将DeepST_refine_domain转换为分类数据类型")


# 如果有真实标签，也可以基于真实标签生成PAGA图进行比较
if 'ground_truth' in adata.obs.columns:
    print("\n===== 基于真实标签生成PAGA图 =====")
    try:
        # 确保ground_truth是分类类型
        adata.obs['ground_truth'] = adata.obs['ground_truth'].astype('category')
        print(f"真实标签类别: {adata.obs['ground_truth'].cat.categories.tolist()}")
        
        # 运行基于真实标签的PAGA
        sc.tl.paga(adata, groups='ground_truth')

        # 绘制基于真实标签的PAGA图
        plt.figure(figsize=(10, 8))
        sc.pl.paga(adata, color='ground_truth', show=False)
        plt.savefig(os.path.join(save_path, f'{data_name}_truth_paga.pdf'), bbox_inches='tight', dpi=300)
        print(f"基于真实标签的PAGA图已保存至: {os.path.join(save_path, f'{data_name}_truth_paga.pdf')}")
    except Exception as e:
        print(f"生成基于真实标签的PAGA图时出错: {str(e)}")
        print("跳过真实标签PAGA图，继续后续分析...")

# 生成PAGA图 - 基于聚类结果的抽象图表示
print("\n===== 生成PAGA图 =====")
try:
    # 确保已经计算了邻居图
    if 'neighbors' not in adata.uns:
        print("计算邻居图...")
        sc.pp.neighbors(adata, use_rep='DeepST_embed')

    # 确保DeepST_refine_domain是分类类型
    adata.obs['DeepST_refine_domain'] = adata.obs['DeepST_refine_domain'].astype('category')
    print(f"聚类结果类别: {adata.obs['DeepST_refine_domain'].cat.categories.tolist()}")

    # 运行PAGA算法
    print("运行PAGA算法...")
    sc.tl.paga(adata, groups='DeepST_refine_domain')

    # 绘制PAGA图
    plt.figure(figsize=(10, 8))
    sc.pl.paga(adata, color='DeepST_refine_domain', show=False)
    plt.savefig(os.path.join(save_path, f'{data_name}_paga.pdf'), bbox_inches='tight', dpi=300)
    print(f"PAGA图已保存至: {os.path.join(save_path, f'{data_name}_paga.pdf')}")

    # 绘制PAGA图与空间位置的叠加图
    plt.figure(figsize=(12, 10))
    sc.pl.paga_compare(
        adata,
        basis='spatial',
        edges=True,
        color='DeepST_refine_domain',
        save=f'{data_name}_paga_spatial.pdf',
        show=False
    )
    print(f"PAGA空间叠加图已保存至: {os.path.join(save_path, f'{data_name}_paga_spatial.pdf')}")
except Exception as e:
    print(f"生成PAGA图时出错: {str(e)}")
    print("跳过PAGA图生成，继续后续分析...")
# endregion

# region UMAP可视化与空间轨迹

print("\n===== 对模型生成的潜在表示进行UMAP可视化 =====")
import umap
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import ConvexHull

# 确保DeepST_embed存在
if 'DeepST_embed' in adata.obsm:
    # 获取潜在表示
    latent_rep = adata.obsm['DeepST_embed']
    print(f"潜在表示形状: {latent_rep.shape}")

    # 使用UMAP进行降维
    print("执行UMAP降维...")
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric='euclidean',
        random_state=42
    )
    umap_embedding = reducer.fit_transform(latent_rep)

    # 将UMAP嵌入保存到adata对象中
    adata.obsm['X_umap_deepst'] = umap_embedding
    print(f"UMAP嵌入形状: {umap_embedding.shape}")

    # 创建基本的UMAP可视化
    plt.figure(figsize=(12, 10))

    # 如果存在聚类结果，则使用聚类标签着色
    if 'DeepST_refine_domain' in adata.obs.columns:
        # 获取聚类标签
        cluster_labels = adata.obs['DeepST_refine_domain'].astype('category').cat.codes.values

        # 创建散点图，按聚类标签着色
        scatter = plt.scatter(
            umap_embedding[:, 0],
            umap_embedding[:, 1],
            c=cluster_labels,
            cmap='tab10',
            s=50,
            alpha=0.7
        )

        # 添加图例
        unique_labels = np.unique(cluster_labels)
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor=plt.cm.tab10(i / 10),
                                      markersize=10, label=f'域 {i}')
                           for i in unique_labels]
        plt.legend(handles=legend_elements, loc='best', title='空间域')

        # 为每个聚类绘制凸包，显示域的边界
        for i in unique_labels:
            # 获取当前聚类的点
            points = umap_embedding[cluster_labels == i]
            if len(points) < 3:  # 凸包需要至少3个点
                continue

            # 计算凸包
            try:
                hull = ConvexHull(points)
                # 绘制凸包边界
                for simplex in hull.simplices:
                    plt.plot(points[simplex, 0], points[simplex, 1], 'k-', alpha=0.3)
            except Exception as e:
                print(f"为域 {i} 计算凸包时出错: {str(e)}")
    else:
        # 如果没有聚类结果，则使用单一颜色
        plt.scatter(
            umap_embedding[:, 0],
            umap_embedding[:, 1],
            c='blue',
            s=50,
            alpha=0.7
        )

    plt.title('STGFusion潜在表示的UMAP可视化')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{data_name}_umap_latent.pdf'), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(save_path, f'{data_name}_umap_latent.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print(f"UMAP可视化已保存至: {os.path.join(save_path, f'{data_name}_umap_latent.pdf')}")

    # 在UMAP空间中呈现空间轨迹
    print("\n===== 在UMAP空间中呈现空间轨迹 =====")

    # 获取空间坐标
    spatial_coords = adata.obsm['spatial']

    # 创建一个新的图形
    plt.figure(figsize=(14, 12))

    # 创建一个自定义的颜色映射，用于表示空间位置
    # 使用x坐标作为颜色的基础
    x_min, x_max = np.min(spatial_coords[:, 0]), np.max(spatial_coords[:, 0])
    y_min, y_max = np.min(spatial_coords[:, 1]), np.max(spatial_coords[:, 1])

    # 归一化空间坐标用于颜色映射
    norm_x = (spatial_coords[:, 0] - x_min) / (x_max - x_min)
    norm_y = (spatial_coords[:, 1] - y_min) / (y_max - y_min)

    # 创建一个2D颜色映射
    # 使用红色表示x轴，蓝色表示y轴
    colors = np.column_stack([norm_x, np.zeros_like(norm_x), norm_y])

    # 绘制UMAP嵌入，使用空间颜色
    plt.scatter(
        umap_embedding[:, 0],
        umap_embedding[:, 1],
        c=colors,
        s=50,
        alpha=0.7
    )

    # 添加空间轨迹
    # 我们将连接空间上相邻的点
    print("添加空间轨迹...")

    # 使用KNN找到空间上的邻居
    from sklearn.neighbors import NearestNeighbors

    n_neighbors = 5  # 每个点连接到的邻居数量

    # 在空间坐标上找到邻居
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(spatial_coords)
    distances, indices = nbrs.kneighbors(spatial_coords)

    # 绘制连接线
    for i in range(len(spatial_coords)):
        for j in indices[i][1:]:  # 跳过第一个邻居（自身）
            plt.plot(
                [umap_embedding[i, 0], umap_embedding[j, 0]],
                [umap_embedding[i, 1], umap_embedding[j, 1]],
                'k-',
                alpha=0.1,
                linewidth=0.5
            )

    # 添加颜色条，显示空间位置
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax1 = divider.append_axes("right", size="5%", pad=0.1)
    cax2 = divider.append_axes("right", size="5%", pad=0.3)

    # 创建颜色条
    cmap_x = LinearSegmentedColormap.from_list('red_gradient', ['white', 'red'])
    cmap_y = LinearSegmentedColormap.from_list('blue_gradient', ['white', 'blue'])

    plt.colorbar(plt.cm.ScalarMappable(cmap=cmap_x), cax=cax1, label='X坐标')
    plt.colorbar(plt.cm.ScalarMappable(cmap=cmap_y), cax=cax2, label='Y坐标')

    plt.title('STGFusion潜在表示的UMAP可视化与空间轨迹')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{data_name}_umap_spatial_trajectory.pdf'), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(save_path, f'{data_name}_umap_spatial_trajectory.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print(f"UMAP空间轨迹可视化已保存至: {os.path.join(save_path, f'{data_name}_umap_spatial_trajectory.pdf')}")

    # 创建一个交互式的可视化，同时显示UMAP和空间位置
    print("\n===== 创建UMAP与空间位置的对比可视化 =====")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # 在左侧显示UMAP嵌入
    if 'DeepST_refine_domain' in adata.obs.columns:
        scatter1 = ax1.scatter(
            umap_embedding[:, 0],
            umap_embedding[:, 1],
            c=cluster_labels,
            cmap='tab10',
            s=50,
            alpha=0.7
        )

        # 添加图例
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor=plt.cm.tab10(i / 10),
                                      markersize=10, label=f'域 {i}')
                           for i in unique_labels]
        ax1.legend(handles=legend_elements, loc='best', title='空间域')
    else:
        scatter1 = ax1.scatter(
            umap_embedding[:, 0],
            umap_embedding[:, 1],
            c=colors,
            s=50,
            alpha=0.7
        )

    ax1.set_title('STGFusion潜在表示的UMAP可视化')
    ax1.set_xlabel('UMAP 1')
    ax1.set_ylabel('UMAP 2')

    # 在右侧显示空间位置
    if 'DeepST_refine_domain' in adata.obs.columns:
        scatter2 = ax2.scatter(
            spatial_coords[:, 0],
            spatial_coords[:, 1],
            c=cluster_labels,
            cmap='tab10',
            s=50,
            alpha=0.7
        )
    else:
        scatter2 = ax2.scatter(
            spatial_coords[:, 0],
            spatial_coords[:, 1],
            c=colors,
            s=50,
            alpha=0.7
        )

    ax2.set_title('空间位置')
    ax2.set_xlabel('X坐标')
    ax2.set_ylabel('Y坐标')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{data_name}_umap_vs_spatial.pdf'), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(save_path, f'{data_name}_umap_vs_spatial.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print(f"UMAP与空间位置对比可视化已保存至: {os.path.join(save_path, f'{data_name}_umap_vs_spatial.pdf')}")

    # 保存处理后的adata对象
    # adata_file = os.path.join(save_path, f'{data_name}_with_umap.h5ad')
    # adata.write(adata_file)
    # print(f"带有UMAP嵌入的adata对象已保存至: {adata_file}")

else:
    print("警告: 未找到DeepST_embed，无法进行UMAP可视化")

# endregion

# region 使用Moran's I检测空间可变基因
"""
print("\n===== 使用Moran's I检测空间可变基因 =====")
try:
    # 导入Moran's I计算模块
    from calculate_moran_I import Moran_I, Moran_I_Progress
    import numpy as np
    import pandas as pd
    
    # 准备数据
    print("准备数据...")
    # 获取原始表达矩阵和空间坐标
    if 'counts' in adata.layers:
        count_matrix = adata.layers['counts']
    else:
        count_matrix = adata.X
    
    # 获取空间坐标
    spatial_coords = adata.obsm['spatial']
    x_array = spatial_coords[:, 0]
    y_array = spatial_coords[:, 1]
    
    # 将表达矩阵转换为pandas DataFrame
    from scipy.sparse import issparse
    if issparse(count_matrix):
        count_matrix = count_matrix.toarray()
    
    # 创建基因表达DataFrame
    genes_exp = pd.DataFrame(count_matrix, columns=adata.var_names)
    
    # 计算每个基因的Moran's I统计量
    print("计算基因的空间自相关性...")
    # moran_scores = Moran_I(genes_exp, x_array, y_array, k=10, knn=True)
    moran_scores = Moran_I_Progress(genes_exp, x_array, y_array, k=10, knn=True) # 在原计算方法中增加进度条

    # 将结果保存到adata对象中
    adata.var['moran_score'] = 0
    for gene in moran_scores.index:
        if gene in adata.var_names:
            adata.var.loc[gene, 'moran_score'] = moran_scores[gene]
    
    # 标记前500个SVG
    adata.var['is_svg'] = False
    top_genes = adata.var.sort_values('moran_score', ascending=False).index[:500]
    adata.var.loc[top_genes, 'is_svg'] = True
    
    # 保存SVG列表
    svg_file = os.path.join(save_path, f'{data_name}_SVGs.csv')
    adata.var[adata.var['is_svg']].to_csv(svg_file)
    print(f"已检测到{len(top_genes)}个空间可变基因，结果已保存至: {svg_file}")
    
    # 可视化前10个SVG的空间表达模式
    print("可视化前10个SVG的空间表达模式...")
    top_10_genes = top_genes[:10]
    print(f"前10个空间可变基因: {top_10_genes.tolist()}")  # 打印前10个基因名称
    
    
    # 检查这些基因是否在数据集中
    missing_genes = [gene for gene in top_10_genes if gene not in adata.var_names]
    if missing_genes:
        print(f"警告: 以下基因在数据集中不存在: {missing_genes}")
    
    # 检查基因表达值
    for gene in top_10_genes:
        if gene in adata.var_names:
            expr = adata[:, gene].X
            from scipy.sparse import issparse
            if issparse(expr):
                expr = expr.toarray()
            print(f"基因 {gene} 的表达统计: 最小值={expr.min():.4f}, 最大值={expr.max():.4f}, 平均值={expr.mean():.4f}")

    # 统计各个域中高表达的SVG数量
    if 'DeepST_refine_domain' in adata.obs.columns and len(top_genes) > 0:
        print("\n===== 各个域中高表达的SVG统计 =====")
        domains = adata.obs['DeepST_refine_domain'].unique()
        domain_svg_counts = {}
        domain_specific_svgs = {}
        
        # 为每个域创建一个空列表，用于存储特异性表达的SVG
        for domain in domains:
            domain_specific_svgs[domain] = []
        
        # 设置表达阈值，可以根据需要调整
        expression_threshold = np.percentile(count_matrix, 75)  # 使用75%分位数作为高表达阈值
        print(f"高表达阈值设置为: {expression_threshold:.4f}")
        
        # 对每个SVG，检查其在各个域中的表达情况
        for gene in top_genes:
            if gene in adata.var_names:
                gene_idx = adata.var_names.get_loc(gene)
                
                # 计算每个域中该基因的平均表达量
                for domain in domains:
                    domain_cells = adata.obs['DeepST_refine_domain'] == domain
                    domain_expr = count_matrix[domain_cells, gene_idx]
                    mean_expr = np.mean(domain_expr)
                    
                    # 如果平均表达量超过阈值，则认为该基因在该域中高表达
                    if mean_expr > expression_threshold:
                        domain_specific_svgs[domain].append(gene)
        
        # 统计每个域中高表达的SVG数量
        for domain in domains:
            domain_svg_counts[domain] = len(domain_specific_svgs[domain])
            print(f"域 {domain} 中高表达的SVG数量: {domain_svg_counts[domain]}")
            if domain_svg_counts[domain] > 0:
                print(f"  前5个高表达SVG: {domain_specific_svgs[domain][:5]}")
        
        # 将结果保存到文件
        domain_svg_file = os.path.join(save_path, f'{data_name}_domain_SVGs.csv')
        with open(domain_svg_file, 'w') as f:
            f.write("Domain,SVG_Count,Top_SVGs\n")
            for domain in domains:
                top_svgs = ','.join(domain_specific_svgs[domain][:5]) if domain_svg_counts[domain] > 0 else "None"
                f.write(f"{domain},{domain_svg_counts[domain]},{top_svgs}\n")
        print(f"各域SVG统计结果已保存至: {domain_svg_file}")
        
        # 可视化各域SVG数量
        plt.figure(figsize=(10, 6))
        domains_list = [str(d) for d in domains]
        counts_list = [domain_svg_counts[d] for d in domains]
        plt.bar(domains_list, counts_list)
        plt.xlabel('空间域')
        plt.ylabel('高表达SVG数量')
        plt.title('各空间域中高表达的SVG数量')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'{data_name}_domain_SVG_counts.pdf'), bbox_inches='tight', dpi=300)
        plt.savefig(os.path.join(save_path, f'{data_name}_domain_SVG_counts.png'), bbox_inches='tight', dpi=300)
        plt.close()
        print(f"各域SVG数量统计图已保存至: {os.path.join(save_path, f'{data_name}_domain_SVG_counts.pdf')}")

    top_10_genes = top_genes[:10]
    plt.figure(figsize=(20, 15))
    for i, gene in enumerate(top_10_genes):
        if i >= 10:
            break
        if gene in adata.var_names:
            plt.subplot(2, 5, i+1)
            try:
                sc.pl.spatial(adata, color=gene, show=False, title=f"Gene: {gene}")
                print(f"成功绘制基因 {gene} 的空间表达图")
            except Exception as e:
                print(f"绘制基因 {gene} 的空间表达图时出错: {str(e)}")
        else:
            print(f"警告: 基因 {gene} 不在数据集中，跳过绘图")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{data_name}_top10_SVGs.pdf'), bbox_inches='tight', dpi=300)
    print(f"前10个SVG的空间表达模式可视化已保存至: {os.path.join(save_path, f'{data_name}_top10_SVGs.pdf')}")
    
    # 尝试单独绘制每个基因的图
    print("尝试单独绘制每个基因的空间表达图...")
    for i, gene in enumerate(top_10_genes):
        if gene in adata.var_names:
            try:
                plt.figure(figsize=(8, 6))
                sc.pl.spatial(adata, color=gene, show=False, title=f"Gene: {gene}")
                plt.tight_layout()
                plt.savefig(os.path.join(save_path, f'{data_name}_gene_{gene}.pdf'), bbox_inches='tight', dpi=300)
                print(f"成功单独绘制基因 {gene} 的空间表达图")
            except Exception as e:
                print(f"单独绘制基因 {gene} 的空间表达图时出错: {str(e)}")
    
    # 为每个空间域构建元基因
    print("\n===== 为每个空间域构建元基因 =====")
    from util import find_meta_gene
    
    # 确保DeepST_refine_domain存在
    if 'DeepST_refine_domain' in adata.obs.columns:
        # 获取所有域的标签
        domains = adata.obs['DeepST_refine_domain'].unique()
        print(f"检测到 {len(domains)} 个空间域")
        
        # 为每个域构建元基因
        meta_genes = {}
        meta_gene_expressions = {}
        
        for domain in domains:
            print(f"\n构建域 {domain} 的元基因...")
            # 选择该域中表达最高的空间可变基因作为起始基因
            domain_cells = adata.obs['DeepST_refine_domain'] == domain
            domain_adata = adata[domain_cells]
            
            # 从前20个SVG中选择在该域中表达最高的基因作为起始基因
            start_gene_candidates = top_genes[:20]
            start_gene = None
            max_expr = -1
            
            for gene in start_gene_candidates:
                if gene in adata.var_names:
                    mean_expr = np.mean(adata[domain_cells, gene].X)
                    # 检查mean_expr是否为数组，如果是则获取第一个元素
                    if hasattr(mean_expr, 'shape') and mean_expr.shape != ():
                        if isinstance(mean_expr, np.ndarray):
                            mean_expr = mean_expr[0]
                    # 确保mean_expr是标量
                    mean_expr = float(mean_expr)
                    if mean_expr > max_expr:
                        max_expr = mean_expr
                        start_gene = gene
            
            if start_gene is None:
                print(f"警告: 无法为域 {domain} 找到合适的起始基因")
                continue
                
            print(f"选择 {start_gene} 作为域 {domain} 的起始基因")
            
            # 使用find_meta_gene函数构建元基因
            try:
                meta_name, meta_expr = find_meta_gene(
                    input_adata=adata,
                    pred=adata.obs['DeepST_refine_domain'].values,
                    target_domain=domain,
                    start_gene=start_gene,
                    mean_diff=0,
                    early_stop=True,
                    max_iter=5
                )
                
                meta_genes[domain] = meta_name
                meta_gene_expressions[domain] = meta_expr
                
                # 将元基因表达添加到adata对象中
                adata.obs[f'meta_gene_{domain}'] = meta_expr
                
                print(f"成功为域 {domain} 构建元基因: {meta_name}")
                
                # 可视化元基因表达
                plt.figure(figsize=(10, 8))
                sc.pl.spatial(adata, color=f'meta_gene_{domain}', title=f"Meta Gene for Domain {domain}", show=False)
                plt.tight_layout()
                plt.savefig(os.path.join(save_path, f'{data_name}_meta_gene_domain_{domain}.pdf'), bbox_inches='tight', dpi=300)
                print(f"域 {domain} 的元基因表达图已保存")
                
            except Exception as e:
                print(f"为域 {domain} 构建元基因时出错: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # 保存所有元基因信息
        meta_gene_info = pd.DataFrame({
            'domain': list(meta_genes.keys()),
            'meta_gene': list(meta_genes.values())
        })
        meta_gene_file = os.path.join(save_path, f'{data_name}_meta_genes.csv')
        meta_gene_info.to_csv(meta_gene_file, index=False)
        print(f"元基因信息已保存至: {meta_gene_file}")
        
        # 创建一个包含所有域元基因表达的图
        if len(meta_genes) > 0:
            plt.figure(figsize=(15, 10))
            for i, domain in enumerate(meta_genes.keys()):
                if i >= 9:  # 最多显示9个域
                    break
                plt.subplot(3, 3, i+1)
                sc.pl.spatial(adata, color=f'meta_gene_{domain}', 
                             title=f"Domain {domain}", show=False)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f'{data_name}_all_meta_genes.pdf'), bbox_inches='tight', dpi=300)
            print(f"所有域的元基因表达图已保存至: {os.path.join(save_path, f'{data_name}_all_meta_genes.pdf')}")
    else:
        print("警告: 未找到聚类结果列 'DeepST_refine_domain'，无法构建特定域的元基因")

except Exception as e:
    print(f"使用Moran's I检测SVG时出错: {str(e)}")
    import traceback
    traceback.print_exc()
"""
# endregion

# region 使用改进的SpaGCN域特异性方法检测空间可变基因(SVG)

print("\n===== 使用改进的域特异性方法检测空间可变基因 =====")
try:
    from scipy.stats import ranksums
    from scipy.spatial import distance_matrix
    import pandas as pd
    import numpy as np
    from scipy.sparse import issparse
    from statsmodels.stats.multitest import multipletests
    
    # 检查是否有聚类结果
    if 'DeepST_refine_domain' not in adata.obs.columns:
        print("警告: 未找到聚类结果列 'DeepST_refine_domain'，无法进行域特异性SVG检测")
    else:
        # 准备数据
        print("准备数据...")
        if 'counts' in adata.layers:
            count_matrix = adata.layers['counts']
        else:
            count_matrix = adata.X
        
        if issparse(count_matrix):
            count_matrix = count_matrix.toarray()
        
        spatial_coords = adata.obsm['spatial']
        domains = adata.obs['DeepST_refine_domain'].unique()
        
        # 参数设置
        SEARCH_RADIUS = np.percentile(distance_matrix(spatial_coords, spatial_coords), 10)  # 使用10%分位数作为搜索半径
        NEIGHBOR_THRESHOLD = 0.5  # 50%的点是邻居
        P_VALUE_THRESHOLD = 0.05
        IN_DOMAIN_FRACTION_THRESHOLD = 0.8  # 域内分数 > 80%
        IN_OUT_RATIO_THRESHOLD = 1.0  # 输入/输出分数比 > 1
        IN_OUT_EXPR_RATIO_THRESHOLD = 1.5  # 输入/输出表达比 > 1.5
        
        print(f"搜索半径: {SEARCH_RADIUS:.2f}")
        print(f"检测到 {len(domains)} 个空间域")
        
        # 为每个域检测SVG
        all_domain_svgs = {}
        all_svg_info = []
        
        for target_domain in domains:
            print(f"\n===== 处理目标域: {target_domain} =====")
            
            # 获取目标域的点
            target_indices = np.where(adata.obs['DeepST_refine_domain'] == target_domain)[0]
            print(f"目标域包含 {len(target_indices)} 个点")
            
            if len(target_indices) < 3:
                print(f"警告: 域 {target_domain} 点数太少，跳过")
                continue
            
            # 步骤1: 确定邻近域
            print("步骤1: 确定邻近域...")
            neighbor_domain_counts = {d: 0 for d in domains if d != target_domain}
            
            for target_idx in target_indices:
                target_coord = spatial_coords[target_idx]
                # 计算到所有点的距离
                distances = np.sqrt(np.sum((spatial_coords - target_coord)**2, axis=1))
                # 找到半径内的邻居
                neighbors_in_radius = np.where(distances <= SEARCH_RADIUS)[0]
                
                # 统计每个域的邻居数量
                for neighbor_idx in neighbors_in_radius:
                    if neighbor_idx not in target_indices:
                        neighbor_domain = adata.obs['DeepST_refine_domain'].iloc[neighbor_idx]
                        if neighbor_domain in neighbor_domain_counts:
                            neighbor_domain_counts[neighbor_domain] += 1
            
            # 确定邻近域（超过50%的点是邻居）
            neighbor_domains = []
            for domain, count in neighbor_domain_counts.items():
                domain_indices = np.where(adata.obs['DeepST_refine_domain'] == domain)[0]
                if len(domain_indices) > 0:
                    fraction = count / len(domain_indices)
                    if fraction > NEIGHBOR_THRESHOLD:
                        neighbor_domains.append(domain)

            if len(neighbor_domains) == 0:
                print(f"警告: 域 {target_domain} 没有邻近域，跳过")
                continue
            
            # 获取邻近域的所有点
            neighbor_indices = np.concatenate([
                np.where(adata.obs['DeepST_refine_domain'] == d)[0] 
                for d in neighbor_domains
            ])
            print(f"邻近域总共包含 {len(neighbor_indices)} 个点")
            
            # 步骤2: Wilcoxon秩和检验
            print("步骤2: 执行Wilcoxon秩和检验...")
            gene_results = []
            
            for gene_idx, gene in enumerate(adata.var_names):
                if gene_idx % 500 == 0:
                    print(f"  处理第 {gene_idx}/{len(adata.var_names)} 个基因...")
                
                # 获取目标域和邻近域的表达值
                target_expr = count_matrix[target_indices, gene_idx]
                neighbor_expr = count_matrix[neighbor_indices, gene_idx]
                
                # 执行Wilcoxon秩和检验
                try:
                    stat, p_value = ranksums(target_expr, neighbor_expr)
                except:
                    p_value = 1.0
                
                # 计算其他指标
                # 1. 域内分数 (in-domain fraction): 目标域中表达该基因的点的比例
                in_domain_expr_cells = np.sum(target_expr > 0)
                in_domain_fraction = in_domain_expr_cells / len(target_indices)
                
                # 2. 输入/输出分数比 (in/out ratio): 目标域vs邻近域表达细胞数比例
                out_domain_expr_cells = np.sum(neighbor_expr > 0)
                in_out_ratio = (in_domain_expr_cells / len(target_indices)) / (out_domain_expr_cells / len(neighbor_indices) + 1e-10)
                
                # 3. 输入/输出表达比 (in/out expression ratio): 平均表达量比例
                mean_target_expr = np.mean(target_expr)
                mean_neighbor_expr = np.mean(neighbor_expr)
                in_out_expr_ratio = mean_target_expr / (mean_neighbor_expr + 1e-10)
                
                gene_results.append({
                    'gene': gene,
                    'p_value': p_value,
                    'in_domain_fraction': in_domain_fraction,
                    'in_out_ratio': in_out_ratio,
                    'in_out_expr_ratio': in_out_expr_ratio,
                    'mean_target_expr': mean_target_expr,
                    'mean_neighbor_expr': mean_neighbor_expr
                })
            
            # 转换为DataFrame
            results_df = pd.DataFrame(gene_results)
            
            # 步骤3: 多重检验校正
            print("步骤3: 多重检验校正...")
            rejected, pvals_corrected, _, _ = multipletests(
                results_df['p_value'].values, 
                alpha=P_VALUE_THRESHOLD, 
                method='fdr_bh'
            )
            results_df['adjusted_p_value'] = pvals_corrected
            results_df['significant'] = rejected
            
            # 步骤4: 应用额外标准
            print("步骤4: 应用额外标准筛选SVG...")
            svg_mask = (
                (results_df['adjusted_p_value'] < P_VALUE_THRESHOLD) &
                (results_df['in_domain_fraction'] > IN_DOMAIN_FRACTION_THRESHOLD) &
                (results_df['in_out_ratio'] > IN_OUT_RATIO_THRESHOLD) &
                (results_df['in_out_expr_ratio'] > IN_OUT_EXPR_RATIO_THRESHOLD)
            )
            
            domain_svgs = results_df[svg_mask].copy()
            domain_svgs = domain_svgs.sort_values('adjusted_p_value')
            domain_svgs['domain'] = target_domain
            
            print(f"域 {target_domain} 检测到 {len(domain_svgs)} 个SVG")
            
            # 保存结果
            all_domain_svgs[target_domain] = domain_svgs
            all_svg_info.append(domain_svgs)
            
            # 保存该域的SVG列表
            domain_svg_file = os.path.join(save_path, f'{data_name}_domain_{target_domain}_SVGs.csv')
            domain_svgs.to_csv(domain_svg_file, index=False)
            print(f"域 {target_domain} 的SVG已保存至: {domain_svg_file}")
            
            # 可视化前10个SVG
            if len(domain_svgs) > 0:
                top_genes = domain_svgs['gene'].iloc[:min(10, len(domain_svgs))].tolist()
                print(f"域 {target_domain} 前{len(top_genes)}个SVG: {top_genes}")
                
                # 为每个域的SVG创建可视化
                n_genes = len(top_genes)
                if n_genes > 0:
                    n_cols = min(5, n_genes)
                    n_rows = (n_genes + n_cols - 1) // n_cols
                    
                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
                    if n_rows == 1 and n_cols == 1:
                        axes = np.array([axes])
                    axes = axes.flatten() if n_genes > 1 else [axes]
                    
                    for i, gene in enumerate(top_genes):
                        try:
                            # spot_size = 150(其他) / 250(HBRC) / 50(stereo)
                            sc.pl.spatial(adata, color=gene, show=False, ax=axes[i], 
                                        title=f"{gene}\n(Domain {target_domain})", 
                                        spot_size=150)
                        except Exception as e:
                            print(f"绘制基因 {gene} 时出错: {str(e)}")
                    
                    # 隐藏多余的子图
                    for i in range(n_genes, len(axes)):
                        axes[i].axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_path, f'{data_name}_domain_{target_domain}_top_SVGs.pdf'), 
                              bbox_inches='tight', dpi=300)
                    plt.savefig(os.path.join(save_path, f'{data_name}_domain_{target_domain}_top_SVGs.png'), 
                              bbox_inches='tight', dpi=300)
                    plt.close()
                    print(f"域 {target_domain} 的SVG可视化已保存")
        
        # 保存所有域的SVG汇总
        if len(all_svg_info) > 0:
            all_svgs_df = pd.concat(all_svg_info, ignore_index=True)
            all_svgs_file = os.path.join(save_path, f'{data_name}_all_domains_SVGs.csv')
            all_svgs_df.to_csv(all_svgs_file, index=False)
            print(f"\n所有域的SVG汇总已保存至: {all_svgs_file}")
            
            # 统计每个域的SVG数量
            svg_counts = all_svgs_df.groupby('domain').size()
            print("\n各域SVG统计:")
            for domain, count in svg_counts.items():
                print(f"  域 {domain}: {count} 个SVG")
            
            # 可视化各域SVG数量
            plt.figure(figsize=(10, 6))
            plt.bar([str(d) for d in svg_counts.index], svg_counts.values)
            plt.xlabel('空间域')
            plt.ylabel('SVG数量')
            plt.title('各空间域检测到的SVG数量')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f'{data_name}_svg_counts_per_domain.pdf'), 
                      bbox_inches='tight', dpi=300)
            plt.savefig(os.path.join(save_path, f'{data_name}_svg_counts_per_domain.png'), 
                      bbox_inches='tight', dpi=300)
            plt.close()
            print(f"SVG数量统计图已保存")
            
            # 为每个域构建元基因
            print("\n===== 为每个域构建元基因 =====")
            
            # 元基因构建参数
            N_GENES_FOR_META = 5  # 使用前5个最显著的SVG构建元基因
            print(f"每个域将使用前 {N_GENES_FOR_META} 个最显著的SVG构建元基因")
            
            meta_gene_list = []  # 用于存储所有域的元基因信息
            
            for target_domain in domains:
                target_indices = np.where(adata.obs['DeepST_refine_domain'] == target_domain)[0]
                print(f"\n域 {target_domain} 包含 {len(target_indices)} 个点，开始构建元基因...")
                
                if target_domain in all_domain_svgs:
                    domain_svgs = all_domain_svgs[target_domain]
                    
                    if len(domain_svgs) > 0:
                        # 只选择前N个最显著的SVG构建元基因
                        n_genes_to_use = min(N_GENES_FOR_META, len(domain_svgs))
                        top_svg_genes = domain_svgs['gene'].iloc[:n_genes_to_use].tolist()
                        
                        print(f"  选择的基因: {', '.join(top_svg_genes)}")
                        
                        # 获取这些基因的表达矩阵
                        gene_indices = [adata.var_names.get_loc(g) for g in top_svg_genes if g in adata.var_names]
                        
                        if len(gene_indices) > 0:
                            svg_expr_matrix = count_matrix[:, gene_indices]
                            
                            # 计算元基因表达（取平均值）
                            meta_gene_expr = np.mean(svg_expr_matrix, axis=1)
                            
                            # 保存到adata
                            adata.obs[f'meta_gene_domain_{target_domain}'] = meta_gene_expr
                            
                            # 记录元基因信息
                            meta_gene_list.append({
                                'domain': target_domain,
                                'n_cells': len(target_indices),
                                'n_svgs_detected': len(domain_svgs),
                                'n_genes_used': len(gene_indices),
                                'genes_used': ','.join(top_svg_genes)  # 保存所有使用的基因名
                            })
                            
                            # 可视化元基因
                            plt.figure(figsize=(10, 8))
                            # spot_size = 150(其他) / 250(HBRC) / 50(stereo)
                            # 构建基因名称标注
                            genes_annotation = ' + '.join(top_svg_genes)
                            sc.pl.spatial(adata, color=f'meta_gene_domain_{target_domain}', 
                                        title=f"Meta Gene for Domain {target_domain}\n{genes_annotation}",
                                        show=False, spot_size=150)
                            plt.tight_layout()
                            plt.savefig(os.path.join(save_path, f'{data_name}_meta_gene_domain_{target_domain}.pdf'), 
                                      bbox_inches='tight', dpi=300)
                            plt.savefig(os.path.join(save_path, f'{data_name}_meta_gene_domain_{target_domain}.png'), 
                                      bbox_inches='tight', dpi=300)
                            plt.close()
                            print(f"  ✓ 域 {target_domain} 的元基因已构建并保存 (使用 {len(top_svg_genes)} 个SVG)")
                        else:
                            print(f"  ✗ 警告: 域 {target_domain} 没有有效的SVG基因，跳过")
                    else:
                        print(f"  ✗ 警告: 域 {target_domain} 未检测到SVG，跳过")
                else:
                    print(f"  ✗ 警告: 域 {target_domain} 在all_domain_svgs中不存在，跳过")
            
            # 保存元基因信息汇总
            if len(meta_gene_list) > 0:
                meta_gene_summary = pd.DataFrame(meta_gene_list)
                meta_gene_summary_file = os.path.join(save_path, f'{data_name}_meta_gene_summary.csv')
                meta_gene_summary.to_csv(meta_gene_summary_file, index=False)
                print(f"\n元基因汇总信息已保存至: {meta_gene_summary_file}")
                
                # 打印元基因构建统计
                print("\n===== 元基因构建统计 =====")
                for _, row in meta_gene_summary.iterrows():
                    print(f"域 {row['domain']}: {row['n_cells']} 个点, 检测到 {row['n_svgs_detected']} 个SVG, 使用 {row['n_genes_used']} 个基因")
                    print(f"  使用的基因: {row['genes_used']}")
                
                # 创建一个包含所有域元基因的综合可视化
                print("\n===== 创建所有域元基因的综合可视化 =====")
                n_domains_with_meta = len(meta_gene_list)
                if n_domains_with_meta > 0:
                    # 计算子图布局
                    n_cols = min(4, n_domains_with_meta)
                    n_rows = (n_domains_with_meta + n_cols - 1) // n_cols
                    
                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
                    if n_rows == 1 and n_cols == 1:
                        axes = np.array([axes])
                    axes = axes.flatten() if n_domains_with_meta > 1 else [axes]
                    
                    for i, meta_info in enumerate(meta_gene_list):
                        domain = meta_info['domain']
                        if f'meta_gene_domain_{domain}' in adata.obs.columns:
                            try:
                                # 构建基因名称标注
                                genes_used = meta_info['genes_used'].split(',')
                                genes_annotation = ' + '.join(genes_used)
                                # spot_size = 150(其他) / 250(HBRC) / 50(stereo)
                                sc.pl.spatial(adata, color=f'meta_gene_domain_{domain}', 
                                            show=False, ax=axes[i], 
                                            title=f"Domain {domain}\n{genes_annotation}",
                                            spot_size=150, frameon=False)
                            except Exception as e:
                                print(f"绘制域 {domain} 元基因时出错: {str(e)}")
                    
                    # 隐藏多余的子图
                    for i in range(n_domains_with_meta, len(axes)):
                        axes[i].axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_path, f'{data_name}_all_meta_genes.pdf'), 
                              bbox_inches='tight', dpi=300)
                    plt.savefig(os.path.join(save_path, f'{data_name}_all_meta_genes.png'), 
                              bbox_inches='tight', dpi=300)
                    plt.close()
                    print(f"所有域的元基因综合可视化已保存至: {os.path.join(save_path, f'{data_name}_all_meta_genes.pdf')}")

            else:
                print("警告: 没有成功构建任何域的元基因")
        else:
            print("警告: 未检测到任何SVG")

except Exception as e:
    print(f"使用域特异性方法检测SVG时出错: {str(e)}")
    import traceback
    traceback.print_exc()

# endregion
######################################################################################################################
# region 测试代码
"""domains = None
n_domains = None
Conv_type = "SGFormer"  # 这个参数为选择GNN模块，GCNConv、 GCN_Transformer_Conv 、 SGFormer 、 TransformerConv
							   #					  GCN模块、GCN+Transformer模块、SGFormer模块、Transformer模块
linear_encoder_hidden = [32,20]
linear_decoder_hidden = [32]
conv_hidden = [32,8]
p_drop = 0.01
dec_cluster_n = 20
kl_weight = 1
mse_weight = 1
bce_kld_weight = 1
domain_weight = 1

# 实例化模型
deepst_model = DeepST_model(
				input_dim = data.shape[1],
                # Conv_type = 'SGFormer',
                Conv_type = Conv_type,
                # Conv_type = 'SAGEConv',
				linear_encoder_hidden = linear_encoder_hidden,
				linear_decoder_hidden = linear_decoder_hidden,
				conv_hidden = conv_hidden,
				p_drop = p_drop,
				dec_cluster_n = dec_cluster_n,
				)

# 单个数据测试是否能跑通
model = deepst_model.to('cuda')
adj = graph_dict['adj_norm'].to('cuda')
dat = torch.tensor(data).to('cuda')
z, mu, logvar, de_feat, out_q, feat_x, gnn_z = model(dat, adj)

print(data.shape)
print(z.shape)
print(mu.shape)

# 训练
deepst_training = train(
					data,
					graph_dict,
					deepst_model,
					pre_epochs = 1,  # 修改迭代参数
					epochs = 1,  # 修改迭代参数
					kl_weight = kl_weight,
                			mse_weight = mse_weight,
                			bce_kld_weight = bce_kld_weight,
                			domain_weight = domain_weight,
                			use_gpu = True
                			)
deepst_training.fit()"""
# endregion