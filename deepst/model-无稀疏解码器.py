domains = None
n_domains = None
Conv_type = "GCNConv" 
linear_encoder_hidden = [32,20]
linear_decoder_hidden = [32]
conv_hidden = [32,8] 
p_drop = 0.01 
dec_cluster_n = 20 
kl_weight = 1
mse_weight = 1
bce_kld_weight = 1
domain_weight = 1

#!/usr/bin/env python
"""
# Author: ChangXu
# Created Time : Mon 23 Apr 2021 08:26:32 PM CST
# File Name: model.py
# Description:`
"""
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
                x=conv(x,edge_index,edge_weight)
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
        attention=torch.einsum("nhm,lhm->nlh", qs, ks).mean(dim=-1) #[N, N]
        normalizer=attention_normalizer.squeeze(dim=-1).mean(dim=-1,keepdims=True) #[N,1]
        attention=attention/normalizer


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
        self.use_act=use_act

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
                x = self.alpha * x + (1-self.alpha) * layer_[i]
            if self.use_bn:
                x = self.bns[i+1](x)
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
                 alpha=0.5, dropout=0.5, use_bn=True, use_residual=True, use_weight=True, use_graph=False, use_act=False, graph_weight=0.8, gnn=None, aggregate='add'):
        super().__init__()
        self.trans_conv=TransConv(in_channels,hidden_channels,num_layers,num_heads,alpha,dropout,use_bn,use_residual,use_weight)
        self.gnn=gnn
        self.use_graph=use_graph
        self.graph_weight=graph_weight
        self.use_act=use_act

        self.aggregate=aggregate

        if aggregate=='add':
            self.fc=nn.Linear(hidden_channels,out_channels)
        elif aggregate=='cat':
            self.fc=nn.Linear(2*hidden_channels,out_channels)
        else:
            raise ValueError(f'Invalid aggregate type:{aggregate}')
        
        self.params1=list(self.trans_conv.parameters())
        self.params2=list(self.gnn.parameters()) if self.gnn is not None else []
        self.params2.extend(list(self.fc.parameters()) )

    def forward(self,x, edge_index, edge_weight=None):
        x1=self.trans_conv(x, edge_index, edge_weight=None)
        if self.use_graph:
            x2=self.gnn(x, edge_index, edge_weight)
            # print('x1:', x1.shape)
            # print('x2:', x2.shape)
            if self.aggregate=='add':
                x=self.graph_weight*x2+(1-self.graph_weight)*x1
            else:
                x=torch.cat((x1,x2),dim=1)
        else:
            x=x1
        x=self.fc(x)
        return x
    
    def get_attentions(self, x):
        attns=self.trans_conv.get_attentions(x) # [layer num, N, N]
        return attns

    def reset_parameters(self):
        self.trans_conv.reset_parameters()
        if self.use_graph:
            self.gnn.reset_parameters()


class DeepST_model(nn.Module):
    def __init__(self, 
                input_dim, 
                Conv_type = 'GCNConv',
                linear_encoder_hidden = [32,20],
                linear_decoder_hidden = [32],
                conv_hidden = [32,8],
                p_drop = 0.01,
                dec_cluster_n = 15,
                alpha = 0.9,
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
                                    buildNetwork(current_encoder_dim, self.linear_encoder_hidden[le], self.activate, self.p_drop))
            current_encoder_dim = linear_encoder_hidden[le]
        current_decoder_dim = linear_encoder_hidden[-1] + conv_hidden[-1]

        self.decoder = nn.Sequential()
        for ld in range(len(linear_decoder_hidden)):
            self.decoder.add_module(f'decoder_L{ld}',
                                    buildNetwork(current_decoder_dim, self.linear_decoder_hidden[ld], self.activate, self.p_drop))
            current_decoder_dim= self.linear_decoder_hidden[ld]
        self.decoder.add_module(f'decoder_L{len(self.linear_decoder_hidden)}', nn.Linear(self.linear_decoder_hidden[-1], 
                                self.input_dim))

        #### a variational graph autoencoder based on pytorch geometric
        '''https://pytorch-geometric.readthedocs.io/en/latest/index.html'''

        # GCN layers
        if self.Conv_type == "GCNConv":
            '''https://arxiv.org/abs/1609.02907'''
            from torch_geometric.nn import GCNConv
            self.conv = Sequential('x, edge_index', [
                        (GCNConv(linear_encoder_hidden[-1], conv_hidden[0]* 2), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden[0]* 2),
                        nn.ReLU(inplace=True), 
                        ])
            self.conv_mean = Sequential('x, edge_index', [
                        (GCNConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ])
            self.conv_logvar = Sequential('x, edge_index', [
                        (GCNConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ])
        
        ############################################
        # 这是新增的GCN+Transformer
        ############################################
        if self.Conv_type == "GCN_Transformer_Conv":
            '''https://arxiv.org/abs/1609.02907'''
            from torch_geometric.nn import GCNConv
            from torch_geometric.nn import TransformerConv
            self.conv = Sequential('x, edge_index', [
                        (GCNConv(linear_encoder_hidden[-1], conv_hidden[0]* 2), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden[0]* 2),
                        nn.ReLU(inplace=True), 
                        (TransformerConv(linear_encoder_hidden[-1], conv_hidden[0]* 2), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden[0]* 2),
                        nn.ReLU(inplace=True), 
                        ])
            self.conv_mean = Sequential('x, edge_index', [
                        (GCNConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden[-1]),
                        nn.ReLU(inplace=True), 
                        (TransformerConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ])
            self.conv_logvar = Sequential('x, edge_index', [
                        (GCNConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden[-1]),
                        nn.ReLU(inplace=True), 
                        (TransformerConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ])
        
        ############################################
        # 这是新增的SGFormer
        ############################################
        elif self.Conv_type == 'SGFormer':
            gcn_1 = GCN(in_channels=linear_encoder_hidden[-1], 
                      hidden_channels=conv_hidden[0]* 2, 
                      out_channels=conv_hidden[0]* 2)
            self.conv = Sequential('x, edge_index', [
                (SGFormer(in_channels=linear_encoder_hidden[-1], 
                         hidden_channels=conv_hidden[0]* 2, 
                         out_channels=conv_hidden[0]* 2, 
                         gnn=gcn_1, 
                         use_graph=True), 'x, edge_index -> x1'), 
                BatchNorm(conv_hidden[0]* 2),
                nn.ReLU(inplace=True), 
            ])
            
            gcn_2 = GCN(in_channels=conv_hidden[0]* 2, 
                      hidden_channels=conv_hidden[0]* 2, 
                      out_channels=conv_hidden[-1])
            self.conv_mean = Sequential('x, edge_index', [
                        (SGFormer(in_channels=conv_hidden[0]* 2, 
                         hidden_channels=conv_hidden[-1], 
                         out_channels=conv_hidden[-1], 
                         gnn=gcn_2, 
                         use_graph=True), 'x, edge_index -> x1'),
                        ])
            
            gcn_3 = GCN(in_channels=conv_hidden[0]* 2, 
                      hidden_channels=conv_hidden[0]* 2, 
                      out_channels=conv_hidden[-1])
            self.conv_logvar = Sequential('x, edge_index', [
                        (SGFormer(in_channels=conv_hidden[0]* 2, 
                         hidden_channels=conv_hidden[-1], 
                         out_channels=conv_hidden[-1], 
                         gnn=gcn_3, 
                         use_graph=True), 'x, edge_index -> x1'),
                        ])

        elif self.Conv_type == "SAGEConv":
            from torch_geometric.nn import SAGEConv
            self.conv = Sequential('x, edge_index', [
                        (SAGEConv(linear_encoder_hidden[-1], conv_hidden[0]* 2), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden[0]* 2),
                        nn.ReLU(inplace=True), 
                        ])
            self.conv_mean = Sequential('x, edge_index', [
                        (SAGEConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ])
            self.conv_logvar = Sequential('x, edge_index', [
                        (SAGEConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ])

        elif self.Conv_type == "GraphConv":
            from torch_geometric.nn import GraphConv
            self.conv = Sequential('x, edge_index', [
                        (GraphConv(linear_encoder_hidden[-1], conv_hidden[0]* 2), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden[0]* 2),
                        nn.ReLU(inplace=True), 
                        ])
            self.conv_mean = Sequential('x, edge_index', [
                        (GraphConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ])
            self.conv_logvar = Sequential('x, edge_index', [
                        (GraphConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ])

        elif self.Conv_type == "GatedGraphConv":
            from torch_geometric.nn import GatedGraphConv
            self.conv = Sequential('x, edge_index', [
                        (GatedGraphConv(linear_encoder_hidden[-1], conv_hidden[0]* 2), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden[0]* 2),
                        nn.ReLU(inplace=True), 
                        ])
            self.conv_mean = Sequential('x, edge_index', [
                        (GatedGraphConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ])
            self.conv_logvar = Sequential('x, edge_index', [
                        (GatedGraphConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ])

        elif self.Conv_type == "ResGatedGraphConv":
            from torch_geometric.nn import ResGatedGraphConv
            self.conv = Sequential('x, edge_index', [
                        (ResGatedGraphConv(linear_encoder_hidden[-1], conv_hidden[0]* 2), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden[0]* 2),
                        nn.ReLU(inplace=True), 
                        ])
            self.conv_mean = Sequential('x, edge_index', [
                        (ResGatedGraphConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ])
            self.conv_logvar = Sequential('x, edge_index', [
                        (ResGatedGraphConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ])

        elif self.Conv_type == "TransformerConv":
            from torch_geometric.nn import TransformerConv
            self.conv = Sequential('x, edge_index', [
                        (TransformerConv(linear_encoder_hidden[-1], conv_hidden[0]* 2), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden[0]* 2),
                        nn.ReLU(inplace=True), 
                        ])
            self.conv_mean = Sequential('x, edge_index', [
                        (TransformerConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ])
            self.conv_logvar = Sequential('x, edge_index', [
                        (TransformerConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ])

        elif self.Conv_type == "TAGConv":
            from torch_geometric.nn import TAGConv
            self.conv = Sequential('x, edge_index', [
                        (TAGConv(linear_encoder_hidden[-1], conv_hidden[0]* 2), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden[0]* 2),
                        nn.ReLU(inplace=True), 
                        ])
            self.conv_mean = Sequential('x, edge_index', [
                        (TAGConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ])
            self.conv_logvar = Sequential('x, edge_index', [
                        (TAGConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ])

        elif self.Conv_type == "ARMAConv":
            from torch_geometric.nn import ARMAConv
            self.conv = Sequential('x, edge_index', [
                        (ARMAConv(linear_encoder_hidden[-1], conv_hidden[0]* 2), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden[0]* 2),
                        nn.ReLU(inplace=True), 
                        ])
            self.conv_mean = Sequential('x, edge_index', [
                        (ARMAConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ])
            self.conv_logvar = Sequential('x, edge_index', [
                        (ARMAConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ])
        elif self.Conv_type == "SGConv":
            from torch_geometric.nn import SGConv
            self.conv = Sequential('x, edge_index', [
                        (SGConv(linear_encoder_hidden[-1], conv_hidden[0]* 2), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden[0]* 2),
                        nn.ReLU(inplace=True), 
                        ])
            self.conv_mean = Sequential('x, edge_index', [
                        (SGConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ])
            self.conv_logvar = Sequential('x, edge_index', [
                        (SGConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ])     
        elif self.Conv_type == "MFConv":
            from torch_geometric.nn import MFConv
            self.conv = Sequential('x, edge_index', [
                        (MFConv(linear_encoder_hidden[-1], conv_hidden[0]* 2), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden[0]* 2),
                        nn.ReLU(inplace=True), 
                        ])
            self.conv_mean = Sequential('x, edge_index', [
                        (MFConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ])
            self.conv_logvar = Sequential('x, edge_index', [
                        (MFConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ]) 
        elif self.Conv_type == "RGCNConv":
            from torch_geometric.nn import RGCNConv
            self.conv = Sequential('x, edge_index', [
                        (RGCNConv(linear_encoder_hidden[-1], conv_hidden[0]* 2, num_relations=3), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden[0]* 2),
                        nn.ReLU(inplace=True), 
                        ])
            self.conv_mean = Sequential('x, edge_index', [
                        (RGCNConv(conv_hidden[0]* 2, conv_hidden[-1], num_relations=3), 'x, edge_index -> x1'),
                        ])
            self.conv_logvar = Sequential('x, edge_index', [
                        (RGCNConv(conv_hidden[0]* 2, conv_hidden[-1], num_relations=3), 'x, edge_index -> x1'),
                        ]) 
        elif self.Conv_type == "FeaStConv":
            from torch_geometric.nn import FeaStConv
            self.conv = Sequential('x, edge_index', [
                        (FeaStConv(linear_encoder_hidden[-1], conv_hidden[0]* 2), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden[0]* 2),
                        nn.ReLU(inplace=True), 
                        ])
            self.conv_mean = Sequential('x, edge_index', [
                        (FeaStConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ])
            self.conv_logvar = Sequential('x, edge_index', [
                        (FeaStConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ]) 
        elif self.Conv_type == "LEConv":
            from torch_geometric.nn import LEConv
            self.conv = Sequential('x, edge_index', [
                        (LEConv(linear_encoder_hidden[-1], conv_hidden[0]* 2), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden[0]* 2),
                        nn.ReLU(inplace=True), 
                        ])
            self.conv_mean = Sequential('x, edge_index', [
                        (LEConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ])
            self.conv_logvar = Sequential('x, edge_index', [
                        (LEConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ]) 
        elif self.Conv_type == "ClusterGCNConv":
            from torch_geometric.nn import ClusterGCNConv
            self.conv = Sequential('x, edge_index', [
                        (ClusterGCNConv(linear_encoder_hidden[-1], conv_hidden[0]* 2), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden[0]* 2),
                        nn.ReLU(inplace=True), 
                        ])
            self.conv_mean = Sequential('x, edge_index', [
                        (ClusterGCNConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ])
            self.conv_logvar = Sequential('x, edge_index', [
                        (ClusterGCNConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ]) 
        self.dc = InnerProductDecoder(p_drop)
        # DEC cluster layer
        self.cluster_layer = Parameter(torch.Tensor(self.dec_cluster_n, self.linear_encoder_hidden[-1] + self.conv_hidden[-1]))
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
        return mse_weight * mse_loss + bce_kld_weight* (bce_logits_loss + KLD)

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
            

def buildNetwork(
    in_features, 
    out_features, 
    activate = "relu", 
    p_drop = 0.0
    ):
    net = []
    net.append(nn.Linear(in_features, out_features))
    net.append(BatchNorm(out_features, momentum=0.01, eps=0.001))
    if activate=="relu":
        net.append(nn.ELU())
    elif activate=="sigmoid":
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
        return mse_weight * mse_loss + bce_logits_loss + bce_kld_weight*KLD

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
