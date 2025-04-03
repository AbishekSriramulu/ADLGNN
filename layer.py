from __future__ import division
import torch
import torch.nn as nn
from torch.nn import init
import numbers
import torch.nn.functional as F
from logging import getLogger
import numpy as np
#from rational.torch import Rational
import math
import matplotlib.pyplot as plt
from math import sqrt
import os

def init_weights(m):
    """
    Initialize network weights using Xavier/Glorot initialization
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

class BasicConv(nn.Module):
    def __init__(self, c_in, c_out, dropout=0.1):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.apply(init_weights)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int, dropout=0.1):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Conv2d(F_g, F_int, 1)
        self.W_x = nn.Conv2d(F_l, F_int, 1)
        self.psi = nn.Conv2d(F_int, 1, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(dropout)
        self.apply(init_weights)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.leaky_relu(g1 + x1)
        psi = self.psi(psi)
        psi = self.dropout(psi)
        return x * psi

class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False, dropout=0.1):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate(256, 256, 128, dropout)
        self.hc = AttentionGate(256, 256, 128, dropout)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate(256, 256, 128, dropout)
        self.apply(init_weights)

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3)
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3)
        x_perm2 = x.permute(0, 3, 2, 1)
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1)
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1/3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1/2 * (x_out11 + x_out21)
        return x_out

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x
    
    

class MFIn(nn.Module):
    """
    req: first feature in feature dimension has to be the target variable
    Normalisation for multivariate multi-feature datasets
    in,out shape (batch_size, feature_dim, num_nodes, input_window)
    Denormalisation  
    in, out shape (batch_size, num_nodes, output_window)
    """
    def __init__(self, num_nodes, num_timesteps_input, num_timesteps_output, dropout=0.1):
        super(MFIn, self).__init__()
        self.num_nodes = num_nodes
        self.num_timesteps_input = num_timesteps_input
        self.num_timesteps_output = num_timesteps_output
        self.dropout = nn.Dropout(dropout)
        self.apply(init_weights)

    def forward(self, x):
        # Normalize input
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        x = (x - mean) / (std + 1e-8)
        x = self.dropout(x)
        return x

    def denormalize(self, x, mean, std):
        return x * std + mean

class series_decomp(nn.Module):
    """
    Series decomposition block 
    (b,t,n)
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class decom_2D(nn.Module):
    """
    Decomposition
    """
    def __init__(self, seq_len, pred_len, var_len):
        super(decom_2D, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.var_len = var_len
        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.Linear_Seasonal = nn.Linear(self.seq_len*self.var_len,self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len*self.var_len,self.pred_len)

    def forward(self, x):
        # x: [Batch, Features, Node, Time] 
        b,c,n,t = x.size()
        x = x.permute(0,1,3,2).flatten(start_dim=0, end_dim=1) # x: [Batch * Features, Time, Node]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal = seasonal_init.reshape(b,c,t,n).permute(0,1,3,2) #[Batch,Features, Node, Time]
        trend = trend_init.reshape(b,c,t,n).permute(0,1,3,2) #[Batch,Features, Node, Time]
        seasonal_init, trend_init = seasonal.permute(0,2,3,1).reshape(b,n,t*c), trend.permute(0,2,3,1).reshape(b,n,t*c) #[Batch, Node, Time*Features] 
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        return x, seasonal, trend # to [Batch, Node, Time], [Batch,Features, Node, Time], [Batch,Features, Node, Time]



class NConv(nn.Module):
    def __init__(self):
        super(NConv, self).__init__()
        self.chunk_size = 1000  # Process data in chunks for memory efficiency

    def forward(self, x, adj):
        """
        Memory-efficient graph convolution operation
        Args:
            x(torch.tensor): (B, input_channels, N, T)
            adj(torch.tensor): N * N
        Returns:
            torch.tensor: (B, input_channels, N, T)
        """
        try:
            batch_size, channels, nodes, time = x.size()
            
            # Process in chunks to save memory
            if nodes > self.chunk_size:
                chunks = []
                for i in range(0, nodes, self.chunk_size):
                    end = min(i + self.chunk_size, nodes)
                    chunk_x = x[:, :, i:end, :]
                    chunk_adj = adj[i:end, :]
                    chunk_result = torch.einsum('ncwl,vw->ncvl', (chunk_x, chunk_adj))
                    chunks.append(chunk_result)
                x = torch.cat(chunks, dim=2)
            else:
                x = torch.einsum('ncwl,vw->ncvl', (x, adj))
            
            return x.contiguous()
        except Exception as e:
            print(f"Error in NConv: {str(e)}")
            raise

class dy_nconv(nn.Module):
    def __init__(self):
        super(dy_nconv, self).__init__()
        self.chunk_size = 1000  # Process data in chunks for memory efficiency

    def forward(self, x, A):
        """
        Memory-efficient dynamic graph convolution operation
        Args:
            x(torch.tensor): (B, input_channels, N, T)
            A(torch.tensor): (B, N, N, T)
        Returns:
            torch.tensor: (B, input_channels, N, T)
        """
        try:
            batch_size, channels, nodes, time = x.size()
            
            # Process in chunks to save memory
            if nodes > self.chunk_size:
                chunks = []
                for i in range(0, nodes, self.chunk_size):
                    end = min(i + self.chunk_size, nodes)
                    chunk_x = x[:, :, i:end, :]
                    chunk_A = A[:, i:end, :, :]
                    chunk_result = torch.einsum('ncvl,nvwl->ncwl', (chunk_x, chunk_A))
                    chunks.append(chunk_result)
                x = torch.cat(chunks, dim=2)
            else:
                x = torch.einsum('ncvl,nvwl->ncwl', (x, A))
            
            return x.contiguous()
        except Exception as e:
            print(f"Error in dy_nconv: {str(e)}")
            raise

class linear(nn.Module):
    def __init__(self,c_in,c_out,bias=True):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=bias)

    def forward(self,x):
        return self.mlp(x)


class prop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(prop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear(c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self,x,adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        dv = d
        a = adj / dv.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
        ho = self.mlp(h)
        return ho

class mixprop_attention(nn.Module):
    def __init__(self,dim,c_in,c_out,gdep,dropout,alpha):
        super(mixprop_attention, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep+1)*c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha
        self.mha2 = MultiHeadAttention(1, dim, 40, 40)

    def forward(self,x,adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x 
        out = [h]
        a = adj / d.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
            out.append(h)
        ho = torch.cat(out,dim=1) 
        ho = self.mlp(ho) 
        for i in range(ho.size(0)):
            for j in range(ho.size(1)):
                temp_ho = ho[i,j,:,:].clone().unsqueeze(0)
                attn_op, attn_weights = self.mha2(temp_ho, temp_ho, temp_ho)
                ho[i,j,:,:] = attn_op.squeeze(0)

        return ho

class mixprop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep+1)*c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self,x,adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x 
        out = [h]
        a = adj / d.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
            out.append(h)
        ho = torch.cat(out,dim=1) 
        ho = self.mlp(ho) 
        return ho

class dy_mixprop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(dy_mixprop, self).__init__()
        self.nconv = dy_nconv()
        self.mlp1 = linear((gdep+1)*c_in,c_out)
        self.mlp2 = linear((gdep+1)*c_in,c_out)

        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha
        self.lin1 = linear(c_in,c_in)
        self.lin2 = linear(c_in,c_in)


    def forward(self,x):
        x1 = torch.tanh(self.lin1(x))
        x2 = torch.tanh(self.lin2(x))
        adj = self.nconv(x1.transpose(2,1),x2)
        adj0 = torch.softmax(adj, dim=2)
        adj1 = torch.softmax(adj.transpose(2,1), dim=2)

        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,adj0)
            out.append(h)
        ho = torch.cat(out,dim=1)
        ho1 = self.mlp1(ho)


        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, adj1)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho2 = self.mlp2(ho)

        return ho1+ho2



class dilated_1D(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_1D, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2,3,6,7]
        self.tconv = nn.Conv2d(cin,cout,(1,7),dilation=(1,dilation_factor))

    def forward(self,input):
        x = self.tconv(input)
        return x

class dilated_inception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_inception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [7] 
        cout = int(cout/len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin,cout,(1,kern),dilation=(1,dilation_factor)))

    def forward(self,input):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][...,-x[-1].size(3):]
        x = torch.cat(x,dim=1)
        return x

'''
self, n_src_vocab, n_trg_vocab, src_pad_idx, trg_pad_idx,
d_word_vec=512, d_model=512, d_inner=2048,
n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200,
trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True
'''
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.elu(attn)
        
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  

        q, attn = self.attention(q, k, v, mask=mask)

        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn

class graph_attention_constructor(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_attention_constructor, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,dim)
            self.lin2 = nn.Linear(dim,dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat
        self.dropout = nn.Dropout(0.1)  # Add dropout for regularization

    def forward(self, idx):
        try:
            if self.static_feat is None:
                nodevec1 = self.emb1(idx)
                nodevec2 = self.emb2(idx)
            else:
                if idx.max() >= self.static_feat.size(0):
                    raise ValueError(f"Index {idx.max()} out of bounds for static features of size {self.static_feat.size(0)}")
                nodevec1 = self.static_feat[idx,:]
                nodevec2 = nodevec1

            nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
            nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

            a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
            adj = F.relu(torch.tanh(self.alpha*a))
            adj = self.dropout(adj)  # Apply dropout
            return adj
        except Exception as e:
            print(f"Error in graph_attention_constructor: {str(e)}")
            raise

class graph_constructor(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,dim)
            self.lin2 = nn.Linear(dim,dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat
        self.dropout = nn.Dropout(0.1)  # Add dropout for regularization

    def forward(self, idx):
        try:
            if self.static_feat is None:
                nodevec1 = self.emb1(idx)
                nodevec2 = self.emb2(idx)
            else:
                if idx.max() >= self.static_feat.size(0):
                    raise ValueError(f"Index {idx.max()} out of bounds for static features of size {self.static_feat.size(0)}")
                nodevec1 = self.static_feat[idx,:]
                nodevec2 = nodevec1

            nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
            nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

            a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
            adj = F.relu(torch.tanh(self.alpha*a))
            adj = self.dropout(adj)  # Apply dropout
            
            mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
            mask.fill_(float('0'))
            s1,t1 = adj.topk(self.k,1)
            mask.scatter_(1,t1,s1.fill_(1))
            adj = adj*mask
            return adj
        except Exception as e:
            print(f"Error in graph_constructor: {str(e)}")
            raise

class graph_global(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_global, self).__init__()
        self.nnodes = nnodes
        self.A = nn.Parameter(torch.randn(nnodes, nnodes).to(device), requires_grad=True).to(device)

    def forward(self, idx):
        return F.relu(self.A)


class graph_undirected(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_undirected, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb1(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin1(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1,t1 = adj.topk(self.k,1)
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask
        return adj



class graph_directed(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_directed, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,dim)
            self.lin2 = nn.Linear(dim,dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1,t1 = adj.topk(self.k,1)
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask
        return adj


class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.grad_clip = 1.0  # Add gradient clipping value

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, idx):
        try:
            if self.elementwise_affine:
                # Clip gradients during forward pass
                if self.weight.requires_grad:
                    self.weight.grad = torch.clamp(self.weight.grad, -self.grad_clip, self.grad_clip)
                if self.bias.requires_grad:
                    self.bias.grad = torch.clamp(self.bias.grad, -self.grad_clip, self.grad_clip)
                
                return F.layer_norm(input, tuple(input.shape[1:]), 
                                  self.weight[:,idx,:], 
                                  self.bias[:,idx,:], 
                                  self.eps)
            else:
                return F.layer_norm(input, tuple(input.shape[1:]), 
                                  self.weight, 
                                  self.bias, 
                                  self.eps)
        except Exception as e:
            print(f"Error in LayerNorm: {str(e)}")
            raise

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)

class Linear(nn.Module):
    def __init__(self, c_in, c_out, bias=True):
        super(Linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias)

    def forward(self, x):
        return self.mlp(x)

class SEAttention(nn.Module):

    def __init__(self, channel=512,reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)



    
    
class MixPropAttention(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        """
        MixPropAttention GCN

        Args:
            c_in: input
            c_out: output
            gdep: GCN layers
            dropout: dropout
            alpha: beta in paper
        """
        super(MixPropAttention, self).__init__()
        self.nconv = NConv()
        self.mlp = Linear((gdep+1)*c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha
        #self.triplet = TripletAttention()
        #self.attention = TripletAttention() 
        self.attention = SEAttention(channel=(gdep+1)*c_in,reduction=16 ) 

    def forward(self, x, adj):
        """
        MixProp GCN

        Args:
            x(torch.tensor):  (B, c_in, N, T)
            adj(torch.tensor):  N * N

        Returns:
            torch.tensor: (B, c_out, N, T)
        """
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        out = [h]  # h(0) = h_in = x
        a = adj / d.view(-1, 1)  # A' = A * D^-1
        for i in range(self.gdep):
            # h(k) = alpha * h_in + (1 - alpha) * A' * H(k-1)
            # h: shape = (B, c_in, N, T)
            h = self.alpha*x + (1-self.alpha)*self.nconv(h, a)
            out.append(h)
        # ho: (B, c_in * (gdep + 1), N, T)
        ho = torch.cat(out, dim=1)
        ho = self.attention(ho)
        ho = self.mlp(ho)
        return ho  # (B, c_out, N, T)


    
    