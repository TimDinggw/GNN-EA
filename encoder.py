import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# --- torch_geometric Packages ---
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, add_remaining_self_loops, degree
from torch_scatter import scatter_add, scatter
# --- torch_geometric Packages end ---

from utils import *
from torch_geometric.nn import SAGEConv, AGNNConv, AntiSymmetricConv, ChebConv, GCNConv, GATConv, GATv2Conv, RGATConv, SGConv
from torch_geometric.nn import SSGConv, TransformerConv, APPNP, TAGConv, ClusterGCNConv, GENConv, FiLMConv
from torch_geometric.nn import GCN2Conv, SuperGATConv, FAConv, EGConv

class Encoder(torch.nn.Module):
    def __init__(self, name, hiddens, heads, skip_conn, activation, feat_drop, bias, ent_num, rel_num):
        super(Encoder, self).__init__()
        #print("encoder init")
        self.name = name
        self.hiddens = hiddens
        self.heads = heads
        self.num_layers = len(hiddens) - 1
        self.gnn_layers = nn.ModuleList()
        self.activation = None
        if activation == 'elu':
            self.activation = F.elu
        elif activation == "relu":
            self.activation = F.relu
        elif activation == "tanh":
            self.activation = torch.tanh
        elif activation == "sigmoid":
            self.activation = torch.sigmoid
        self.feat_drop = feat_drop
        self.bias = bias
        self.skip_conn = skip_conn
        self.ent_num = ent_num
        self.rel_num = rel_num
        self.device = None
        # Dual-AMN
        self.negative_slope = 0 
        self.attn_drop = 0

        for l in range(0, self.num_layers):
            if self.name == "gcn-align":
                self.gnn_layers.append(
                    GCNAlign_GCNConv(in_channels=self.hiddens[l], out_channels=self.hiddens[l+1], improved=False, cached=True, bias=self.bias)
                )
            elif self.name == "mlp":
                self.gnn_layers.append(
                    nn.Linear(self.hiddens[l], self.hiddens[l+1], bias=self.bias)
                )
            elif self.name == "compgcn":
                self.gnn_layers.append(
                    CompGCNConv(self.hiddens[l], self.hiddens[l+1])
                )
            elif self.name == "kecg":
                self.gnn_layers.append(
                    KECGMultiHeadGraphAttention(self.heads[l], self.hiddens[l] * self.heads[l - 1] if l else self.hiddens[l], self.hiddens[l+1], attn_dropout=0.0, init=nn.init.ones_, bias=self.bias)
                )
            elif self.name == "graphsage":
                self.gnn_layers.append(
                    SAGEConv(self.hiddens[l], self.hiddens[l+1], bias=self.bias)
                )
            elif self.name == "agnn":
                self.gnn_layers.append(
                    AGNNConv(add_self_loops = False)
                )
            elif self.name == "antisymmetric":
                self.gnn_layers.append(
                    AntiSymmetricConv(self.hiddens[l])
                )
            elif self.name == "cheb":
                self.gnn_layers.append(
                    ChebConv(self.hiddens[l], self.hiddens[l+1], 1, bias=self.bias)
                )
            elif self.name == "gcn":
                self.gnn_layers.append(
                    GCNConv(self.hiddens[l], self.hiddens[l+1], bias=self.bias)
                )
            elif self.name == "gat":
                self.gnn_layers.append(
                    GATConv(self.hiddens[l], self.hiddens[l+1], bias=self.bias)
                )
            elif self.name == "gatv2":
                self.gnn_layers.append(
                    GATv2Conv(self.hiddens[l], self.hiddens[l+1], bias=self.bias)
                )
            elif self.name == "rgat":
                self.gnn_layers.append(
                    RGATConv(self.hiddens[l], self.hiddens[l+1], rel_num)
                )
            elif self.name == "sg":
                self.gnn_layers.append(
                    SGConv(self.hiddens[l], self.hiddens[l+1])
                )
            elif self.name == "ssg":
                self.gnn_layers.append(
                    SSGConv(self.hiddens[l], self.hiddens[l+1], alpha=0.5)
                )
            elif self.name == "transformer":
                self.gnn_layers.append(
                    TransformerConv(self.hiddens[l], self.hiddens[l+1])
                )
            elif self.name == "appnp":
                self.gnn_layers.append(
                    APPNP(K=1, alpha=0.5)
                )
            elif self.name == "tag":
                self.gnn_layers.append(
                    TAGConv(self.hiddens[l], self.hiddens[l+1])
                )
            elif self.name == "clustergcn":
                self.gnn_layers.append(
                    ClusterGCNConv(self.hiddens[l], self.hiddens[l+1])
                )
            elif self.name == "gen":
                self.gnn_layers.append(
                    GENConv(self.hiddens[l], self.hiddens[l+1])
                )
            elif self.name == "film":
                self.gnn_layers.append(
                    FiLMConv(self.hiddens[l], self.hiddens[l+1])
                )
            elif self.name == "gcn2":
                self.gnn_layers.append(
                    GCN2Conv(self.hiddens[l], alpha=0.5)
                )
            elif self.name == "supergat":
                self.gnn_layers.append(
                    SuperGATConv(self.hiddens[l], self.hiddens[l+1])
                )
            elif self.name == "fa":
                self.gnn_layers.append(
                    FAConv(self.hiddens[l])
                )
            elif self.name == "eg":
                self.gnn_layers.append(
                    EGConv(self.hiddens[l], self.hiddens[l+1], num_heads=4, num_bases=4)
                )
            elif self.name == "dual-amn":
                self.gnn_layers.append(
                    Dual_AMN(in_channels=self.hiddens[l]*self.heads[l-1], out_channels=self.hiddens[l+1], heads=self.heads[l], concat=True, negative_slope=self.negative_slope, dropout=self.attn_drop, bias=self.bias)
                )
            elif self.name == "rrea":
                self.gnn_layers.append(
                    RREA(in_channels=self.hiddens[l]*self.heads[l-1], out_channels=self.hiddens[l+1], heads=self.heads[l], concat=True, negative_slope=self.negative_slope, dropout=self.attn_drop, bias=self.bias)
                )
            elif self.name == "vrgcn":
                self.gnn_layers.append(
                    VRGCNConv(self.hiddens[l])
                )
            elif self.name == "hybrid":
                break
            # elif self.name == "SLEF-DESIGN":
            #     self.gnn_layers.append(
            #         SLEF-DESIGN_Conv()
            #     )
            else:
                raise NotImplementedError("bad encoder name: " + self.name)
        
        if self.name == "hybrid":
            l = 0
            self.gnn_layers.append(
                norel_layer(self.hiddens[l], self.hiddens[l+1]) # 无关系的
            )
            l = 1
            self.gnn_layers.append(
                withrel_layer(self.hiddens[l], self.hiddens[l+1])
            )

        if self.skip_conn == "highway" or self.skip_conn == "concatallhighway":
            self.gate_weights = nn.ParameterList()
            self.gate_biases = nn.ParameterList()
            for l in range(self.num_layers):
                self.gate_weights.append(nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.hiddens[l], self.hiddens[l]))))    
                self.gate_biases.append(nn.Parameter(torch.zeros(self.hiddens[l])))
        # if self.name == "SLEF-DESIGN":
        #     '''SLEF-DESIGN: extra parameters''

    def forward(self, edges, rels, x, r):
        if self.device is None:
            self.device = x.device
        #print("encoder forward")
        edges = edges.t()
        #print("x before layers", x)

        x_0 = x
        all_layer_outputs = [x]

        for l in range(self.num_layers):

            if self.skip_conn == "residual":
                residual = x
                residual = residual.to(self.device)
            if self.skip_conn == "highway" or self.skip_conn == "concatallhighway":
                highway_features = x
                highway_features = highway_features.to(self.device)

            x = F.dropout(x, p=self.feat_drop, training=self.training)

            if self.name == "mlp":
                x = self.gnn_layers[l](x)
            elif self.name in ["gcn2", "fa"]:
                x = self.gnn_layers[l](x, x_0 ,edges)
            elif self.name in ["rgat"]:
                x = self.gnn_layers[l](x, edges, rels)
            elif self.name == "kecg":
                self.diag = True
                x = self.gnn_layers[l](x, edges)
                if self.diag:
                    x = x.mean(dim=0)
            elif self.name in ["dual-amn", "rrea", "vrgcn", "hybrid"]:
                x = self.gnn_layers[l](x, edges, rels, r)
            elif self.name in ["compgcn"]:
                x, r = self.gnn_layers[l](x, edges, rels, r)
            # elif self.name == "SLEF-DESIGN":
            #     '''SLEF-DESIGN: special encoder forward'''
            else:
                x = self.gnn_layers[l](x, edges)         

            if l != self.num_layers - 1:
                if self.activation:
                    x = self.activation(x)

            if self.skip_conn == "residual":
                x = x + residual

            if self.skip_conn == "highway" or self.skip_conn == "concatallhighway":
                gate = torch.matmul(highway_features, self.gate_weights[l])
                gate = gate + self.gate_biases[l]
                gate = torch.sigmoid(gate)
                x = x * gate + highway_features * (1.0 - gate)

            all_layer_outputs.append(x)

        if self.skip_conn == "concatall" or self.skip_conn == "concatallhighway":
            return torch.cat(all_layer_outputs, dim=1)
        elif self.skip_conn == 'concat0andl':
            return torch.cat([all_layer_outputs[0], all_layer_outputs[self.num_layers]], dim=1)
        return x   

    def __repr__(self):
        return '{}(name={}): {}'.format(self.__class__.__name__, self.name, "\n".join([layer.__repr__() for layer in self.gnn_layers]))
# --- Main Models: Encoder end ---


# --- Encoding Modules ---
class GCNAlign_GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, improved=False, cached=False,
                 bias=True, **kwargs):
        super(GCNAlign_GCNConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.device = None

        self.weight = Parameter(torch.Tensor(1, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        if self.device is None:
            self.device = x.device
        self.weight = self.weight.to(self.device)
        x = torch.mul(x, self.weight)

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight,
                                         self.improved, x.dtype)
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

    
# KECG
class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class KECGMultiHeadGraphAttention(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    https://github.com/Diego999/pyGAT/blob/master/layers.py
    """
    def __init__(self, n_head, f_in, f_out, attn_dropout, diag=True, init=None, bias=False):
        super(KECGMultiHeadGraphAttention, self).__init__()
        self.n_head = n_head
        self.f_in = f_in
        self.f_out = f_out
        self.diag = diag
        self.device = None
        if self.diag:
            self.w = Parameter(torch.Tensor(n_head, 1, f_out))
        else:
            self.w = Parameter(torch.Tensor(n_head, f_in, f_out))
        self.a_src_dst = Parameter(torch.Tensor(n_head, f_out * 2, 1))
        self.attn_dropout = attn_dropout
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.special_spmm = SpecialSpmm()
        if bias:
            self.bias = Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)
        if init is not None and diag:
            init(self.w)
            stdv = 1. / math.sqrt(self.a_src_dst.size(1))
            nn.init.uniform_(self.a_src_dst, -stdv, stdv)
        else:
            nn.init.xavier_uniform_(self.w)
            nn.init.xavier_uniform_(self.a_src_dst)

    def forward(self, input, edge):
        if self.device is None:
            self.device = input.device
        self.w = self.w.to(self.device)
        if self.bias is not None:
            self.bias = self.bias.to(self.device)
        output = []
        for i in range(self.n_head):
            N = input.size()[0]
            if self.diag:
                h = torch.mul(input, self.w[i])
            else:
                h = torch.mm(input, self.w[i])

            edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1) # edge: 2*D x E
            edge_e = torch.exp(-self.leaky_relu(edge_h.mm(self.a_src_dst[i]).squeeze())) # edge_e: 1 x E
            e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1)).cuda()  if next(self.parameters()).is_cuda else torch.ones(size=(N, 1))) # e_rowsum: N x 1
            edge_e = F.dropout(edge_e, self.attn_dropout, training=self.training)   # edge_e: 1 x E
            h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
            h_prime = h_prime.div(e_rowsum)
            output.append(h_prime.unsqueeze(0))
        
        output = torch.cat(output, dim=0)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    
    def __repr__(self):
        if self.diag:
            return self.__class__.__name__ + ' (' + str(self.f_out) + ' -> ' + str(self.f_out) + ') * ' + str(self.n_head) + ' heads'
        else:
            return self.__class__.__name__ + ' (' + str(self.f_in) + ' -> ' + str(self.f_out) + ') * ' + str(self.n_head) + ' heads'


class Dual_AMN(MessagePassing):
    def __init__(self, in_channels, out_channels, heads = 1, concat = True, dropout = 0, negative_slope=0.2, 
                 bias = False, **kwargs):
        super(Dual_AMN, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.pi = 3.1415926535897932384626
        self.weight_1 = Parameter(
            torch.Tensor(2 * in_channels, out_channels))
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight_1)
        nn.init.xavier_uniform_(self.att)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_type, r = None, size=None):
        """"""
        # 本来下面两行没有被注释,不使用自环
        #if size is None and torch.is_tensor(x):
        #    edge_index, _ = remove_self_loops(edge_index)

            # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        return self.propagate(edge_index, size=size, x=x, r_ij = r)

    def message(self, edge_index_i, ptr, x_i, x_j, size_i, r_ij):
        r_ij = F.normalize(r_ij, p = 2)
        trans_embed = (x_j * r_ij).sum(dim = -1).view(-1, 1) * r_ij
        xj_rel = x_j - 2.0 * trans_embed
        xj_rel = xj_rel.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)
        r_ij = r_ij.view(-1, self.heads, self.out_channels)
        alpha = (r_ij * self.att[:, :, self.out_channels:]).sum(dim=-1)
        alpha = F.elu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, ptr, size_i)

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = xj_rel * alpha.view(-1, self.heads, 1)

        return out.view(-1, self.out_channels)

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
    
class RREA(MessagePassing):
    def __init__(self, in_channels, out_channels, heads = 1, concat = True, dropout = 0, negative_slope=0.2, 
                 bias = False, **kwargs):
        super(RREA, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.pi = 3.1415926535897932384626
        self.weight_1 = Parameter(
            torch.Tensor(2 * in_channels, out_channels))
        self.att = Parameter(torch.Tensor(1, heads, 3 * out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight_1)
        nn.init.xavier_uniform_(self.att)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_type, r = None, size=None):
        """"""
        # 本来下面两行没有被注释,不使用自环
        #if size is None and torch.is_tensor(x):
        #    edge_index, _ = remove_self_loops(edge_index)

            # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        return self.propagate(edge_index, size=size, x=x, r_ij = r)

    def message(self, edge_index_i, ptr, x_i, x_j, size_i, r_ij):
        # RREA
        r_ij = F.normalize(r_ij, p = 2)
        #trans_embed = (r_ij * r_ij).sum(dim = -1).view(-1, 1) * x_j
        trans_embed = torch.sum(x_j * r_ij, 1, keepdim=True) * r_ij
        xj_rel = x_j - 2.0 * trans_embed
        xj_rel = xj_rel.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)
        r_ij = r_ij.view(-1, self.heads, self.out_channels)
        x_i = x_i.view(-1, self.heads, self.out_channels)
        y = torch.concat([x_i, x_j, r_ij], dim=-1)
        alpha = (torch.concat([x_i, x_j, r_ij],dim=-1) * self.att).sum(dim=-1)
        alpha = F.elu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, ptr, size_i)
        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = xj_rel * alpha.view(-1, self.heads, 1)


        return out.view(-1, self.out_channels)

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
    


import inspect

def scatter_(name, src, index, dim_size=None):
	r"""Aggregates all values from the :attr:`src` tensor at the indices
	specified in the :attr:`index` tensor along the first dimension.
	If multiple indices reference the same location, their contributions
	are aggregated according to :attr:`name` (either :obj:`"add"`,
	:obj:`"mean"` or :obj:`"max"`).

	Args:
		name (string): The aggregation to use (:obj:`"add"`, :obj:`"mean"`,
			:obj:`"max"`).
		src (Tensor): The source tensor.
		index (LongTensor): The indices of elements to scatter.
		dim_size (int, optional): Automatically create output tensor with size
			:attr:`dim_size` in the first dimension. If set to :attr:`None`, a
			minimal sized output tensor is returned. (default: :obj:`None`)

	:rtype: :class:`Tensor`
	"""
	if name == 'add': name = 'sum'
	assert name in ['sum', 'mean', 'max']
	out = scatter(src, index, dim=0, out=None, dim_size=dim_size, reduce=name)
	return out[0] if isinstance(out, tuple) else out

class CompGCNMessagePassing(torch.nn.Module):
	r"""Base class for creating message passing layers

	.. math::
		\mathbf{x}_i^{\prime} = \gamma_{\mathbf{\Theta}} \left( \mathbf{x}_i,
		\square_{j \in \mathcal{N}(i)} \, \phi_{\mathbf{\Theta}}
		\left(\mathbf{x}_i, \mathbf{x}_j,\mathbf{e}_{i,j}\right) \right),

	where :math:`\square` denotes a differentiable, permutation invariant
	function, *e.g.*, sum, mean or max, and :math:`\gamma_{\mathbf{\Theta}}`
	and :math:`\phi_{\mathbf{\Theta}}` denote differentiable functions such as
	MLPs.
	See `here <https://rusty1s.github.io/pytorch_geometric/build/html/notes/
	create_gnn.html>`__ for the accompanying tutorial.

	"""

	def __init__(self, aggr='add'):
		super(CompGCNMessagePassing, self).__init__()

		self.message_args = inspect.getargspec(self.message)[0][1:]	# In the defined message function: get the list of arguments as list of string| For eg. in rgcn this will be ['x_j', 'edge_type', 'edge_norm'] (arguments of message function)
		self.update_args  = inspect.getargspec(self.update)[0][2:]	# Same for update function starting from 3rd argument | first=self, second=out

	def propagate(self, aggr, edge_index, **kwargs):
		r"""The initial call to start propagating messages.
		Takes in an aggregation scheme (:obj:`"add"`, :obj:`"mean"` or
		:obj:`"max"`), the edge indices, and all additional data which is
		needed to construct messages and to update node embeddings."""

		assert aggr in ['add', 'mean', 'max']
		kwargs['edge_index'] = edge_index


		size = None
		message_args = []
		for arg in self.message_args:
			if arg[-2:] == '_i':					# If arguments ends with _i then include indic
				tmp  = kwargs[arg[:-2]]				# Take the front part of the variable | Mostly it will be 'x', 
				size = tmp.size(0)
				message_args.append(tmp[edge_index[0]])		# Lookup for head entities in edges
			elif arg[-2:] == '_j':
				tmp  = kwargs[arg[:-2]]				# tmp = kwargs['x']
				size = tmp.size(0)
				message_args.append(tmp[edge_index[1]])		# Lookup for tail entities in edges
			else:
				message_args.append(kwargs[arg])		# Take things from kwargs

		update_args = [kwargs[arg] for arg in self.update_args]		# Take update args from kwargs

		out = self.message(*message_args)
		out = scatter_(aggr, out, edge_index[0], dim_size=size)		# Aggregated neighbors for each vertex
		out = self.update(out, *update_args)

		return out

	def message(self, x_j):  # pragma: no cover
		r"""Constructs messages in analogy to :math:`\phi_{\mathbf{\Theta}}`
		for each edge in :math:`(i,j) \in \mathcal{E}`.
		Can take any argument which was initially passed to :meth:`propagate`.
		In addition, features can be lifted to the source node :math:`i` and
		target node :math:`j` by appending :obj:`_i` or :obj:`_j` to the
		variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`."""

		return x_j

	def update(self, aggr_out):  # pragma: no cover
		r"""Updates node embeddings in analogy to
		:math:`\gamma_{\mathbf{\Theta}}` for each node
		:math:`i \in \mathcal{V}`.
		Takes in the output of aggregation as first argument and any argument
		which was initially passed to :meth:`propagate`."""

		return aggr_out


class CompGCNConv(CompGCNMessagePassing):

    def __init__(self, in_channels, out_channels, act=lambda x:x, params=None, drop = 0):
        super(self.__class__, self).__init__()
        self.dropout = drop
        self.p 			= params
        self.in_channels	= in_channels
        self.out_channels	= out_channels
        self.act 		= act
        self.device		= None

        def get_param(shape):
            param = Parameter(torch.Tensor(*shape))
            nn.init.kaiming_normal_(param.data)
            return param

        self.w_loop		= get_param((out_channels, out_channels))
        self.w_in		= get_param((out_channels, out_channels))
        self.w_out		= get_param((out_channels, out_channels))
        self.w_rel 		= get_param((out_channels // 2, out_channels // 2))
        self.loop_rel 		= get_param((1, out_channels // 2));
        self.drop		= torch.nn.Dropout(self.dropout)
        self.bn			= torch.nn.BatchNorm1d(out_channels)
        self.pi = 3.1415926535897932384626

    def forward(self, x, edge_index, edge_type, rel_embed): 
        if self.device is None:
            self.device = edge_index.device
        
        #compgcn 要移除自环
        #edge_index, _ = remove_self_loops(edge_index)

        rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)
        num_edges = edge_index.size(1) // 2
        num_ent   = x.size(0)
        self.in_index, self.out_index = edge_index[:, :num_edges], edge_index[:, num_edges:]
        self.in_type,  self.out_type  = edge_type[:num_edges], 	 edge_type[num_edges:]

        self.loop_index  = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).cuda()
        self.loop_type   = torch.full((num_ent,), rel_embed.size(0)-1, dtype=torch.long).to(self.device)

        self.in_norm     = self.compute_norm(self.in_index,  num_ent)
        self.out_norm    = self.compute_norm(self.out_index, num_ent)
        
        in_res		= self.propagate('add', self.in_index,   x=x, edge_type=self.in_type,   rel_embed=rel_embed, edge_norm=self.in_norm, 	mode='in')
        loop_res	= self.propagate('add', self.loop_index, x=x, edge_type=self.loop_type, rel_embed=rel_embed, edge_norm=None, 		mode='loop')
        out_res		= self.propagate('add', self.out_index,  x=x, edge_type=self.out_type,  rel_embed=rel_embed, edge_norm=self.out_norm,	mode='out')
        out		= self.drop(in_res)*(1/3) + self.drop(out_res)*(1/3) + loop_res*(1/3)
        
#         out = self.bn(out)
        
        return out, torch.matmul(rel_embed, self.w_rel)[:-1]		# Ignoring the self loop inserted

    def rel_transform(self, ent_embed, rel_embed):
        '''
        if   self.p.opn == 'corr': 	trans_embed  = ccorr(ent_embed, rel_embed)
        elif self.p.opn == 'sub': 	trans_embed  = ent_embed - rel_embed
        elif self.p.opn == 'mult': 	trans_embed  = ent_embed * rel_embed
        else: raise NotImplementedError
        '''
        re_head, im_head = torch.chunk(ent_embed, 2, dim = -1)
        r = rel_embed / (1.5 / self.pi)
        re_relation = torch.cos(r)
        im_relation = torch.sin(r)
        re_score = re_head * re_relation + im_head * im_relation
        im_score = re_head * im_relation - im_head * re_relation
        trans_embed = torch.cat([re_score, im_score], dim=-1)
        return trans_embed

    def message(self, x_j, edge_type, rel_embed, edge_norm, mode):
        weight 	= getattr(self, 'w_{}'.format(mode))
        rel_emb = torch.index_select(rel_embed, 0, edge_type)
        xj_rel  = self.rel_transform(x_j, rel_emb)
        out	= torch.mm(xj_rel, weight)

        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out):
        return aggr_out

    def compute_norm(self, edge_index, num_ent):
        row, col	= edge_index
        edge_weight 	= torch.ones_like(row).float()
        deg		= scatter_add( edge_weight, row, dim=0, dim_size=num_ent)	# Summing number of weights of the edges
        deg_inv		= deg.pow(-0.5)							# D^{-0.5}
        deg_inv[deg_inv	== float('inf')] = 0
        norm		= deg_inv[row] * edge_weight * deg_inv[col]			# D^{-0.5}

        return norm

    def __repr__(self):
        return '{}({}, {})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels)
    

class VRGCNConv(nn.Module):
    def __init__(self, dim,dropout_rate=0.0):
        super(VRGCNConv, self).__init__()
        self.dim = dim
        self.dropout_rate = dropout_rate

        self.batch_normalization = nn.BatchNorm1d(self.dim)
        self.kernels = nn.Parameter(torch.ones(1, self.dim), requires_grad=True)

        self.device = None

    def get_sparse_tensor_in(self, e, KG):
        #KG_transpose = KG.transpose()
        receiver_indices = KG[2] #KG_transpose[2]
        mtr_values = torch.ones_like(receiver_indices).float()
        #message_indices = torch.arange(len(KG))
        message_indices = torch.arange(KG.size(1)).to(self.device)
        mtr_indices = torch.stack([receiver_indices, message_indices]).long() #.t()
        #mtr_shape = torch.tensor([e, len(KG)], dtype=torch.long)
        #M = torch.sparse_coo_tensor(mtr_indices, mtr_values, mtr_shape)
        M = torch.sparse_coo_tensor(indices=mtr_indices, values=mtr_values, size=(e, KG.size(1)))
        return M

    def get_sparse_tensor_out(self, e, KG):
        #KG_transpose = KG.transpose()
        sender_indices = KG[0] # KG_transpose[0]
        mtr_values = torch.ones_like(sender_indices).float()
        message_indices = torch.arange(KG.size(1)).to(self.device)
        mtr_indices = torch.stack([sender_indices, message_indices]).long() #.t()
        #mtr_shape = torch.tensor([e, len(KG)], dtype=torch.long)
        #M = torch.sparse_coo_tensor(mtr_indices, mtr_values, mtr_shape)
        M = torch.sparse_coo_tensor(indices=mtr_indices, values=mtr_values, size=(e, KG.size(1)))
        return M

    def get_degree(self, e, KG):
        du = torch.ones(e).to(self.device)
        for tri in KG:
            if tri[0] != tri[2]:
                du[tri[0]] += 1
                du[tri[2]] += 1
        return du.view(-1, 1).float()


    def forward(self, x, edges, rels, r):

        # 暂时
        #edges, _ = remove_self_loops(edges)
        self.device = edges.device

        x = self.batch_normalization(x)
        if self.dropout_rate > 0.0:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            r = F.dropout(r, p=self.dropout_rate, training=self.training)
        
        x = x * self.kernels
        r = r * self.kernels

        head_embedding = x.index_select(0, edges[0])
        relation_embedding = r.index_select(0, rels)
        tail_embedding = x.index_select(0, edges[1])
        


        receiver_embedding = head_embedding + relation_embedding
        sender_embedding = tail_embedding - relation_embedding


        self.e = x.size(0)
        self.triples = torch.stack((edges[0], rels, edges[1]), dim=0)
        self.M_in = self.get_sparse_tensor_in(self.e, self.triples).to(self.device)
        self.M_out = self.get_sparse_tensor_out(self.e, self.triples).to(self.device)
        self.du = self.get_degree(self.e, self.triples).to(self.device)

        sum_in = torch.sparse.mm(self.M_in, receiver_embedding)
        sum_out = torch.sparse.mm(self.M_out, sender_embedding)
        outputs = (1.0 / self.du) * (sum_out + sum_in + x)
        
        return outputs
    

class norel_layer(MessagePassing):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(norel_layer, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index, edge_type, r = None, size=None):
        return self.propagate(edge_index, size=size, x=x, r_ij = r)

    def message(self, edge_index_i, ptr, x_i, x_j, size_i, r_ij):
        return x_j

    def update(self, aggr_out):
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
    

class withrel_layer(MessagePassing):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(withrel_layer, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index, edge_type, r = None, size=None):
        return self.propagate(edge_index, size=size, x=x, r_ij = r)

    def message(self, edge_index_i, ptr, x_i, x_j, size_i, r_ij):
        out = torch.add(x_j, r_ij)
        return out.view(-1, self.out_channels)

    def update(self, aggr_out):
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)