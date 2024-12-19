from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor, PairTensor

import torch
from torch import Tensor, nn
from torch.nn import Parameter, Sequential, ReLU, Linear
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
from torch_geometric.nn.inits import zeros
from torch_geometric.nn.conv import MessagePassing


class DMPNNLayer(MessagePassing):
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels: int, out_channels: int, hidden_size: int, _nn, dropout: float,
                 improved: bool = False, cached: bool = False,
                 bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(DMPNNLayer, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.improved = improved
        self.cached = cached

        self._cached_edge_index = None
        self._cached_adj_t = None
        self.nn = _nn
        # W_i = Linear(in_channels, out_channels, bias=False)
        self.W_i = Linear(self.in_channels, self.hidden_size, bias=False)
        self.W_h = Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_o = Linear(self.in_channels + self.hidden_size, self.out_channels, bias=True)
        self.act_func = ReLU()
        self.dropout_layer = nn.Dropout(p=self.dropout)
        # self.lin = Linear(in_channels, out_channels, bias=False,
        #                   weight_initializer='glorot')

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.W_i.reset_parameters()
        self.W_h.reset_parameters()
        self.W_o.reset_parameters()
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_attr: OptTensor = None) -> Tensor:
        self.f_atom = x
        input = self.W_i(x)
        x = self.act_func(input)
        # x = (x, x)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_i: Tensor, x_j: Tensor, index: Adj, edge_index_j: Adj, edge_attr: OptTensor) -> Tensor:
        x_edge_i = torch.cat([x_i, edge_attr], dim=-1)
        x_edge_i = self.nn(x_edge_i)

        x_message = scatter_add(x_edge_i, edge_index_j, dim=0)
        a_message = x_message.index_select(dim=0, index=edge_index_j) + x_j
        rev_message = x_message.index_select(dim=0, index=index) + x_i
        message = a_message - rev_message

        message = self.W_h(message)
        message = self.act_func(x_j + message)
        message = self.dropout_layer(message)
        return message

    # def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
    #     # return matmul(adj_t, x, reduce=self.aggr)
    #     pass
    def update(self, out):
        a_input = torch.cat([self.f_atom, out], dim=1)
        atom_hiddens = self.act_func(self.W_o(a_input))
        return self.dropout_layer(atom_hiddens)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class DMPNNAlphaLayer(MessagePassing):
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, a_in_channels: int, out_channels: int, hidden_size: int, b_in_channels: int, dropout: float,
                 improved: bool = False, cached: bool = False,
                 bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(DMPNNAlphaLayer, self).__init__(**kwargs)

        self.a_in_channels = a_in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.b_in_channels = b_in_channels
        self.dropout = dropout
        self.improved = improved
        self.cached = cached

        self._cached_edge_index = None
        self._cached_adj_t = None
        # W_i = Linear(in_channels, out_channels, bias=False)
        self.W_i = Linear(self.a_in_channels, self.hidden_size, bias=False)
        self.W_h = Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_o = Linear(self.a_in_channels + self.hidden_size, self.out_channels, bias=True)
        self.nn_node_edge = Sequential(Linear(self.a_in_channels + self.hidden_size, self.out_channels),
                                       ReLU())
        self.nn_edge = Sequential(Linear(self.b_in_channels, self.out_channels, bias=True),
                                  ReLU()
                                  )
        self.act_func = ReLU()
        self.dropout_layer = nn.Dropout(p=self.dropout)
        # self.lin = Linear(in_channels, out_channels, bias=False,
        #                   weight_initializer='glorot')

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.W_i.reset_parameters()
        self.W_h.reset_parameters()
        self.W_o.reset_parameters()
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_attr: OptTensor = None) -> Tensor:
        self.f_atom = x
        input = self.W_i(x)
        x = self.act_func(input)
        # x = (x, x)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_i: Tensor, x_j: Tensor, index: Adj, edge_index_j: Adj, edge_attr: OptTensor) -> Tensor:
        edge_attr = self.nn_edge(edge_attr)
        x_edge_i = torch.cat([x_i, edge_attr], dim=-1)
        x_edge_i = self.nn_node_edge(x_edge_i)

        x_message = scatter_add(x_edge_i, edge_index_j, dim=0)
        a_message = x_message.index_select(dim=0, index=edge_index_j) + x_j
        rev_message = x_message.index_select(dim=0, index=index) + x_i
        message = a_message - rev_message

        message = self.W_h(message)
        message = self.act_func(x_j + message)
        message = self.dropout_layer(message)
        return message

    # def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
    #     # return matmul(adj_t, x, reduce=self.aggr)
    #     pass
    def update(self, out):
        a_input = torch.cat([self.f_atom, out], dim=1)
        atom_hiddens = self.act_func(self.W_o(a_input))
        return self.dropout_layer(atom_hiddens)

    def __repr__(self):
        return '{}({}, {}, {})'.format(self.__class__.__name__, self.a_in_channels,
                                       self.out_channels, self.b_in_channels)
