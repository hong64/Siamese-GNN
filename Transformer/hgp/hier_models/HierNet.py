

import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import SAGEConv, GCNConv, GATConv
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.models import JumpingKnowledge
from torch_geometric.nn.pool import global_add_pool, global_max_pool, SAGPooling


target = ['lut', 'ff', 'dsp', 'bram', 'uram', 'srl', 'cp', 'power']
tar_idx = 3
jknFlag = 0


class HierNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, conv_type, hls_dim, drop_out=0.0, pool_ratio=0.5):
        super(HierNet, self).__init__()

        self.drop_out = drop_out
        self.pool_ratio = pool_ratio
        if conv_type == 'gcn':
            conv = GCNConv
        elif conv_type == 'gat':
            conv = GATConv
        elif conv_type == 'sage':
            conv = SAGEConv
        else:
            conv = GCNConv

        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.convs.append(conv(in_channels, hidden_channels))
            else:
                self.convs.append(conv(hidden_channels, hidden_channels))
            self.pools.append(SAGPooling(hidden_channels, self.pool_ratio))
        if jknFlag:
            self.jkn = JumpingKnowledge('lstm', channels=hidden_channels, num_layers=2)

        self.global_pool = global_add_pool
        self.channels = [hidden_channels * 2 + hls_dim, 64, 64, 1]
        self.mlps = torch.nn.ModuleList()

        for i in range(len(self.channels) - 1):
            fc = Linear(self.channels[i], self.channels[i + 1])
            self.mlps.append(fc)

    def forward(self, x, edge_index, batch, hls_attr):

        x = x.to(torch.float32)
        h_list = []

        for step in range(len(self.convs)):
            x = self.convs[step](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_out, training=self.training)
            x, edge_index, _, batch, _, _ = self.pools[step](x, edge_index, None, batch, None)
            h = torch.cat([global_max_pool(x, batch), global_add_pool(x, batch)], dim=1)
            h_list.append(h)

        if jknFlag:
            x = self.jkn(h_list)
        x = h_list[0] + h_list[1] + h_list[2]
        x = torch.cat([x, hls_attr], dim=-1)

        for f in range(len(self.mlps)):
            if f < len(self.mlps) - 1:
                x = F.relu(self.mlps[f](x))
                x = F.dropout(x, p=self.drop_out, training=self.training)
            else:
                x = self.mlps[f](x)

        return x





