import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_geometric.nn.conv import SAGEConv, GCNConv, GATConv,TransformerConv,ResGatedGraphConv,CuGraphSAGEConv
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.pool import global_add_pool,SAGPooling
from torch_geometric.nn import BatchNorm,LayerNorm
from lsoftmax import LSoftmaxLinear

torch.manual_seed(128)
class SiameseNet(nn.Module):
    def __init__(self, embedding_net,embedding_size,hls_dim,drop_out=0.0, pool_ratio=0.2):
        super(SiameseNet, self).__init__()
        hidden_size = 32
        self.drop_out = drop_out
        self.pool_ratio = pool_ratio
        self.nonlinear = nn.ReLU()
        self.embedding_net = embedding_net
        self.channels = [3*embedding_size+hls_dim,hidden_size,hidden_size,1]

        margin = 1
        self.mlps = torch.nn.ModuleList()
        self.graph_norm = torch.nn.ModuleList()
        self.PRELU = torch.nn.PReLU()
        self.lsoftmax_linear = LSoftmaxLinear(
            input_dim=hidden_size, output_dim=2, margin=margin)

        for i in range(len(self.channels) - 1):
            fc = Linear(self.channels[i], self.channels[i + 1])
            self.mlps.append(fc)
            if i < len(self.channels):
                self.graph_norm.append(BatchNorm(self.channels[i+1]))


    def forward(self, x1, x2,edge_index1,edge_index2,batch1,batch2,hls_attr1,hls_attr2,target=None):
        output1 = self.embedding_net(x1, edge_index1, batch1)
        output2 = self.embedding_net(x2,edge_index2,batch2)
        diff_graph_embeddings = torch.sub(output1,output2)
        diff_hls_attr = torch.sub(hls_attr1,hls_attr2)
        x = torch.cat((diff_graph_embeddings,diff_hls_attr),1)
        x = self.PRELU(x)
        for f in range(len(self.mlps)):
            if f < len(self.mlps) - 1:
                x = self.mlps[f](x)
                x = F.relu(x)
                x = F.dropout(x, p=0.3, training=self.training)

            else:
                x = self.lsoftmax_linear(input=x, target=target)

        return x

    def get_embedding(self, x):
        return self.embedding_net(x)

class EmbeddingNet(nn.Module):
    def __init__(self,in_channels, hidden_channels, num_layers, conv_type,  drop_out=0.0, pool_ratio=0.8):
        super(EmbeddingNet, self).__init__()

        self.drop_out = drop_out
        self.pool_ratio = pool_ratio
        if conv_type == 'gcn':
            conv = GCNConv
        elif conv_type == 'gat':
            conv = GATConv
        elif conv_type == 'sage':
            conv = SAGEConv
        elif conv_type == "rgconv":
            conv = ResGatedGraphConv
        else:

            conv = TransformerConv

        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.graph_norms = torch.nn.ModuleList()
        self.layer_norms = torch.nn.ModuleList()
        self.num_layers = num_layers
        for i in range(num_layers):
            if i == 0:
                self.convs.append(conv(in_channels, hidden_channels))

            else:
                self.convs.append(conv(hidden_channels, hidden_channels))
            self.graph_norms.append(BatchNorm(hidden_channels))
            self.layer_norms.append(LayerNorm(hidden_channels,affine=True))
            self.pools.append(SAGPooling(self.num_layers*hidden_channels, self.pool_ratio))







    def forward(self, x, edge_index, batch):
        x = x.to(torch.float32)
        h = list()
        for step in range(len(self.convs)):
            x = self.convs[step](x, edge_index)
            x = self.layer_norms[step](x)
            x = F.relu(x)
            h.append(x)
        x = h[0]
        for i in range(1,self.num_layers):
            x = torch.cat([x,h[i]],dim=1)
        x, edge_index, _, batch, _, _ = self.pools[0](x, edge_index, None, batch, None)
        x = global_add_pool(x, batch)
        return x

def ModelConfig(num_features,device):
    embedding_size = 16
    embeddingNet = EmbeddingNet(in_channels=num_features, hidden_channels=embedding_size, num_layers=3,
                                conv_type='trans',
                                drop_out=0.4)
    model = SiameseNet(embeddingNet, hls_dim=6, embedding_size=embedding_size)
    model = model.to(device)
    return model

