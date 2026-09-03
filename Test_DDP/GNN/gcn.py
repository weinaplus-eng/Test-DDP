import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, args, in_dim, num_classes):
        super().__init__()
        self.dropout = getattr(args, 'dropout', 0.5)
        self.convs = nn.ModuleList()
        for i in range(args.layers):
            in_c = in_dim if i == 0 else args.hidden_dim
            self.convs.append(GCNConv(in_c, args.hidden_dim))
        self.classifier = nn.Linear(args.hidden_dim, num_classes)

    def forward(self, x, edge_index, edge_weight=None):
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x), x
