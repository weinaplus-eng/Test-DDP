import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    def __init__(self, args, in_dim, num_classes):
        super().__init__()
        self.dropout = getattr(args, 'dropout', 0.5)
        heads = getattr(args, 'heads', 8)
        self.convs = nn.ModuleList()
        for i in range(args.layers):
            in_c = in_dim if i == 0 else args.hidden_dim * heads
            self.convs.append(GATConv(in_c, args.hidden_dim, heads=heads, dropout=self.dropout))
        self.classifier = nn.Linear(args.hidden_dim * heads, num_classes)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x), x
