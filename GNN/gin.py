import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv


class GIN(nn.Module):
    def __init__(self, args, in_dim, num_classes):
        super().__init__()
        hidden = args.hidden_dim
        self.dropout = getattr(args, 'dropout', 0.5)
        self.convs = nn.ModuleList()
        for i in range(args.layers):
            in_c = in_dim if i == 0 else hidden
            mlp = nn.Sequential(nn.Linear(in_c, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.convs.append(GINConv(mlp))
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, x, edge_index, edge_weight=None):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x), x
