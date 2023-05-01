import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class MLP(nn.Module):
    def __init__(self, in_feature, hidden_feature, out_feature, num_layers=4):
        super().__init__()
        self.linears = nn.ModuleList([
            nn.Linear(in_feature, hidden_feature) if i == 0
            else nn.Linear(hidden_feature, hidden_feature)
            for i in range(num_layers)
        ]
        )
        self.bn1 = nn.BatchNorm1d(hidden_feature)
        self.bn2 = nn.BatchNorm1d(out_feature)
        self.out = nn.Linear(hidden_feature, out_feature)
        self.out_feature = out_feature
        self.in_feature = in_feature
        self.patches = 98

    def forward(self, x):
        """
        :param x: [4,15,98,1024]
        :return: [4,15,98,256]
        """
        b = x.shape[0]
        num_windows = x.shape[2]
        out = x.view(-1, num_windows, self.in_feature)
        for linear in self.linears:
            out = linear(out)
            out = self.bn1(out.transpose(1, 2)).transpose(1, 2)
            out = F.leaky_relu(out, negative_slope=0.02)
            out = F.dropout(out, p=0.5)
        return F.leaky_relu(self.bn2(self.out(out).transpose(1, 2)).transpose(1, 2), negative_slope=0.02).view(b, -1,
                                                                                                               num_windows,
                                                                                                               self.out_feature)


class Model(nn.Module):
    def __init__(self, in_feature=1024, hidden_feature=512, out_feature=256, hidden_2=128, out_2=64):
        super(Model, self).__init__()
        self.mlp = MLP(in_feature, hidden_feature, out_feature)
        self.mlp2 = MLP(out_feature, hidden_2, out_2)
        self.out2 = out_2
        self.init_weights()

    def forward(self, x):
        out = self.mlp(x)  # [4,15,98,256]
        avg_magnitude = out.norm(p=2, dim=-1).mean(dim=-1)  # [4,15]
        topk_avg_mag = avg_magnitude.topk(k=3, dim=-1).indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, out.shape[2],
                                                                                                  out.shape[3])
        out_selected = self.mlp2(torch.gather(out, dim=1, index=topk_avg_mag))
        idx_out_selected = out_selected.norm(p=2, dim=-1).topk(k=3, dim=-1).indices.unsqueeze(-1).expand(-1, -1, -1,
                                                                                                         self.out2)
        out2 = torch.gather(out_selected, dim=2, index=idx_out_selected)
        return out2, out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)


if __name__ == '__main__':
    inp = torch.randn(4, 15, 98, 1024)
    m = Model()
    out2, out = m(inp)
    print(out2.shape)
    print(out.shape)
