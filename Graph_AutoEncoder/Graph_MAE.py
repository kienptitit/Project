import time

import dgl
import torch
from sklearn.preprocessing import StandardScaler
from edcoder import PreModel
from tqdm import tqdm
import torch.nn as nn
import copy
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def make_graph_topology(feat):
    num_nodes = feat.shape[0]
    node_h = []
    node_t = []
    for i in range(num_nodes):
        node_h += [i] * (num_nodes - 1)
        node_t += [j for j in list(range(num_nodes)) if j != i]
    return dgl.graph((node_h, node_t))


def collate(samples):
    batched_graph = dgl.batch(samples)
    return batched_graph


class Mydataset(Dataset):
    def __init__(self, graphs):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


def make_data_train(inp):
    """
    :param inp:[4,15,98,256]
    :return: [List of dgl graph]
    """
    num_patches = inp.shape[-2]
    num_features = inp.shape[-1]
    inp = inp.reshape(-1, num_patches, num_features)
    global_graph_topology = make_graph_topology(inp[0])
    out = []
    for graph_feat in inp:
        copy_topology = copy.deepcopy(global_graph_topology)
        copy_topology.ndata['x'] = graph_feat
        out.append(copy_topology)
    return out


def train(model, graph, feat, optimizer, max_epochs, device):
    graph = graph.to(device)
    feat = feat.to(device)
    epoch_iter = tqdm(range(max_epochs))
    for epoch in epoch_iter:
        model.train()

        loss, loss_dict = model(graph, feat)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss.item():.4f}")


if __name__ == '__main__':
    # Test data

    # Graph
    inp = torch.randn(4, 15, 98, 256)
    graphs_list = make_data_train(inp)
    graph_data = Mydataset(graphs_list)
    graph_data_loader = DataLoader(graph_data, batch_size=5, shuffle=False, collate_fn=collate)
    num_features = inp.shape[-1]
    # Config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PreModel(1433, 512, 2, 4, 4, "prelu", 0.2, 0.1, 0.1, True, "batchnorm", mask_rate=0.3)
    x = torch.randn(98, 1433)
    print(model.forward_test(x))
    # model = model.to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=2e-4)
    # max_epoch = 1500
    # # Training
    # train(model, graph, feat, optimizer, max_epoch, device)
