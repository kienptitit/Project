import torch
from Phase1.model import Model
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os
from Phase1.utils import Mydata
import dgl
import copy
from edcoder import PreModel
import matplotlib.pyplot as plt


def make_graph_topology(feat):
    num_nodes = feat.shape[0]
    node_h = []
    node_t = []
    for i in range(num_nodes):
        node_h += [i] * (num_nodes - 1)
        node_t += [j for j in list(range(num_nodes)) if j != i]
    return dgl.graph((node_h, node_t)).to(device)


def collate(samples):
    batched_graph = dgl.batch(samples)
    return batched_graph


class Graph(Dataset):
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


def get_data_normal(feature_path):
    output = []
    for file_name in os.listdir(feature_path):
        if 'abnormal' not in file_name:
            file_path = os.path.join(feature_path, file_name)
            output.append(np.load(file_path))
    return torch.from_numpy(np.stack(output))


def get_model():
    model = Model()
    model.load_state_dict(torch.load(r"E:\Python test Work\HopingProject\Weight\model.pt"))
    for name, param in model.named_parameters():
        param.requires_grad = False
    return model


def train_res(epoch, data_loader, model1, pre_model, optimizer):
    pre_model.train()
    l = 0.0
    model1.eval()
    for batch_data, data in enumerate(data_loader):
        data = data.to(device)
        _, data_phase1 = model1(data.float())
        graph_list = make_data_train(data_phase1)
        graph_data = Graph(graph_list)
        graph_loader = DataLoader(graph_data, batch_size=4, shuffle=False, collate_fn=collate)
        losses = 0.0
        for batch_graph, g in enumerate(graph_loader):
            feat = g.ndata['x']
            loss, _ = pre_model(g, feat)
            optimizer.zero_grad()
            loss.backward()
            losses += loss.detach().cpu().item()
            optimizer.step()
        print("Epoch {} Batch data {} Avg_loss {:.2f}".format(epoch, batch_data, losses / len(graph_loader)))
        l += losses / len(graph_loader)
    return l / len(data_loader)


if __name__ == '__main__':
    # Model
    device = torch.device("cuda")
    model1 = get_model().to(device)
    Graph_model = PreModel(256, 128, 4, 4, 4, "prelu", 0.1, 0.1, 0.1, True, "layernorm", mask_rate=0.1, alpha_l=2).to(
        device)
    # Data
    data_normal = get_data_normal(r"E:\Python test Work\HopingProject\New_feature_swin3-4")
    mydata_graph = Mydata(data_normal)
    dataloader = DataLoader(mydata_graph, shuffle=True, batch_size=4, num_workers=2)
    # Config
    epochs = 5
    optimizer = torch.optim.Adam(Graph_model.parameters(), lr=0.001)
    # Training resconstruct
    losses = []
    for epoch in range(epochs):
        losses.append(train_res(epoch, dataloader, model1, Graph_model, optimizer))
    torch.save(Graph_model.state_dict(), r"E:\Python test Work\HopingProject\Weight\graph_weight.pt")
