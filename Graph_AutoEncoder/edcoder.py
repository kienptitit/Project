from typing import Optional
from itertools import chain
from functools import partial

import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
from Graph_AutoEncoder.GAT import GAT
import dgl
import numpy as np
from torch.utils.data import DataLoader, Dataset


def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    else:
        return nn.Identity


def setup_module(m_type, enc_dec, in_dim, num_hidden, out_dim, num_layers, dropout, activation, residual, norm, nhead,
                 nhead_out, attn_drop, negative_slope=0.2, concat_out=True) -> nn.Module:
    if m_type == "gat":
        mod = GAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )

    return mod


def mask_edge(graph, mask_prob):
    E = graph.num_edges()

    mask_rates = torch.FloatTensor(np.ones(E) * mask_prob)
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx


def drop_edge(graph, drop_rate, return_edges=False):
    if drop_rate <= 0:
        return graph

    n_node = graph.num_nodes()
    edge_mask = mask_edge(graph, drop_rate)
    src = graph.edges()[0]
    dst = graph.edges()[1]

    nsrc = src[edge_mask]
    ndst = dst[edge_mask]

    ng = dgl.graph((nsrc, ndst), num_nodes=n_node)
    ng = ng.add_self_loop()

    dsrc = src[~edge_mask]
    ddst = dst[~edge_mask]

    if return_edges:
        return ng, (dsrc, ddst)
    return ng


def sce_loss(x, y, alpha=1):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss_mean = loss.mean()
    return loss_mean, loss


class PreModel(nn.Module):
    def __init__(
            self,
            in_dim: int,
            num_hidden: int,
            num_layers: int,
            nhead: int,
            nhead_out: int,
            activation: str,
            feat_drop: float,
            attn_drop: float,
            negative_slope: float,
            residual: bool,
            norm: Optional[str],
            mask_rate: float = 0.3,
            encoder_type: str = "gat",
            decoder_type: str = "gat",
            loss_fn: str = "sce",
            drop_edge_rate: float = 0.0,
            replace_rate: float = 0.1,
            alpha_l: float = 2,
            concat_hidden: bool = False,
    ):
        super(PreModel, self).__init__()
        self._mask_rate = mask_rate

        self._encoder_type = encoder_type
        self._decoder_type = decoder_type
        self._drop_edge_rate = drop_edge_rate
        self._output_hidden_size = num_hidden
        self._concat_hidden = concat_hidden

        self._replace_rate = replace_rate
        self._mask_token_rate = 1 - self._replace_rate

        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0
        if encoder_type in ("gat", "dotgat"):
            enc_num_hidden = num_hidden // nhead
            enc_nhead = nhead
        else:
            enc_num_hidden = num_hidden
            enc_nhead = 1

        dec_in_dim = num_hidden
        dec_num_hidden = num_hidden // nhead_out if decoder_type in ("gat", "dotgat") else num_hidden

        # build encoder
        self.encoder = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
        )

        # build decoder for attribute prediction
        self.decoder = setup_module(
            m_type=decoder_type,
            enc_dec="decoding",
            in_dim=dec_in_dim,
            num_hidden=dec_num_hidden,
            out_dim=in_dim,
            num_layers=1,
            nhead=nhead,
            nhead_out=nhead_out,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            concat_out=False,
        )

        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        if concat_hidden:
            self.encoder_to_decoder = nn.Linear(dec_in_dim * num_layers, dec_in_dim, bias=False)
        else:
            self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)

        # * setup loss function
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)

    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    def encoding_mask_noise(self, g, x, mask_rate=0.3):
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        if self._replace_rate > 0:
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token
        use_g = g.clone()

        return use_g, out_x, (mask_nodes, keep_nodes)

    def forward(self, g, x):
        # ---- attribute reconstruction ----
        loss, res_feat = self.mask_attr_prediction(g, x)
        loss_item = {"loss": loss.item()}
        return loss, loss_item

    def forward_test(self, x):
        self.eval()

        def similarity_compute(feat):
            return F.cosine_similarity(feat.unsqueeze(1), feat.unsqueeze(0), dim=-1)

        def make_graph_topology(feat):
            num_nodes = feat.shape[0]
            node_h = []
            node_t = []
            for i in range(num_nodes):
                node_h += [i] * (num_nodes - 1)
                node_t += [j for j in list(range(num_nodes)) if j != i]
            return dgl.graph((node_h, node_t))

        def make_feature(feat, topk):
            """
            :param feat:[98,256]
            :return: [98,98,256]
            """
            similarity_matrix = similarity_compute(feat)
            selected_idx = torch.topk(similarity_matrix, k=topk, dim=-1).indices
            out = []
            for i in range(feat.shape[0]):
                feat_cop = feat.clone()
                feat_cop[selected_idx[i]] = 0.0
                out.append(feat_cop)
            return torch.stack(out)

        def make_graph_for_test(feat):
            graph = make_graph_topology(feat)
            feat_modified = make_feature(feat, topk=5)
            all_graph = []
            for i in range(feat.shape[0]):
                copy_graph = copy.deepcopy(graph)
                copy_graph.ndata['x'] = feat_modified[i]
                all_graph.append(copy_graph)
            return all_graph

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

        def test_each_graph():
            graphs_list = make_graph_for_test(x)
            score = []
            for idx, g in enumerate(graphs_list):
                feat = g.ndata['x']
                enc_rep, all_hidden = self.encoder(g, feat, return_hidden=True)
                if self._concat_hidden:
                    enc_rep = torch.cat(all_hidden, dim=1)
                # ---- attribute reconstruction ----
                rep = self.encoder_to_decoder(enc_rep)
                recon = self.decoder(g, rep)  # [batch_size*num_nodes,dim]
                score.append(sce_loss(x[idx], recon[idx])[0].detach().item())
            return score

        def test_batch_graph():
            batch_size = 16
            feature_dim = x.shape[-1]
            num_nodes = x.shape[0]
            graphs_list = make_graph_for_test(x)
            graph_Data = Mydataset(graphs_list)
            graph_loader = DataLoader(graph_Data, batch_size=batch_size, shuffle=False, collate_fn=collate)
            score = []
            for batch_idx, g in enumerate(graph_loader):
                idx = torch.tensor(list(range(batch_idx * batch_size, min((batch_idx + 1) * batch_size, num_nodes))))
                idx_s = idx.unsqueeze(1).unsqueeze(-1).repeat(1, 1, feature_dim)
                feat = g.ndata['x']
                enc_rep, all_hidden = self.encoder(g, feat, return_hidden=True)
                if self._concat_hidden:
                    enc_rep = torch.cat(all_hidden, dim=1)
                # ---- attribute reconstruction ----
                rep = self.encoder_to_decoder(enc_rep)
                recon = self.decoder(g, rep)
                recon = recon.reshape(-1, num_nodes, feature_dim)

                score += sce_loss(x[idx], torch.gather(recon, dim=1, index=idx_s).squeeze(1))[
                    1].detach().cpu().numpy().tolist()
            return score

        return test_batch_graph()

    def mask_attr_prediction(self, g, x):
        pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, x, self._mask_rate)

        if self._drop_edge_rate > 0:
            use_g, masked_edges = drop_edge(pre_use_g, self._drop_edge_rate, return_edges=True)
        else:
            use_g = pre_use_g

        enc_rep, all_hidden = self.encoder(use_g, use_x, return_hidden=True)

        if self._concat_hidden:
            enc_rep = torch.cat(all_hidden, dim=1)

        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)

        if self._decoder_type not in ("mlp", "linear"):
            # * remask, re-mask
            rep[mask_nodes] = 0
        if self._decoder_type in ("mlp", "liear"):
            recon = self.decoder(rep)
        else:
            recon = self.decoder(pre_use_g, rep)

        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]

        loss = self.criterion(x_rec, x_init)
        return loss

    def embed(self, g, x):
        rep = self.encoder(g, x)
        return rep

    @property
    def enc_params(self):
        return self.encoder.parameters()

    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])


if __name__ == '__main__':
    g = dgl.graph(([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2]))
    feat = torch.randn(4, 3)
    model = PreModel(3, 3, 1, 1, 1, "relu", 0.2, 0.2, 0.01, True, "layernorm", replace_rate=0.0)
    print(model(g, feat))
