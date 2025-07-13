import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        # seq: [N, F] or [1, N, F]
        # adj: [N, N] or [1, N, N]
        if get_dim(seq) == 3 and seq.shape[0] == 1:
            seq = seq.squeeze(0)
        if get_dim(adj) == 3 and adj.shape[0] == 1:
            adj = adj.squeeze(0)
        seq_fts = self.fc(seq)
        if sparse:
            # 保证adj和seq_fts都是2D
            out = torch.spmm(adj, seq_fts)
            # 兼容性unsqueeze
            if hasattr(out, 'unsqueeze'):
                out = out.unsqueeze(0)
            elif isinstance(out, np.ndarray):
                out = np.expand_dims(out, axis=0)
            elif 'scipy' in str(type(out)):
                out = out[np.newaxis, ...]
            else:
                raise TypeError('Unsupported out type for unsqueeze')
        else:
            if get_dim(adj) == 2:
                if hasattr(adj, 'unsqueeze'):
                    adj = adj.unsqueeze(0)
                elif isinstance(adj, np.ndarray):
                    adj = np.expand_dims(adj, axis=0)
                elif 'scipy' in str(type(adj)):
                    adj = adj[np.newaxis, ...]
                else:
                    raise TypeError('Unsupported adj type for unsqueeze')
            if get_dim(seq_fts) == 2:
                if hasattr(seq_fts, 'unsqueeze'):
                    seq_fts = seq_fts.unsqueeze(0)
                elif isinstance(seq_fts, np.ndarray):
                    seq_fts = np.expand_dims(seq_fts, axis=0)
                elif 'scipy' in str(type(seq_fts)):
                    seq_fts = seq_fts[np.newaxis, ...]
                else:
                    raise TypeError('Unsupported seq_fts type for unsqueeze')
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        return self.act(out)


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 1)


class MaxReadout(nn.Module):
    def __init__(self):
        super(MaxReadout, self).__init__()

    def forward(self, seq):
        return torch.max(seq, 1).values


class MinReadout(nn.Module):
    def __init__(self):
        super(MinReadout, self).__init__()

    def forward(self, seq):
        return torch.min(seq, 1).values


class WSReadout(nn.Module):
    def __init__(self):
        super(WSReadout, self).__init__()

    def forward(self, seq, query):
        query = query.permute(0, 2, 1)
        sim = torch.matmul(seq, query)
        sim = F.softmax(sim, dim=1)
        sim = sim.repeat(1, 1, 64)
        out = torch.mul(seq, sim)
        out = torch.sum(out, 1)
        return out


class Discriminator(nn.Module):
    def __init__(self, n_h, negsamp_round):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

        self.negsamp_round = negsamp_round

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl):
        scs = []
        # positive
        scs.append(self.f_k(h_pl, c))

        # negative
        c_mi = c
        for _ in range(self.negsamp_round):
            c_mi = torch.cat((c_mi[-2:-1, :], c_mi[:-1, :]), 0)
            scs.append(self.f_k(h_pl, c_mi))

        logits = torch.cat(tuple(scs))

        return logits


class Model(nn.Module):
    def __init__(self, n_in, n_h, activation, negsamp_round, readout, adj_csr=None):
        super(Model, self).__init__()
        self.read_mode = readout
        self.gcn1 = GCN(n_in, n_h, activation)
        self.gcn2 = GCN(n_h, n_h, activation)
        self.gcn3 = GCN(n_h, n_h, activation)
        self.fc1 = nn.Linear(n_h, int(n_h / 2), bias=False)
        self.fc2 = nn.Linear(int(n_h / 2), int(n_h / 4), bias=False)
        self.fc3 = nn.Linear(int(n_h / 4), 1, bias=False)
        self.fc4 = nn.Linear(n_h, n_h, bias=False)
        self.fc6 = nn.Linear(n_h, n_h, bias=False)
        self.fc5 = nn.Linear(n_h, n_in, bias=False)
        self.act = nn.ReLU()
        if readout == 'max':
            self.read = MaxReadout()
        elif readout == 'min':
            self.read = MinReadout()
        elif readout == 'avg':
            self.read = AvgReadout()
        elif readout == 'weighted_sum':
            self.read = WSReadout()

        self.disc = Discriminator(n_h, negsamp_round)
        self.adj_csr = adj_csr

    def forward(self, seq1, adj, sample_abnormal_idx, normal_idx, train_flag, args, sparse=False):
        h_1 = self.gcn1(seq1, adj, sparse)
        # emb = h_1
        emb = self.gcn2(h_1, adj, sparse)


        emb_con = None
        emb_combine = None
        emb_abnormal = emb[:, sample_abnormal_idx, :]

        noise = torch.randn(emb_abnormal.size()) * args.var + args.mean
        emb_abnormal = emb_abnormal + noise
        # emb_abnormal = emb_abnormal + noise.cuda()
        if train_flag:
            # Add noise into the attribute of sampled abnormal nodes
            # degree = torch.sum(raw_adj[0, :, :], 0)[sample_abnormal_idx]
            # neigh_adj = raw_adj[0, sample_abnormal_idx, :] / torch.unsqueeze(degree, 1)

            # 大图用scipy.sparse高效索引，彻底避免OOM
            import numpy as np
            if self.adj_csr is not None:
                emb_con_list = []
                device = emb.device
                import gc
                print(f'逐节点聚合开始，异常节点总数: {len(sample_abnormal_idx)}')
                for i, idx in enumerate(sample_abnormal_idx):
                    row = self.adj_csr.getrow(idx)
                    neighbors = row.indices
                    weights = row.data
                    max_neighbors = 100
                    if len(neighbors) > max_neighbors:
                        neighbors = neighbors[:max_neighbors]
                        weights = weights[:max_neighbors]
                    if len(neighbors) == 0:
                        emb_con_list.append(torch.zeros(emb.shape[-1]))
                    else:
                        emb_neighbors = emb[0, neighbors, :].cpu()  # 强制在CPU
                        weights_tensor = torch.from_numpy(weights).float()
                        emb_con = (emb_neighbors * weights_tensor.unsqueeze(1)).sum(0) / (weights_tensor.sum() + 1e-8)
                        emb_con_list.append(emb_con)
                    print(f'Processed abnormal node {i+1}/{len(sample_abnormal_idx)}')
                    if (i+1) % 10 == 0:
                        import gc
                        gc.collect()
                print('逐节点聚合完成')
                emb_con = torch.stack(emb_con_list, dim=0)
                emb_con = self.act(self.fc4(emb_con))
            else:
                # 小图可用稠密
                if hasattr(adj, 'is_sparse') and adj.is_sparse:
                    adj_dense = adj.to_dense()
                else:
                    adj_dense = adj
                neigh_adj = adj_dense[sample_abnormal_idx, :]
                emb_con = torch.mm(neigh_adj, emb[0, :, :])
            emb_con = self.act(self.fc4(emb_con))
            # emb_con = self.act(self.fc6(emb_con))

            emb_combine = torch.cat((emb[:, normal_idx, :], torch.unsqueeze(emb_con, 0)), 1)

            # TODO ablation study add noise on the selected nodes

            # std = 0.01
            # mean = 0.02
            # noise = torch.randn(emb[:, sample_abnormal_idx, :].size()) * std + mean
            # emb_combine = torch.cat((emb[:, normal_idx, :], emb[:, sample_abnormal_idx, :] + noise), 1)

            # TODO ablation study generate outlier from random noise
            # std = 0.01
            # mean = 0.02
            # emb_con = torch.mm(neigh_adj, emb[0, :, :])
            # noise = torch.randn(emb_con.size()) * std + mean
            # emb_con = self.act(self.fc4(noise))
            # emb_combine = torch.cat((emb[:, normal_idx, :], torch.unsqueeze(emb_con, 0)), 1)

            f_1 = self.fc1(emb_combine)
            f_1 = self.act(f_1)
            f_2 = self.fc2(f_1)
            f_2 = self.act(f_2)
            f_3 = self.fc3(f_2)
            # f_3 = torch.sigmoid(f_3)
            emb[:, sample_abnormal_idx, :] = emb_con
        else:
            f_1 = self.fc1(emb)
            f_1 = self.act(f_1)
            f_2 = self.fc2(f_1)
            f_2 = self.act(f_2)
            f_3 = self.fc3(f_2)
            # f_3 = torch.sigmoid(f_3)

        return emb, emb_combine, f_3, emb_con, emb_abnormal

# 兼容多类型的dim判断函数
def get_dim(x):
    if hasattr(x, 'dim'):
        return x.dim()
    elif hasattr(x, 'ndim'):
        return x.ndim
    elif hasattr(x, 'shape'):
        return len(x.shape)
    else:
        raise TypeError('Unknown type for adj')
