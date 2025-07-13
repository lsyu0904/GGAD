import torch.nn as nn

from model import Model
from utils import *

from sklearn.metrics import roc_auc_score
import random
import dgl
from sklearn.metrics import average_precision_score
import argparse
from tqdm import tqdm
import time
import scipy.sparse as sp
import numpy as np

# 参数字典和默认值
DATASET_NUM_EPOCH = {
    'amazon': 800,
    'tfinance': 500,
    'reddit': 300,
    'elliptic': 150,
    'photo': 100,
}
DEFAULT_LR = 1e-3
DEFAULT_NUM_EPOCH = 300
DEFAULT_MEAN = 0.0
DEFAULT_VAR = 0.0
SPECIAL_MEAN_VAR = {
    'reddit': (0.02, 0.01),
    'photo': (0.02, 0.01),
}

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [3]))
# os.environ["KMP_DUPLICATE_LnIB_OK"] = "TRUE"
# Set argument
parser = argparse.ArgumentParser(description='')

parser.add_argument('--dataset', type=str,
                    default='reddit')
parser.add_argument('--lr', type=float)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--embedding_dim', type=int, default=300)
parser.add_argument('--num_epoch', type=int)
parser.add_argument('--drop_prob', type=float, default=0.0)
parser.add_argument('--readout', type=str, default='avg')  # max min avg  weighted_sum
parser.add_argument('--auc_test_rounds', type=int, default=256)
parser.add_argument('--negsamp_ratio', type=int, default=1)
parser.add_argument('--mean', type=float, default=0.0)
parser.add_argument('--var', type=float, default=0.0)



args = parser.parse_args()

args.lr = args.lr or DEFAULT_LR
args.num_epoch = args.num_epoch or DATASET_NUM_EPOCH.get(args.dataset, DEFAULT_NUM_EPOCH)
args.mean, args.var = SPECIAL_MEAN_VAR.get(args.dataset, (DEFAULT_MEAN, DEFAULT_VAR))

print('Dataset: ', args.dataset)

# Set random seed
dgl.random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
# os.environ['PYTHONHASHSEED'] = str(args.seed)
# os.environ['OMP_NUM_THREADS'] = '1'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load and preprocess data
adj, features, labels, all_idx, idx_train, idx_val, \
idx_test, ano_label, str_ano_label, attr_ano_label, normal_label_idx, abnormal_label_idx = load_mat(args.dataset, max_nodes=1000)

# 保持特征和邻接矩阵为稀疏格式
if not sp.issparse(features):
    features = sp.csr_matrix(features)
if not sp.issparse(adj):
    adj = sp.csr_matrix(adj)

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

# 邻接矩阵加单位阵，保持稀疏
raw_adj = adj.copy()
raw_adj = raw_adj + sp.eye(raw_adj.shape[0])
adj = adj + sp.eye(adj.shape[0])

# 全局自动修复：大邻接矩阵禁止稠密化
adj_is_large = adj.shape[0] > 100000
sparse_flag = False  # 统一初始化，防止未定义

def safe_toarray(mat):
    if hasattr(mat, 'toarray') and mat.shape[0] <= 100000:
        return mat.toarray()
    return mat

def safe_todense(mat):
    if hasattr(mat, 'todense') and mat.shape[0] <= 100000:
        return mat.todense()
    return mat

# 后续所有对 adj、raw_adj 的稠密化操作都用 safe_toarray/safe_todense
# 例如：
# adj = adj.toarray()  ->  adj = safe_toarray(adj)
# raw_adj = raw_adj.todense()  ->  raw_adj = safe_todense(raw_adj)

# 判断邻接矩阵大小，超过10万节点用稀疏格式
# adj_is_large = adj.shape[0] > 100000

def scipy_sparse_to_torch_sparse(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

# 删除所有对大邻接矩阵的 torch.FloatTensor 操作，只允许小数据集分支：
# 只保留如下结构：
if adj_is_large:
    adj_torch = scipy_sparse_to_torch_sparse(adj)
    raw_adj_torch = scipy_sparse_to_torch_sparse(raw_adj)
    sparse_flag = True
else:
    if not isinstance(adj, torch.Tensor):
        adj = safe_toarray(adj)
    if not isinstance(raw_adj, torch.Tensor):
        raw_adj = safe_toarray(raw_adj)
    if get_dim(adj) == 2:
        if hasattr(adj, 'unsqueeze'):
            adj_torch = adj.unsqueeze(0)
        elif isinstance(adj, np.ndarray):
            adj_torch = np.expand_dims(adj, axis=0)
        elif 'scipy' in str(type(adj)):
            adj_torch = adj[np.newaxis, ...]
        else:
            raise TypeError('Unsupported adj type for unsqueeze')
    else:
        adj_torch = adj

    if get_dim(raw_adj) == 2:
        if hasattr(raw_adj, 'unsqueeze'):
            raw_adj_torch = raw_adj.unsqueeze(0)
        elif isinstance(raw_adj, np.ndarray):
            raw_adj_torch = np.expand_dims(raw_adj, axis=0)
        elif 'scipy' in str(type(raw_adj)):
            raw_adj_torch = raw_adj[np.newaxis, ...]
        else:
            raise TypeError('Unsupported raw_adj type for unsqueeze')
    else:
        raw_adj_torch = raw_adj
# 后续所有模型输入、loss、邻接相关操作都只用 adj_torch/raw_adj_torch，不再用原始 adj/raw_adj
adj = None
raw_adj = None

# 特征转为稠密张量
if not isinstance(features, torch.Tensor):
    if hasattr(features, 'toarray'):
        features = features.toarray()
    features = torch.FloatTensor(features)
    features = features.squeeze()
    if features.dim() > 2:
        features = features.squeeze()
    if features.shape[0] < features.shape[1]:
        print('Auto transpose features!')
        features = features.T
print('features.shape (final):', features.shape)
ft_size = features.shape[1]
print('ft_size (final):', ft_size)
print('features.shape (before model init):', features.shape)
print('ft_size (before model init):', features.shape[1])
# Initialize model and optimiser
model = Model(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout, adj_csr=adj if adj_is_large else None)
optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#
# if torch.cuda.is_available():
#     print('Using CUDA')
#     model.cuda()
#     features = features.cuda()
#     adj = adj.cuda()
#     labels = labels.cuda()
#     raw_adj = raw_adj.cuda()

# idx_train = idx_train.cuda()
# idx_val = idx_val.cuda()
# idx_test = idx_test.cuda()
#
# if torch.cuda.is_available():
#     b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]).cuda())
# else:
#     b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]))

b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]))
xent = nn.CrossEntropyLoss()


# Train model
with tqdm(total=args.num_epoch) as pbar:
    pbar.set_description('Training')
    total_time = 0
    for epoch in range(args.num_epoch):
        start_time = time.time()
        model.train()
        optimiser.zero_grad()

        # Train model
        train_flag = True
        # 在模型调用处，传递 sparse_flag 和 adj_torch/raw_adj_torch
        features = features.squeeze()
        print('features.shape (input to model):', features.shape)
        emb, emb_combine, logits, emb_con, emb_abnormal = model(features, adj_torch, abnormal_label_idx, normal_label_idx, train_flag, args, sparse=sparse_flag)
        print('emb.shape (output from model):', emb.shape)
        if epoch % 10 == 0:
            # save data for tsne
            pass

            # tsne_data_path = 'draw/tfinance/tsne_data_{}.mat'.format(str(epoch))
            # io.savemat(tsne_data_path, {'emb': np.array(emb.cpu().detach()), 'ano_label': ano_label,
            #                             'abnormal_label_idx': np.array(abnormal_label_idx),
            #                             'normal_label_idx': np.array(normal_label_idx)})

        # BCE loss
        lbl = torch.unsqueeze(torch.cat(
            (torch.zeros(len(normal_label_idx)), torch.ones(len(emb_con)))),
            1).unsqueeze(0)
        # if torch.cuda.is_available():
        #     lbl = lbl.cuda()

        loss_bce = b_xent(logits, lbl)
        loss_bce = torch.mean(loss_bce)

        # Local affinity margin loss
        emb = torch.squeeze(emb)

        emb_inf = torch.norm(emb, dim=-1, keepdim=True)
        emb_inf = torch.pow(emb_inf, -1)
        emb_inf[torch.isinf(emb_inf)] = 0.
        emb_norm = emb * emb_inf

        sim_matrix = torch.mm(emb_norm, emb_norm.T)
        # 在loss、相似度等所有地方，只用 adj_torch/raw_adj_torch，不再用 raw_adj.sqeeze() 等
        # 例如：
        similar_matrix = sim_matrix * raw_adj_torch.squeeze()  # 保证 raw_adj_torch 是正确的 shape

        r_inv = torch.pow(torch.sum(raw_adj, 0), -1)
        r_inv[torch.isinf(r_inv)] = 0.
        affinity = torch.sum(similar_matrix, 0) * r_inv

        affinity_normal_mean = torch.mean(affinity[normal_label_idx])
        affinity_abnormal_mean = torch.mean(affinity[abnormal_label_idx])

        # if epoch % 10 == 0:
        #     real_abnormal_label_idx = np.array(all_idx)[np.argwhere(ano_label == 1).squeeze()].tolist()
        #     real_normal_label_idx = np.array(all_idx)[np.argwhere(ano_label == 0).squeeze()].tolist()
        #     overlap = list(set(real_abnormal_label_idx) & set(real_normal_label_idx))
        #
        #     real_affinity, index = torch.sort(affinity[real_abnormal_label_idx])
        #     real_affinity = real_affinity[:300]
        #     draw_pdf(np.array(affinity[real_normal_label_idx].detach().cpu()),
        #              np.array(affinity[abnormal_label_idx].detach().cpu()),
        #              np.array(real_affinity.detach().cpu()), args.dataset, epoch)

        confidence_margin = 0.7
        loss_margin = (confidence_margin - (affinity_normal_mean - affinity_abnormal_mean)).clamp_min(min=0)

        diff_attribute = torch.pow(emb_con - emb_abnormal, 2)
        loss_rec = torch.mean(torch.sqrt(torch.sum(diff_attribute, 1)))

        loss = 1 * loss_margin + 1 * loss_bce + 1 * loss_rec

        loss.backward()
        optimiser.step()
        end_time = time.time()
        total_time += end_time - start_time
        print('Total time is', total_time)
        if epoch % 2 == 0:
            logits = np.squeeze(logits.cpu().detach().numpy())
            lbl = np.squeeze(lbl.cpu().detach().numpy())
            auc = roc_auc_score(lbl, logits)
            # print('Traininig {} AUC:{:.4f}'.format(args.dataset, auc))
            # AP = average_precision_score(lbl, logits, average='macro', pos_label=1, sample_weight=None)
            # print('Traininig AP:', AP)

            print("Epoch:", '%04d' % (epoch), "train_loss_margin=", "{:.5f}".format(loss_margin.item()))
            print("Epoch:", '%04d' % (epoch), "train_loss_bce=", "{:.5f}".format(loss_bce.item()))
            print("Epoch:", '%04d' % (epoch), "rec_loss=", "{:.5f}".format(loss_rec.item()))
            print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(loss.item()))
            print("=====================================================================")
        if epoch % 10 == 0:
            model.eval()
            train_flag = False
            # 在模型调用处，传递 sparse_flag 和 adj_torch/raw_adj_torch
            emb, emb_combine, logits, emb_con, emb_abnormal = model(features, adj_torch, abnormal_label_idx, normal_label_idx,
                                                                    train_flag, args, sparse=sparse_flag)
            # evaluation on the valid and test node
            logits = np.squeeze(logits[:, idx_test, :].cpu().detach().numpy())
            auc = roc_auc_score(ano_label[idx_test], logits)
            print('Testing {} AUC:{:.4f}'.format(args.dataset, auc))
            AP = average_precision_score(ano_label[idx_test], logits, average='macro', pos_label=1, sample_weight=None)
            print('Testing AP:', AP)
            # === Rec@K (K=前10%) ===
            def recall_at_k(y_true, y_score, k):
                idx = np.argsort(y_score)[::-1][:k]
                return np.sum(y_true[idx]) / np.sum(y_true)
            n_test = len(ano_label[idx_test])
            k = max(1, int(n_test * 0.1))
            rec = recall_at_k(ano_label[idx_test], logits, k)
            print(f"Testing Rec@{k} (前10%): {rec:.4f}")
            # === 保存到CSV ===
            import os
            csv_path = 'result_log.csv'
            write_header = not os.path.exists(csv_path)
            with open(csv_path, 'a', encoding='utf-8') as f:
                if write_header:
                    f.write('dataset,epoch,AUC,AP,Rec@K\n')
                f.write(f'{args.dataset},{epoch},{auc:.4f},{AP:.4f},{rec:.4f}\n')

# abnormal_label_idx 采样限制，极限规避OOM
if len(abnormal_label_idx) > 10:
    print(f'采样异常节点数量过大，仅取前10个用于极限实验')
    abnormal_label_idx = abnormal_label_idx[:10]
print('abnormal_label_idx采样后数量:', len(abnormal_label_idx))

# 极限规避OOM：只取前1000个节点
max_nodes = 1000
features = features[:max_nodes]
labels = labels[:max_nodes]
adj = adj[:max_nodes, :max_nodes]
print('features.shape (after node limit):', features.shape)
print('labels.shape (after node limit):', labels.shape)
print('adj.shape (after node limit):', adj.shape)
