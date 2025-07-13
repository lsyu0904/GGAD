import os
import dgl
import numpy as np
import scipy.sparse as sp
import scipy.io as sio

# 数据集目录
DATASET_DIR = os.path.join(os.path.dirname(__file__), 'dataset')

# 遍历所有无扩展名文件
def is_graph_file(filename):
    return os.path.isfile(os.path.join(DATASET_DIR, filename)) and '.' not in filename

files = [f for f in os.listdir(DATASET_DIR) if is_graph_file(f)]

for fname in files:
    graph_path = os.path.join(DATASET_DIR, fname)
    save_path = graph_path + '.mat'
    print(f'处理: {fname}')
    try:
        graphs, _ = dgl.load_graphs(graph_path)
        graph = graphs[0]
        # 邻接矩阵
        adj = graph.adjacency_matrix(scipy_fmt='csr')
        # 特征
        if 'feature' in graph.ndata:
            features = graph.ndata['feature'].numpy()
        else:
            raise ValueError(f"{fname} 未找到节点特征 'feature'")
        # 标签
        if 'label' in graph.ndata:
            labels = graph.ndata['label'].numpy()
        else:
            raise ValueError(f"{fname} 未找到节点标签 'label'")
        # 保存为 .mat
        data = {
            'Network': adj,
            'Attributes': features,
            'Label': labels
        }
        sio.savemat(save_path, data)
        print(f"已保存为 {save_path}")
    except Exception as e:
        print(f"处理 {fname} 时出错: {e}") 