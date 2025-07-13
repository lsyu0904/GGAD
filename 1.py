import dgl

graph_path = r'D:\PythonProject\0710-gad\GGAD\dataset\amazon'
try:
    graphs, _ = dgl.load_graphs(graph_path)
    graph = graphs[0]
    print("读取成功，graph 类型：", type(graph))
    print("节点数：", graph.number_of_nodes())
    print("边数：", graph.number_of_edges())
except Exception as e:
    print("dgl.load_graphs 读取失败：", e)