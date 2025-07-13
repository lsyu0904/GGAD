# GGAD 论文复现与实验对比

## 1. 环境配置
- Python 3.7
- PyTorch 1.11.0
- DGL 0.6.1
- torch-geometric 2.0.4
- 其余依赖见 requirements.txt

## 2. 数据集准备
请将 .mat 数据集放入 `GGAD/dataset/` 目录。数据集获取方式见论文或[官方仓库](https://github.com/mala-lab/GGAD)。

## 3. 运行方法
```bash
python run.py --dataset amazon
python run.py --dataset tfinance
python run.py --dataset reddit
python run.py --dataset elliptic
python run.py --dataset photo
python run.py --dataset dgraphfin
# 依次运行所有基准数据集
```

## 4. 结果对比
| 数据集   | 论文AUROC | 复现AUROC | 论文AUPRC | 复现AUPRC | 备注 |
|----------|----------|----------|----------|----------|------|
| Amazon   | 0.9443   |          | 0.8228   |          |      |
| T-Finance| 0.8228   |          | 0.6466   |          |      |
| Reddit   | 0.6749   |          | 0.5943   |          |      |
| Elliptic | 0.7490   |          | 0.4720   |          |      |
| Photo    | 0.6466   |          | 0.5943   |          |      |
| DGraph   | 0.4792   |          | 0.1885   |          |      |

> 请将你复现的结果填写到表格中，并与论文结果对比。

## 5. 性能差异分析
- 依赖版本差异（如 PyTorch、DGL、PyG 等）可能导致结果不同。
- 硬件差异（CPU/GPU、内存等资源）可能影响训练稳定性。
- 随机种子设置不同会导致结果有波动。
- 参数设置（如 batch size、学习率、训练轮数等）与论文不一致会影响结果。
- 数据预处理方式不同也可能导致差异。
- 平台差异（Windows/Linux）下部分包行为略有不同。

建议在复现实验时，尽量固定随机种子，并严格按照论文参数设置。多次运行取平均，减少偶然性。

## 6. 开源说明
- 整理好你的代码、依赖说明（requirements.txt/environment.yml）、运行脚本、实验结果和对比分析。
- 上传到 GitHub/Gitee 等平台，完善 README，说明如何复现你的实验。
