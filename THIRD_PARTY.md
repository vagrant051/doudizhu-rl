# 第三方代码说明

本仓库的 `doudizhu/env/` 来自 [kwai/DouZero](https://github.com/kwai/DouZero)（ICML 2021，Apache-2.0）。
规则引擎、合法出牌生成、4×15 牌矩阵编码均保持原实现，便于对照论文。

本仓库自己写的部分：

- `doudizhu/dmc/models.py`：用 1D ResNet 替代 LSTM
- `doudizhu/dmc/dmc.py` / `utils.py` / `arguments.py`：DMC 训练与设备分配
- `doudizhu/evaluation/`：WP / ADP 评测
