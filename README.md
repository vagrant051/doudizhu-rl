# doudizhu-rl

斗地主强化学习基线：**DouZero 环境 + ResNet Q 网络 + Deep Monte Carlo (DMC)**。

环境（规则、合法出牌、特征编码）复用 [DouZero](https://github.com/kwai/DouZero)（ICML 2021）。
网络和训练循环是本仓库重写的，方便之后加 Minimum Splits、启发式剪枝、CTDE 等改进。

## 为什么先做这一版

| 模块 | 这一版做什么 | 以后再加 |
|------|----------------|----------|
| 环境 | DouZero 的出牌编码 `(z: 5×162, x: 373/484)` | 叫分、完美信息特征 |
| 算法 | DMC：用整局回报 G 回归 Q(s,a) | PPO、奖励塑形 |
| 网络 | 1D ResNet 编码历史，替代 LSTM | 对手建模、CTDE critic |
| 设备 | Windows 默认 CPU Actor + GPU Learner | Linux 多卡 Actor |

你这台机器是 **RTX 4060 Laptop 8GB + Windows**。CUDA 张量没法在 Windows 进程间 `share_memory`，所以 Actor 走 CPU、Learner 走 GPU。这是 DouZero 官方也承认的限制，不是实现错误。

## 安装

建议在项目目录建虚拟环境：

```powershell
cd C:\Users\31962\Projects\doudizhu-rl
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy
# GPU（4060 推荐，当前默认 pip 会装到 CPU 版）
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

没有 GPU 或只想先跑通：

```powershell
pip install torch numpy
```

## 先跑通

```powershell
python smoke_test.py
```

应打印观察形状，并完成一次 ResNet 前向。

生成 1000 副评测牌，随机 AI 互打（确认评测流水线）：

```powershell
python generate_eval_data.py --num_games 1000
python evaluate.py --landlord random --landlord_up random --landlord_down random --eval_data eval_data.pkl
```

## 训练

笔记本推荐：

```powershell
python train.py --num_actors 4 --batch_size 32 --training_device 0
```

权重写到 `checkpoints/douzero-resnet/`：

- `model.tar`：完整断点（三个位置 + 优化器）
- `landlord_weights_<frames>.ckpt`：评测用单位置权重

继续训练：

```powershell
python train.py --load_model --num_actors 4 --training_device 0
```

纯 CPU：

```powershell
python train.py --actor_device_cpu --training_device cpu --num_actors 2
```

## 评测

把三个位置的 ckpt 填进去（路径按实际 frames 改）：

```powershell
python evaluate.py `
  --landlord checkpoints/douzero-resnet/landlord_weights_0.ckpt `
  --landlord_up random `
  --landlord_down random `
  --eval_data eval_data.pkl
```

指标：

- **WP**：胜率
- **ADP**：带炸弹翻倍的平均分差

## 代码怎么读

1. `doudizhu/env/env.py` 的 `get_obs()`：特征从哪来
2. `doudizhu/dmc/models.py`：ResNet 怎么吃 `z` 和 `x`
3. `doudizhu/dmc/utils.py` 的 `act()`：self-play 如何把终局回报写成 MC target
4. `doudizhu/dmc/dmc.py` 的 `learn()`：MSE 更新 + 把权重同步回 Actor
5. `doudizhu/dmc/arguments.py` 的 `apply_device_defaults()`：Windows 设备策略

## 路线图

当前只做基线。下一阶段按侵入性从小到大：

1. **Minimum Splits**：在 DMC 的 G 上加「最少拆牌数」内在奖励（改 `act()` 里的 target）
2. **启发式剪枝**：合法动作太多时先按牌型/拆牌启发式丢掉明显差的候选
3. **CTDE / 完美信息蒸馏**：训练时 critic 看见三家手牌，执行时仍然只用自己的观察
