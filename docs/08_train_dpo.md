## MiniMind DPO 训练脚本逐行解析（trainer/train_dpo.py）

本文件详细、逐行解释 `trainer/train_dpo.py` 的实现，覆盖导入、函数、训练循环、分布式与日志、模型与数据初始化、命令行参数等。适合想快速理解和二次开发该脚本的同学。

- **脚本目标**: 进行 DPO（Direct Preference Optimization）阶段训练，通过参考模型和当前模型的响应对（chosen/rejected），最小化 DPO 损失，实现偏好对齐。
- **依赖模型**: 使用 `MiniMindForCausalLM` 作为被训练模型与参考模型（ref model）。参考模型权重与被训练模型初始相同，但在训练中冻结不更新。
- **数据需求**: 需要一个 DPO 格式的数据集（默认 `../dataset/dpo.jsonl`），`DPODataset` 会产出 chosen 与 rejected 样本对及其 mask。

---

## 快速使用

```bash
# 进入项目根目录
cd minimind

# 训练（单卡示例）
python trainer/train_dpo.py \
  --out_dir ../out \
  --epochs 2 \
  --batch_size 4 \
  --learning_rate 1e-8 \
  --hidden_size 512 \
  --num_hidden_layers 8 \
  --max_seq_len 1024 \
  --data_path ../dataset/dpo.jsonl

# 分布式（示例，需按环境配置 RANK/LOCAL_RANK/WORLD_SIZE）
RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python -m torch.distributed.run --nproc_per_node=1 trainer/train_dpo.py --ddp
```

- **重要**: 在此 DPO 阶段，建议学习率非常小（默认 `1e-8`），否则容易遗忘已学到的 SFT 能力。
- **前置权重**: 脚本会尝试从 `--out_dir` 下加载名为 `full_sft_<hidden_size>[_moe].pth` 的 SFT 检查点作为初始化。

---

## 逐行解析

### 顶部导入与初始化

- L1-L4: 基本路径设置，确保可从上级目录导入包
  - L1-L2: 导入 `os`, `sys`
  - L3: 设置包名 `__package__ = "trainer"`
  - L4: 将项目根路径加入 `sys.path`
- L6-L19: 常用依赖导入
  - L6: `argparse` 用于命令行参数
  - L7-L9: `time`, `math`, `warnings`
  - L10-L12: `torch` 与功能子模块
  - L13: `nullcontext`，CPU 情况下不做混合精度
  - L14-L16: 优化器、分布式并行、数据加载器/分布式采样器
  - L17: `transformers` 的 `AutoTokenizer`, `AutoModelForCausalLM`（此处未直接用到 `AutoModelForCausalLM`）
  - L18: 项目模型 `MiniMindConfig`, `MiniMindForCausalLM`
  - L19: 数据集 `DPODataset`
- L21: 关闭警告显示

### 日志与学习率

- L24-L27 `Logger`:
  - 仅在非 DDP 或者 rank 0 上打印日志
- L29-L31 `get_lr`:
  - 余弦退火学习率调度：`lr/10 + 0.5*lr*(1+cos(pi*current/total))`

### 概率函数与 DPO 损失

- L33-L39 `logits_to_probs`:
  - 输入: `logits` (B, T, V), `labels` (B, T)
  - 先对词表维度做 `log_softmax` 得到对数概率
  - 用 `gather` 取每个位置标签的对数概率，得到 (B, T)
- L42-L61 `dpo_loss`:
  - 输入: `ref_probs`, `probs`, `mask`, `beta`
  - L45-L47: 按 mask 平均序列上的 log-prob（对 ref 和 pi 都做）
  - L49-L55: 将 batch 前半当作 `chosen`，后半当作 `rejected`
  - L56-L60: DPO 的核心：计算 `pi_logratios - ref_logratios`，再做 `-logsigmoid(beta * logits)` 并取均值

### 训练轮函数

- L63-L131 `train_epoch(epoch, wandb)`：单轮训练
  - L65-L74: 从 `train_loader` 取 batch，并将 chosen/rejected 的输入、标签、mask 拼接（batch 维度）
  - L76-L79: 计算并设置当前 step 的学习率（余弦调度）
  - L80: 使用混合精度上下文（GPU）或空上下文（CPU）
  - L81-L89: 前向
    - L81-L85: 参考模型 `ref_model` 前向，禁止梯度；取标签位置的对数概率，并乘 mask
    - L86-L89: 当前模型前向，取标签位置的对数概率，并乘 mask
  - L90-L92: 计算 DPO 损失，并按梯度累积步数做缩放
  - L93: `scaler.scale(loss).backward()` 进行混合精度反传
  - L95-L101: 梯度累积到步数后：取消缩放、梯度裁剪、`optimizer.step()`、更新 scaler、清零梯度
  - L102-L118: 日志与可选 wandb 记录
  - L119-L130: 定期保存检查点
    - L121-L123: 路径名中包含是否使用 MoE
    - L124-L129: 处理 DDP 包裹、转半精度存储、`torch.save`

### 模型初始化

- L133-L151 `init_model(lm_config)`
  - L134: 从 `../model/` 加载分词器
  - L135: 构建待训练模型 `MiniMindForCausalLM`
  - L136-L139: 从 `--out_dir` 下加载 `full_sft_<hidden_size>[_moe].pth` 权重为初始权重（若存在）
  - L141-L144: 构建参考模型、加载相同权重、`eval()`、`requires_grad_(False)`（冻结）
  - L146-L149: 打印可训练参数量，将模型与参考模型迁移到设备
  - L150: 返回 `(model, ref_model, tokenizer)`

### 分布式初始化

- L153-L163 `init_distributed_mode()`
  - L154: 非 DDP 直接返回
  - L157: 初始化进程组（NCCL）
  - L158-L161: 读取 `RANK/LOCAL_RANK/WORLD_SIZE` 环境变量
  - L161-L163: 设置 CUDA 设备

### 主入口

- L165: `if __name__ == "__main__":`
- L166-L189: 命令行参数
  - `--out_dir`: 输出目录（默认 `../out`）
  - `--epochs`: 轮数（默认 2）
  - `--batch_size`: 批大小（默认 4）
  - `--learning_rate`: 学习率（默认 `1e-8`，强烈建议极小）
  - `--device`: 设备（默认自动）
  - `--dtype`: 精度（默认 `bfloat16`）
  - `--use_wandb`, `--wandb_project`
  - `--num_workers`: DataLoader 线程数
  - `--ddp`: 是否分布式
  - `--accumulation_steps`: 梯度累积步数
  - `--grad_clip`: 梯度裁剪阈值
  - `--warmup_iters`: 预热步数（未显式使用到）
  - `--log_interval`, `--save_interval`
  - `--hidden_size`, `--num_hidden_layers`, `--max_seq_len`, `--use_moe`
  - `--data_path`: DPO 数据路径（默认 `../dataset/dpo.jsonl`）
- L192: 构建 `MiniMindConfig`
- L193-L195: 准备保存目录
- L196: 计算每 iter token 数（未后续使用）
- L197: 标记设备类型
- L199: 组装 wandb run name
- L201: 设置混合精度上下文（GPU: `torch.cuda.amp.autocast()`；CPU: `nullcontext()`）
- L202: 检测是否 DDP（通过 `RANK` 环境变量）
- L203-L207: 固定随机种子（`base_seed=1337`）
- L208-L215: 若 DDP，初始化分布式，设置设备与 per-rank 种子
- L216-L221: 初始化 wandb（仅 rank 0）或禁用
- L223: 调用 `init_model`，获取 `model, ref_model, tokenizer`
- L225: 构建 `DPODataset`（会读取数据并编码为模型输入）
- L226: 若 DDP，构造 `DistributedSampler`
- L227-L235: 构建 `DataLoader`
- L237: 混合精度 `GradScaler`（在 `float16/bfloat16` 精度时启用）
- L238: 优化器 `AdamW`
- L240-L243: 若 DDP，设置忽略参数并用 `DistributedDataParallel` 包裹模型
- L244-L247: 计算每轮 step 数，并按轮训练（调用 `train_epoch`）

---

## DPO 核心公式回顾

- 设当前模型 π 与参考模型 π_ref 对 chosen 与 rejected 的序列平均 log-prob 分别为：
  - `log π(y_c | x)`, `log π(y_r | x)`，`log π_ref(y_c | x)`, `log π_ref(y_r | x)`
- 定义：
  - `pi_logratios = log π(y_c|x) - log π(y_r|x)`
  - `ref_logratios = log π_ref(y_c|x) - log π_ref(y_r|x)`
- 损失：
  - `L = - E[ log σ( β * (pi_logratios - ref_logratios) ) ]`
- 本实现细节：
  - 对序列维度按 `mask` 求平均，求得每个样本的 log-prob
  - batch 前半为 chosen，后半为 rejected
  - `beta` 在脚本中固定为 `0.1`

---

## 训练与保存要点

- **学习率**: DPO 阶段非常小（默认 1e-8），并用余弦退火调度
- **参考模型**: 冻结权重，仅用于提供 log-prob 参照
- **混合精度**: 默认启用（`bfloat16`），配合 `GradScaler`
- **梯度累积**: 通过 `--accumulation_steps` 控制
- **梯度裁剪**: `torch.nn.utils.clip_grad_norm_` 防止梯度爆炸
- **断点保存**: 定期保存半精度权重到 `--out_dir/rlhf_<hidden_size>[_moe].pth`

---

## 常见问题（FAQ）

- **加载 SFT 检查点失败**
  - 确保 `--out_dir` 下存在 `full_sft_<hidden_size>[_moe].pth`；或调整脚本中加载路径。
- **DDP 无法启动**
  - 检查环境变量 `RANK/LOCAL_RANK/WORLD_SIZE` 与 NCCL 环境设置；确保 GPU 与驱动正常。
- **显存不足**
  - 调小 `--batch_size`、`--max_seq_len`，或提高梯度累积步数，或使用更小 `hidden_size/num_hidden_layers`。
- **loss 不收敛/发散**
  - 进一步减小学习率；检查数据质量和 `mask`；确保参考模型冻结、未被错误更新。

---

## 版本与联系

- 适配脚本版本: 与仓库同版 `trainer/train_dpo.py`
- 若你在使用中遇到问题或发现文档错误，欢迎在仓库 issue 中反馈并附带复现实例。

---

## 更深入的逐行解析：作用、向量/张量形状、为什么这么做

以下在原有逐行解析的基础上，明确每一步的张量形状与动机。记号约定：
- B: 单个分支（chosen 或 rejected）的 batch 大小；拼接后为 2B
- T: 序列长度 `max_seq_len`
- V: 词表大小

### 导入与全局设置（L1-L21）
- L1-L4: 路径设置
  - 作用: 确保 `trainer` 能从项目根导入模块
  - 形状: 无
  - 原因: 便于从 `model/`、`dataset/` 等目录导入自定义代码
- L6-L19: 导入依赖
  - 作用: 训练、分布式、数据与模型等所需库
  - 形状: 无
  - 原因: 标准 PyTorch/NLP 训练脚本依赖
- L21: `warnings.filterwarnings('ignore')`
  - 作用: 忽略冗余警告
  - 原因: 清爽日志输出

### 日志与学习率（L24-L31）
- L24-L27 `Logger`
  - 作用: 仅主进程打印（非 DDP 或 rank 0）
  - 原因: 分布式下避免重复日志
- L29-L31 `get_lr(current_step, total_steps, lr)`
  - 作用: 余弦退火 + 下限项
  - 公式: `lr_eff = lr/10 + 0.5*lr*(1 + cos(pi*current/total))`
  - 性质:
    - step=0 时: `1.1*lr`（稍高于初始值，加快早期收敛）
    - step=total 时: `0.1*lr`（不降到 0，保持微小学习率以稳定收尾）
  - 原因: 让学习率平滑衰减且保留最小学习率，避免完全停更

### 概率函数与 DPO 损失（L33-L61）
- L33-L39 `logits_to_probs(logits, labels)`
  - 输入: `logits` 形状 (2B, T, V), `labels` 形状 (2B, T)
  - 过程:
    - `F.log_softmax(logits, dim=2)` → `log_probs` (2B, T, V)
    - `gather(..., index=labels.unsqueeze(2))` → (2B, T, 1) → `squeeze` → (2B, T)
  - 输出: 每个 token 的对数概率 `log p(y_t | x, θ)`，形状 (2B, T)
  - 原因: DPO 目标基于对数概率的差，直接在对数域计算更稳定；函数名中的 `probs` 实际是 log-probs
- L42-L61 `dpo_loss(ref_probs, probs, mask, beta)`
  - 输入: `ref_probs`, `probs` 均为 (2B, T) 的对数概率；`mask` 为 (2B, T)，0/1 标记有效 token
  - L45: `seq_lengths = mask.sum(dim=1, keepdim=True)` → (2B, 1)
  - L46-L47: 对序列维度按 mask 求均值，对每个样本得到标量 log-prob：`(2B,)`
  - L49-L55: 按 batch 一分为二：前 B 为 chosen，后 B 为 rejected
    - `chosen_ref_probs` (B,), `reject_ref_probs` (B,)
    - `chosen_probs` (B,), `reject_probs` (B,)
  - L56-L60: DPO 公式
    - `pi_logratios = chosen_probs - reject_probs` → (B,)
    - `ref_logratios = chosen_ref_probs - reject_ref_probs` → (B,)
    - `logits = pi_logratios - ref_logratios` → (B,)
    - `loss = -logsigmoid(beta * logits)` → (B,) → `.mean()` → 标量
  - 原因: 目标让当前策略对 chosen 的偏好超过 rejected，且以参考策略的偏好差作基准；`beta` 控制温度/力度

注意: 在 `train_epoch` 中，传入前已做过一次 `* mask`，此处再次 `* mask` 等价但稍显冗余；不影响正确性。

### 单轮训练（L63-L131）
- L65-L74: 取 batch 并拼接
  - `x_chosen, x_rejected`: (B, T)
  - `y_chosen, y_rejected`: (B, T)
  - `mask_chosen, mask_rejected`: (B, T)
  - 拼接后：
    - `x = cat([x_chosen, x_rejected], dim=0)` → (2B, T)
    - `y = cat([y_chosen, y_rejected], dim=0)` → (2B, T)
    - `mask = cat([mask_chosen, mask_rejected], dim=0)` → (2B, T)
  - 原因: 组合成一个 batch 以便一次前向，节约算力；后续按前/后半还原 chosen/rejected
- L76-L79: 计算并设置学习率
  - 形状: 标量
  - 原因: 逐 step 调度，适配训练进度
- L80: dtype 上下文
  - CPU: `nullcontext()`；CUDA: `torch.cuda.amp.autocast()`
  - 原因: 自动混合精度，减少显存、提速
- L81-L85: 参考模型前向（无梯度）
  - `ref_outputs = ref_model(x)` → `ref_logits` (2B, T, V)
  - `ref_probs = logits_to_probs(ref_logits, y)` → (2B, T) 对数概率
  - `ref_probs = ref_probs * mask` → (2B, T)，padding 位置置零
  - 原因: 作为 DPO 目标中的 π_ref，不参与更新
- L86-L89: 当前模型前向
  - `logits = model(x).logits` → (2B, T, V)
  - `probs = logits_to_probs(logits, y)` → (2B, T) 对数概率
  - `probs = probs * mask` → (2B, T)
  - 原因: 产生 π 的对数概率用于 DPO 目标
- L90-L92: 计算损失并按累积步数缩放
  - `loss = dpo_loss(ref_probs, probs, mask, beta=0.1)` → 标量
  - `loss = loss / accumulation_steps`
  - 原因: 梯度累积时按步数缩小单步 loss，确保等效总梯度
- L93: `scaler.scale(loss).backward()`
  - 作用: 在 AMP 下安全反传
  - 原因: 避免溢出，提高稳定性
- L95-L101: 到达累积步执行优化
  - `scaler.unscale_(optimizer)`：将梯度还原到真实量级
  - `clip_grad_norm_`：梯度裁剪防止爆炸
  - `scaler.step(optimizer)`：执行一步优化
  - `scaler.update()`：动态缩放更新
  - `optimizer.zero_grad(set_to_none=True)`：清梯度（`set_to_none` 更省内存）
  - 原因: 标准 AMP + 梯度累积最佳实践
- L102-L118: 日志与可选 wandb
  - 形状: 标量记录
  - 原因: 便于监控训练过程
- L119-L130: 定期保存
  - 取出 `state_dict`（DDP 需 `model.module`）并转换为半精度后保存
  - 原因: 节省磁盘与加载显存；DPO 阶段推理多用半精度/混合精度

### 模型初始化（L133-L151）
- L134: `AutoTokenizer.from_pretrained('../model/')`
  - 作用: 载入项目内已有分词器
- L135: `MiniMindForCausalLM(lm_config)`
  - 作用: 构建待训练模型
- L136-L139: 加载 SFT 权重
  - 路径: `full_sft_<hidden_size>[_moe].pth`
  - `strict=False` 允许部分键不匹配（例如新增头部）
  - 原因: 从 SFT 阶段起步，DPO 是微调偏好
- L141-L144: 构建参考模型并冻结
  - `eval()`、`requires_grad_(False)`
  - 原因: DPO 中 π_ref 固定不更新
- L146-L149: 打印可训练参数量并迁移设备

### 分布式初始化（L153-L163）
- L157: `dist.init_process_group(backend='nccl')`
  - 原因: 多 GPU 通信
- L158-L163: 从环境变量获取 `RANK/LOCAL_RANK/WORLD_SIZE` 并设置设备
  - 原因: 正确映射本地进程到对应 GPU

### 主入口（L165-L247）
- L166-L189: 参数
  - 特别提示:
    - `--learning_rate` 默认 1e-8，DPO 阶段极小 LR 避免灾难性遗忘
    - `--dtype` 默认 bfloat16；AMP + GradScaler 在 bf16 场景下收益有限，保留兼容
- L192: `MiniMindConfig`
- L193-L195: 创建保存目录
- L196: `tokens_per_iter = batch_size * max_seq_len`（未使用，信息统计）
- L197: `device_type` 用于 AMP 上下文
- L199: 组装 wandb 运行名
- L201: 上下文 `ctx`：CUDA 用 `autocast`，CPU 用 `nullcontext`
- L202-L215: DDP 与随机种子
  - 若 DDP：初始化分布式，且每个 rank 拥有不同种子（`base_seed + rank`）
- L216-L221: wandb init（仅主进程）
- L223: `init_model` 返回 `model, ref_model, tokenizer`
- L225: `DPODataset(data_path, tokenizer, max_length)`
  - 期望输出键: `x_chosen, y_chosen, mask_chosen, x_rejected, y_rejected, mask_rejected`
  - 形状: 均为 (B, T)
- L226-L235: `DistributedSampler`（可选）与 `DataLoader`
- L237: `GradScaler(enabled=(dtype in ['float16','bfloat16']))`
  - 说明: 对 bf16 的缩放在 PyTorch 中通常是无效/不必要的，保留不影响正确性
- L238: `optim.AdamW`
- L240-L243: DDP 包裹并忽略 `pos_cis`（位置编码缓存）
- L244-L247: 训练若干 epoch

---

## 批次形状追踪（示例）

假设 `B=4, T=1024, V=50000`：

- 数据集输出（每次迭代）
  - `x_chosen, y_chosen, mask_chosen`: (4, 1024)
  - `x_rejected, y_rejected, mask_rejected`: (4, 1024)
- 拼接后
  - `x, y, mask`: (8, 1024)
- 模型前向
  - `ref_logits, logits`: (8, 1024, 50000)
- 取标签项对数概率
  - `ref_log_probs, log_probs`: (8, 1024)
  - 乘 mask → (8, 1024)
- DPO 损失内部
  - `seq_lengths`: (8, 1)
  - 序列均值（按 mask）→ `ref_seq_logp, seq_logp`: (8,)
  - 切分 chosen/rejected → (4,) / (4,)
  - `pi_logratios, ref_logratios, logits`: (4,)
  - `loss`: 标量

---

## 设计动机与取舍

- 使用对数概率而非概率
  - 稳定性高，DPO 直接使用对数差；避免 underflow
- 对序列做 mask 平均
  - 避免不同长度样本在损失上不公平（长样本累积 log-prob 更小）
- 拼接 chosen/rejected 再分割
  - 充分利用批处理能力，减少两次前向开销
- 参考模型冻结
  - DPO 理论设定：π_ref 提供偏好基准，不参与更新
- 余弦退火 + 底座学习率
  - 平滑、稳定；DPO 阶段 LR 极小，防遗忘
- 混合精度 + 梯度缩放
  - 节省显存，提升吞吐；对 bf16 的缩放收益有限但不妨碍
- 半精度保存
  - 节省磁盘与加载显存，推理/继续训练友好

---

## 常见陷阱与校验

- 数据顺序假设
  - 代码假设 `cat([chosen, rejected])` 后，前半是 chosen、后半是 rejected；请确保 `DPODataset` 的 `__getitem__` 与 `collate_fn` 保持这一顺序一致
- mask 应为 0/1 且与 `y` 对齐
  - 否则序列均值与最终损失会偏移
- 学习率过大
  - DPO 阶段应极小（默认 1e-8），过大会导致灾难性遗忘
- GradScaler 与 bf16
  - 在 bf16 下缩放通常不是必须的；若出现不稳定，可切换到 fp16 或关闭缩放
- 断点与权重严格匹配
  - `strict=False` 容忍键不匹配；若结构差异大，请检查 config 与权重是否对应