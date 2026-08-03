# MiniMax-H3-FineTuning

[English](README.md) | 中文

[MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) 的一套最小可用微调方案——
H3 是开放权重的全模态生成模型,能联合生成视频与同步立体声音频。

MiniMax 放出权重时表示"支持包括微调在内的进一步开发",但**没有发布任何训练器**,Hugging Face
Diffusers 集成也只覆盖推理。本仓库补上这个缺口:基于官方 Diffusers 集成实现了一个监督式
rectified-flow 训练器(约 150 行)+ latent 离线缓存预处理器,并完整记录了 **4 个 H3 特有的数值
约定**——搞错任何一个,训练都会悄悄把权重往错误方向推(见 [FIXES.md](FIXES.md),我们踩过)。

## 实现原理

**两阶段设计**(训练时 GPU 显存里只有 33B transformer 一个大件):

1. **`prepare_cache.py`** —— 离线编码每条训练样本:目标视频 → H3-VisualVAE latent(patch 化行)、
   目标音频 → H3-AudioVAE latent、caption → Qwen3-VL 第 50 层 hidden states。每样本一个 `.pt`。
2. **`train.py`** —— 只加载 transformer(`transformer` = FL2VA 变体 / `transformer_ref` = Ref2VA
   变体),用官方 `build_packed_sequence` / `build_row_timesteps` 重建视频/音频/文本打包序列布局,
   优化 rectified-flow MSE loss。

**训练目标。** H3 发布的是 guidance 蒸馏后的 rectified-flow 权重,相对常见的 SD3/Wan 约定有两个
特殊点:

- transformer 的时间输入是 `t = 1 − σ`(t=1 干净、t=0 纯噪);
- transformer 预测的是**指向数据的速度** `v = x₀ − ε`(scheduler 用 `x₀ = x_t + (1−t)·v` 还原,
  注意是**加号**)。

因此 loss 按 `x_t = (1−σ)·x₀ + σ·ε` 加噪,回归目标为 `pred → (x₀ − ε)`;且每步用**两个不同的 σ**:
视频和音频各走一条 shift 曲线(`σ = shift·u / (1+(shift−1)·u)`,视频 shift=12.0、音频 shift=3.0),
共用同一个 `u` 采样——与推理时两个 scheduler 同步推进的 (σ_v, σ_a) 配对分布一致。

**可训练参数模式:**

| `--trainable` | 训练什么 | 适用场景 |
|---|---|---|
| `heads`(默认) | `proj_out` + `audio_proj_out`(约百万参数) | 单卡冒烟,验证整条链路 |
| `lora` | PEFT LoRA(`to_qkv`、`to_out.0`、`linear_1`、`linear_2`) | 单机实用微调 |
| `all` | 全部 33B | 需 `--strategy deepspeed`(附 ZeRO-3 配置) |

`heads`/`lora` 模式的 checkpoint 只保存可训练张量(几 MB,而不是 ~66 GB——全量序列化每 100 步会让
rank 0 卡住数分钟,其余 rank 在集合通信处等待直至 NCCL watchdog 杀掉进程,见 FIXES.md 第 5 条)。

## 快速开始

```bash
# 1. 环境:Python 3.11、torch >= 2.8。H3 的类还没进 diffusers 正式发行版——
#    install_env.sh 会安装固定 revision 的集成版本。
bash install_env.sh

# 2. 按 examples/ 里的 schema 写一份 manifest
python validate_manifest.py --manifest path/to/train.jsonl --check-files

# 3. 缓存 latent(debug 规格:256x256、22 帧;正式训练自行调大)
python prepare_cache.py \
  --metadata path/to/train.jsonl \
  --output cache/train \
  --model /path/to/MiniMax-H3 \
  --height 256 --width 256 --frames 22 --encode-text

# 4. 训练(单卡冒烟)
CUDA_VISIBLE_DEVICES=0 python train.py \
  --model /path/to/MiniMax-H3 --variant ref2va \
  --cache cache/train --output runs/smoke \
  --max-steps 10 --trainable heads

# 4'. 多卡(8 卡 DDP)
python -m torch.distributed.run --nproc_per_node 8 train.py \
  --model /path/to/MiniMax-H3 --variant ref2va \
  --cache cache/train --output runs/heads_1000 \
  --max-steps 1000 --trainable heads
```

来自模型本身的约束:`num_frames % 17 == 5`、宽高均为 32 的倍数、视频 24 fps、音频 32 kHz 立体声、
`audio_latent_frames = round(num_frames / 24 * 40)`。

## 效果验证

数值约定搞错时(timestep 方向 + 速度符号同时反),heads 模式的 loss **越训越高**(10 步内
7.2 → 9.5);用本仓库的修正版,同样配置 loss 稳定在 ≈ 0.3–1.0,8×A800 上 1000 步全程平稳,训出的
输出头 patch 回官方推理管线后生成正常。详见 [FIXES.md](FIXES.md)。

## 当前局限

- `prepare_cache.py` 是最小缓存路径:只编码**目标视频**——音频 latent 是全零占位(训练器会检测并
  将音频 loss 权重置零),参考素材也尚未编码进条件序列。把缓存扩展到真实音频 + 参考编码,是实现
  完整 Ref2VA 微调的主要 TODO。
- epoch 之间无样本 shuffle;batch 恒为每步 1 条序列。
- H3 最终训练阶段使用的稀疏注意力实现未开源,训练只能走全注意力——请控制片长和分辨率。

## 许可

模型权重受 [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)
约束(有地域与用途限制;微调产物属 Model Derivatives)。本仓库只含代码——不含模型权重与训练媒体数据。
