## 概述
LLaMA 模型是一个基于 **自注意力机制** 和 **旋转位置编码** 的语言模型。它具有高度的可配置性，适用于自然语言处理任务，如语言建模和文本生成。

## 功能
- **旋转位置编码**：高效捕捉长序列的位置信息。
- **RMSNorm**：增强训练稳定性。
- **自注意力机制**：多头自注意力用于建模词之间的关系。
- **灵活配置**：可以调整层数、注意力头数、词汇表大小等。

## 安装
安装依赖：

```bash
pip install torch
```

## 使用方法

### 模型初始化
```python
from llama_model import llamaModel, ModelArgs

args = ModelArgs(dim=4096, n_layers=32, num_heads=32, vocab_size=50257, device='cuda')
model = llamaModel(args)
```

### 前向传播
```python
tokens = torch.tensor([[1]], device=args.device)
start_pos = 0
output = model(tokens, start_pos)
```

## 配置项
- `dim`：隐藏状态维度（默认：`4096`）
- `n_layers`：编码器层数（默认：`32`）
- `num_heads`：注意力头数（默认：`32`）
- `vocab_size`：词汇表大小
- `max_seq_length`：最大序列长度（默认：`2048`）

## 示例
初始化并执行前向传播：
```python
tokens = torch.tensor([[1]], device=args.device)
start_pos = 0
output = model(tokens, start_pos)
```
