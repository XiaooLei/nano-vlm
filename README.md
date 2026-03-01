# Qwen VLM 模型

## 模型简介

Qwen VLM 是一个基于 Qwen2.5 语言模型和 CLIP 视觉模型的多模态视觉-语言模型，能够处理图像和文本输入，实现图像描述、视觉问答等任务。

## 模型架构

### 核心组件
- **语言模型**：Qwen2.5-0.5B-Instruct
- **视觉编码器**：openai/clip-vit-base-patch16
- **视觉-语言投影器**：线性投影网络，将视觉特征映射到语言模型的嵌入空间

### 技术特点
- 支持多种设备（CUDA、MPS、CPU）
- 实现了 LoRA 微调技术
- 动态处理输入序列长度
- 严格对齐 Qwen 的 ChatML 格式
- 支持图像-文本多模态输入

## 安装依赖

```bash
pip install transformers torch pillow peft
```

## 模型使用

### 初始化模型

```python
from lab6-vlm.model import VLMModel

# 初始化模型
model = VLMModel(
    llm_name="Qwen/Qwen2.5-0.5B-Instruct",
    vision_name="openai/clip-vit-base-patch16",
    train_mode="both"  # 可选: "both", "lora", "projector"
)
```

### 加载 LoRA 权重

```python
# 加载预训练的 LoRA 权重
lora_params = torch.load("lora_weights.pt")
model.load_lora(lora_params)
```

### 推理示例

```python
from PIL import Image

# 加载图像
image = Image.open("example.jpg")

# 构建提示
prompt = """
<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
<image>
请描述这张图片。
<|im_end|>
<|im_start|>assistant
"""

# 生成回答
response = model.answer(image, prompt, max_new_tokens=128)
print(response)
```

## 模型训练

### 训练配置

模型支持三种训练模式：
- `both`：同时训练投影器和语言模型（使用 LoRA）
- `lora`：仅训练语言模型（使用 LoRA）
- `projector`：仅训练投影器

### 训练示例

```bash
# 训练命令示例
python3 train.py --sample_size 10000 --batch_size 2 --llm_name Qwen/Qwen2.5-0.5B-Instruct --num_epochs 3 --projector_init_file projector_init.pt --chat_round 2 --max_seq_len 768 --train_mode lora
```

## 技术细节

### 视觉特征处理

1. 使用 CLIP 视觉模型提取图像特征
2. 通过投影器将视觉特征映射到语言模型的嵌入空间
3. 在输入序列中找到 `<image>` 标记的位置
4. 将视觉特征插入到对应位置，替换 `<image>` 标记

### 输入输出示例

#### 输入
```python
image = Image.open("cat.jpg")
prompt = """
<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
<image>
这是什么动物？
<|im_end|>
<|im_start|>assistant
"""
```

#### 输出
```
<|im_start|>assistant
这是一个猫。
<|im_end|>
```



## 注意事项

1. **设备选择**：模型会自动选择可用的设备（CUDA > MPS > CPU）
2. **数据类型**：建议使用 `bf16` 或 `fp32`，Qwen2.5 在 `fp16` 下有时不稳定
3. **输入格式**：必须使用 ChatML 格式，并包含 `<image>` 标记
4. **批量处理**：模型支持批量处理，但需要确保输入序列长度一致

## 性能优化

- 使用 `sdpa` 注意力实现，提高计算效率
- 采用动态 Padding，减少计算开销
- 支持 LoRA 微调，减少可训练参数数量

## 许可证

本项目基于 MIT 许可证开源。