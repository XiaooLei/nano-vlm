# Nano VLM Model

## Overview

Nano VLM is a lightweight multimodal vision-language model built upon the Qwen2.5-0.5B-instruct language model and CLIP vision encoder. It processes image and text inputs to enable capabilities such as image captioning and visual question answering.

## Architecture

### Core Components

- **Language Model**: Qwen2.5-0.5B-Instruct
- **Vision Encoder**: openai/clip-vit-base-patch16
- **Vision-Language Projector**: Multi-layer perceptron (MLP) that maps visual features to the language model's embedding space

### Projector Architecture

```
CLIP Visual Features (768-dim) → Linear(768, 2048) → LayerNorm → GELU → Linear(2048, 896) → LayerNorm → LLM Embedding Space (896-dim)
```

### Key Features

- Multi-device support (CUDA > MPS > CPU, auto-selected)
- LoRA fine-tuning (r=64, lora_alpha=128)
- Dynamic sequence length handling
- Strict ChatML format compliance
- Image-text multimodal input support
- SDPA attention implementation

## Installation

```bash
pip install transformers torch pillow peft
```

## Quick Start

### Model Initialization

```python
from lab6_nano_vlm.model import VLMModel
import torch

# Recommended: use bf16 or fp32 (Qwen2.5 can be unstable with fp16)
target_dtype = torch.float32

model = VLMModel(
    llm_name="Qwen/Qwen2.5-0.5B-Instruct",
    vision_name="openai/clip-vit-base-patch16",
    train_mode="both"  # Options: "both", "lora", "projector"
)
```

### Special Token Configuration

The model uses `<image>` as the image placeholder. It must be added to the tokenizer as a special token:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(llm_name)
tokenizer.add_tokens(["<image>"], special_tokens=True)
image_token_id = tokenizer.convert_tokens_to_ids("<image>")
```

### Loading LoRA Weights

```python
# Load pretrained LoRA weights
lora_params = torch.load("lora_weights.pt", weights_only=True)
model.load_lora(lora_params)
```

### Inference Example

```python
from PIL import Image

# Load image
image = Image.open("example.jpg")

# Construct prompt (must use ChatML format with <image> token)
prompt = """
<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
<image>
Please describe this image.
<|im_end|>
<|im_start|>assistant
"""

# Generate response
response = model.answer(image, prompt, max_new_tokens=128)
print(response)
```

## Training

### Training Modes

| Mode | Description |
|------|-------------|
| `both` | Train both projector and language model (with LoRA) |
| `lora` | Train language model only (with LoRA) |
| `projector` | Train projector only |

### Training Command

```bash
python train.py \
    --sample_size 10000 \
    --batch_size 2 \
    --llm_name Qwen/Qwen2.5-0.5B-Instruct \
    --num_epochs 3 \
    --projector_init_file projector_init.pt \
    --chat_round 2 \
    --max_seq_len 768 \
    --train_mode lora
```

### LoRA Configuration

- **Rank (r)**: 64
- **Alpha**: 128
- **Target Modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Dropout**: 0.05

## Technical Details

### Visual Feature Processing Pipeline

1. **Visual Encoding**: Extract image features using CLIP vision encoder, output dimension `[B, 577, 768]`
2. **Feature Projection**: Map visual features to LLM embedding space via projector, discard first CLS token to obtain `[B, 576, 768]`
3. **Token Replacement**: Locate `<image>` token position in input sequence and insert visual features in place of this token
4. **Dynamic Padding**: Pad sequences to the longest sequence in current batch for computational efficiency

### Input/Output Example

#### Input

```python
image = Image.open("cat.jpg")
prompt = """
<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
<image>
What animal is this?
<|im_end|>
<|im_start|>assistant
"""
```

#### Output

```
<|im_start|>assistant
This is a cat.
<|im_end|>
```

## Important Notes

1. **Device Selection**: Model automatically selects available device (CUDA > MPS > CPU)
2. **Data Type**: Use `bf16` or `fp32` (Qwen2.5 can be unstable with `fp16`)
3. **Input Format**: Must use ChatML format with `<image>` token
4. **Batch Processing**: Supports batch processing, but input sequence lengths should be consistent
5. **Special Token**: Must add `<image>` as special token to tokenizer before use

## Performance Optimization

- SDPA attention implementation for improved computational efficiency
- Dynamic padding to reduce computational overhead
- LoRA fine-tuning to reduce trainable parameter count
- Vision encoder and LLM base parameters frozen; only projector or LoRA parameters are trained

## Project Structure

```
lab6-nano-vlm/
├── model.py          # VLM model definition
├── train.py          # Training script
├── data_set.py       # Dataset processing
├── eval.py           # Evaluation script
├── api.py            # API interface
├── README.md         # Project documentation
└── resume/           # Resume related files
```

## Demonstration

Below are some examples demonstrating the model's capabilities on various vision-language tasks.

### Image Captioning

| | |
|---|---|
| **Image** | ![Kitchen Scene](http://images.cocodataset.org/val2017/000000397133.jpg) |
| **Prompt** | `<|im_start|>user\n<image>\n详细描述一下图片内容\n<|im_end|>\n<|im_start|>assistant\n` |
| **Response** | 这张图片展示了一位穿着黑色西装、戴着墨镜的男子正在使用一把剪刀。他看起来非常专注，似乎在进行某种工作或活动。\n\n从照片中我们可以看到他的双手紧握着剪刀，并且他在用它来切割一些东西或者物体。这可能是一个裁缝的工作台或者是某个需要精细加工的物品。\n\n此外，在背景上可以看到一个垃圾桶和几个工具箱，这些都可能是用来存放材料或其他工作的设备。\n\n整体来看，这张图片描绘 |

| | |
|---|---|
| **Image** | ![Birthday Scene](http://images.cocodataset.org/val2017/000000001000.jpg) |
| **Prompt** | `<|im_start|>user\n<image>\n描述一下这个图片\n<|im_end|>\n<|im_start|>assistant\n` |
| **Response** | 这个图片展示了一群孩子在阳光明媚的天气中，聚集在一起庆祝他们的生日。孩子们穿着鲜艳的衣服和帽子，脸上洋溢着快乐的笑容。\n\n画面中的主角是一位小男孩，他站在一个大圆桌旁，周围坐着他的父母、老师和其他家长。他们围坐在桌子旁边，享受着美食，并且互相分享彼此的故事和笑声。\n\n其他的孩子们则在一旁玩耍，有的正在画画或做手工制作，而另一些人则是阅读书籍或者玩游戏。整个 |

## License

This project is licensed under the MIT License.
