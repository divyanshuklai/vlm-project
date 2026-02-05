# vlm-project 🚀

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
>
> A lightweight, highly modular, and hackable framework for training, finetuning, and deploying state-of-the-art Vision-Language Models (VLMs). Designed for researchers and engineers who want full control over the VLM stack without the bloat.

## 🌟 Key Features

vlm-project provides a unified interface for the entire VLM lifecycle:

*   **⚡ Efficient Pretraining**: Optimized data loading and masking strategies for large-scale image-text pretraining.
*   **🔧 Advanced Fine-tuning**:
    *   **SFT**: Standard Supervised Fine-Tuning.
    *   **RLHF**: Reinforcement Learning from Human Feedback.
    *   **DPO**: Direct Preference Optimization (Memory efficient!).
    *   **GRPO (RLVR)**: Group Relative Policy Optimization for verified reasoning tasks.
*   **📦 Quantization & Deployment**:
    *   Native support for **4-bit** and **8-bit** loading via `bitsandbytes`.
    *   Export to **ONNX** and **GGUF** for edge deployment.
    *   High-throughput serving server included.
*   **📊 Simple Evaluation**:
    *   Integrated directly with **VLMEvalKit**.
    *   Supports benchmarks: `RefCOCO`, `POPE`, `GQA`, `DocVQA`, `ChartQA`, and `CountBenchQA`.
*   **🧱 Modular Architecture**:
    *   **Vision Encoders**: Plug-and-play support for DINOv3, SigLIP, CLIP.
    *   **LLM Backends**: Seamless integration with Gemma 2, Qwen 2.5, SmolVLM, and Llama 3.

## 🛠️ Supported Models

We support a wide range of architectures out-of-the-box. If it's on HuggingFace, it likely works here.

| Vision Encoder | LLM Backbone |
| :--- | :--- |
| **DINOv3** (ViT-B/16, ViT-S/16) | **SmolVLM** (256M, 2.2B) |
| **SigLIP 2** (Patch32) | **Gemma 2** (2B, 9B, 27B) |
| **CLIP** (ViT-Base) | **Qwen 2.5 / 2.5-VL** (0.5B, 7B, 72B) |
| **vlm-project** (Custom 222M) | **Llama 3.1** (8B, 70B) |

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/your-username/vlm-project.git
cd vlm-project
pip install -r requirements.txt
```

### 1. Training

Pretrain or fine-tune your model using our unified CLI. The framework automatically handles distributed training (FSDP/DDP).

```bash
# Standard Fine-tuning
python src/train.py \
    --model_type "smolvlm" \
    --dataset "coco_captions" \
    --batch_size 32 \
    --lr 2e-5 \
    --epochs 3

# Align with DPO
python src/train.py \
    --stage "dpo" \
    --model_path "checkpoints/sft_model" \
    --dataset "preference_data"
```

### 2. Reinforcement Learning (RLHF/GRPO)

Unlock reasoning capabilities with our RL pipeline.

```python
from src.train import RLTrainer
from src.config import RLConfig

config = RLConfig(
    method="grpo",  # Group Relative Policy Optimization
    reward_model="cost_model_v1",
    kl_coeff=0.01
)

trainer = RLTrainer(model, config)
trainer.train()
```

### 3. Inference & Chat

```python
from src.model import VLM
from PIL import Image

model = VLM.from_pretrained("lusxvr/nanoVLM-222M", quantization="4bit")
image = Image.open("examples/cat.jpg")

response = model.chat(
    image=image,
    text="Describe the lighting in this image and how it affects the mood.",
    max_new_tokens=256
)
print(response)
```

### 4. Evaluation Strategy

We take evaluation seriously. Benchmarking is just one command away:

```bash
# Evaluate on POPE, GQA and DocVQA
python src/eval.py \
    --model "checkpoints/best_model" \
    --benchmarks "POPE,GQA,DocVQA" \
    --batch_size 16
```

*Results are automatically logged to `outputs/` and formatted for leaderboard submission.*

## 📂 Project Structure

```
vlm-project/
├── src/
│   ├── model.py       # Architecture definitions (Connector + LLM + Vision)
│   ├── train.py       # Unified training loop (SFT, DPO, PPO)
│   ├── eval.py        # Evaluation pipeline via VLMEvalKit
│   ├── dataset.py     # High-performance data loaders
│   └── config.py      # Hydra-based configuration
├── libs/
│   └── VLMEvalKit/    # Submodule for SOTA evaluation
└── scripts/           # Deployment and utility scripts
```

## 🤝 Contributing

We love contributions! Please check out `CONTRIBUTING.md` for guidelines on how to add new vision backends or training recipes.

## 📜 References

- **VLMEvalKit**: https://github.com/open-compass/VLMEvalKit
```bibtex
@inproceedings{duan2024vlmevalkit,
  title={Vlmevalkit: An open-source toolkit for evaluating large multi-modality models},
  author={Duan, Haodong and Yang, Junming and Qiao, Yuxuan and Fang, Xinyu and Chen, Lin and Liu, Yuan and Dong, Xiaoyi and Zang, Yuhang and Zhang, Pan and Wang, Jiaqi and others},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={11198--11201},
  year={2024}
}
```

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.