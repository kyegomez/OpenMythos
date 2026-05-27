# OpenMythos-BerkahKarya

> Fork of [kyegomez/OpenMythos](https://github.com/kyegomez/OpenMythos) with consumer hardware optimizations.

## 🚀 What's New in This Fork

### Sprint 1: INT4/INT8 Quantization + Expert Offloading ✅
- **INT4/INT8 weight quantization** — 4x memory reduction for MoE expert layers
- **Expert offloading** — GPU ↔ CPU ↔ NVMe memory hierarchy
- **Consumer hardware support** — Run mythos_1b on RTX 3060 12GB

### Sprint 2: LoRA Training Pipeline ✅
- **LoRA adapters** — Fine-tune only ~0.5% of parameters
- **Colab notebook** — Free T4 GPU training (~30-60 min)
- **QLoRA mode** — INT4 + LoRA = 8GB VRAM
- **Finance demo data** — Trading, business plans, ad optimization

## 📦 Installation

```bash
git clone https://github.com/oyi77/OpenMythos.git
cd OpenMythos
pip install -e .
```

## 🎯 Quick Start

### Quantized Inference (Consumer Hardware)
```python
from open_mythos import OpenMythos, mythos_1b
from open_mythos.quantization import quantize_model
from open_mythos.expert_offloader import ExpertOffloader

model = OpenMythos(mythos_1b())
model = quantize_model(model, bits=4, group_size=128)

offloader = ExpertOffloader(model, gpu_experts=4, cache_experts=16)
offloader.prepare()
```

### LoRA Fine-tuning
```python
from open_mythos import OpenMythos, mythos_1b
from open_mythos.lora import LoRAConfig, apply_lora, save_lora_adapter

model = OpenMythos(mythos_1b())
model = apply_lora(model, LoRAConfig(rank=16, alpha=32))

# Train on your data...

save_lora_adapter(model, 'my_adapter.pt')
```

### CLI Training
```bash
# Standard LoRA (16GB VRAM)
python training/lora_finetune.py --variant 1b --dataset finance

# QLoRA (8GB VRAM, fits Colab free T4)
python training/lora_finetune.py --variant 1b --dataset finance --qlora
```

## 📊 PRs to Upstream

| PR | Feature | Status |
|----|---------|--------|
| [#74](https://github.com/kyegomez/OpenMythos/pull/74) | INT4/INT8 Quantization + Expert Offloading | Open |
| [#75](https://github.com/kyegomez/OpenMythos/pull/75) | LoRA Training Pipeline + Colab Notebook | Open |

## 🏗️ Development Roadmap

- [x] Sprint 1: INT4/INT8 Quantization + Expert Offloading
- [x] Sprint 2: LoRA Training Pipeline + Colab Notebook
- [ ] Sprint 3: Ring Attention + KV Cache Compression (1M context)
- [ ] Sprint 4: Finance Domain Fine-tuning
- [ ] Sprint 5: vLLM/GGUF Export

## 📝 License

MIT (same as upstream)

## 🤝 Contributing

1. Fork this repo
2. Create a feature branch
3. Make your changes
4. Submit a PR to upstream (kyegomez/OpenMythos)

## 🔗 Links

- [Upstream Repo](https://github.com/kyegomez/OpenMythos)
- [HuggingFace Models](https://huggingface.co/models?search=openmythos)
- [Original Paper](https://arxiv.org/abs/2502.05171) (Huginn/Raven)
