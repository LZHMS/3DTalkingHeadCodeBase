# Talking Head Codebase

<div align="center">

**A Modular and Extensible Framework for 3D Talking Head Generation Research**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

## 🎯 Overview

This repository provides a comprehensive and modular codebase for 3D talking head generation research. It implements a decoupled, trainer-based training paradigm that fully manages the entire pipeline from data loading to model evaluation, with a complete configuration management system.

**Key Features:**
- 🔧 **Modular Architecture**: Decoupled components for easy extension and customization
- 🎨 **Multi-Model Support**: Integrates various model architectures (VQ-VAE, Transformer, etc.)
- 📊 **Unified Training Framework**: Trainer-based system with full pipeline automation
- ⚙️ **Flexible Configuration**: YACS-based hierarchical configuration management
- 📈 **Experiment Tracking**: Built-in TensorBoard and WandB support
- 🚀 **Production Ready**: Comprehensive logging, checkpointing, and evaluation tools

## 📁 Project Structure

```
TalkingHeadCodebase/
├── base/                       # Core base classes
│   ├── base_config.py         # Configuration base class
│   ├── base_dataset.py        # Dataset base class
│   ├── base_datamanager.py    # Data manager base class
│   ├── base_model.py          # Model base class
│   ├── base_trainer.py        # Trainer base class
│   └── base_evaluator.py      # Evaluator base class
│
├── config/                     # Configuration files
│   ├── trainer_config.py      # Training configuration
│   └── codetalker/            # Model-specific configs
│
├── datasets/                   # Dataset implementations
│   ├── Vocaset.py             # VOCASET dataset
│   └── HDTF_TFHP.py           # HDTF-TFHP dataset
│
├── models/                     # Model implementations
│   ├── vqae.py                # VQ-VAE model
│   ├── StyleVQAE.py           # Style VQ-VAE model
│   ├── DiffPoseTalk/          # DiffPoseTalk model family
│   └── lib/                   # Model components
│       ├── base_models.py     # Transformer, Attention, etc.
│       ├── quantizer.py       # Vector quantization
│       └── head/              # Model heads
│
├── trainers/                   # Training logic
│   ├── codetalker.py          # CodeTalker trainer
│   ├── codestyle.py           # CodeStyle trainer
│   └── DiffPoseTalk/          # DiffPoseTalk trainers
│
├── evaluation/                 # Evaluation metrics
│   └── CodeTalkerEvaluator.py # Evaluation pipeline
│
├── utils/                      # Utility functions
│   ├── optim/                 # Optimizers and schedulers
│   ├── tools.py               # General utilities
│   ├── meters.py              # Metric tracking
│   └── registry.py            # Component registration
│
├── main/                       # Entry points
│   ├── train.py               # Main training script
│   └── codetalker/            # Model-specific scripts
│
└── scripts/                    # Shell scripts
    └── setup.sh               # Setup and launch script
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/LZHMS/TalkingHeadCodebase.git
cd TalkingHeadCodebase

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Train with default configuration
python main/train.py --config-file config/trainer_config.yaml

# Train with custom settings
python main/train.py \
    --config-file config/trainer_config.yaml \
    --gpu 0,1 \
    TRAINER.NAME CodeTalkerTrainer \
    DATASET.NAME HDTF_TFHP \
    OPTIM.LR 0.0001
```

### Using the Setup Script

```bash
bash scripts/setup.sh
```

## 🏗️ Architecture

### Trainer-Based Training Paradigm

The framework adopts a **decoupled trainer-based architecture** that separates concerns:

```python
# Automatic pipeline management
trainer = build_trainer(config)
trainer.train()  # Handles entire training loop
```

**Trainer responsibilities:**
- ✅ Data loading and preprocessing
- ✅ Model initialization and checkpointing
- ✅ Training loop with gradient updates
- ✅ Validation and evaluation
- ✅ Logging and visualization
- ✅ Learning rate scheduling

### Configuration System

Hierarchical configuration powered by YACS:

```yaml
# Example configuration
ENV:
  OUTPUT_DIR: ./output
  GPU: [0, 1]
  SEED: 42

MODEL:
  NAME: VQAutoEncoder
  BACKBONE:
    HIDDEN_SIZE: 768
    NUM_HIDDEN_LAYERS: 6
    NUM_ATTENTION_HEADS: 8

OPTIM:
  NAME: adamw
  LR: 0.0001
  WEIGHT_DECAY: 0.0001
  MAX_EPOCH: 100

DATASET:
  NAME: HDTF_TFHP
  ROOT: ./data
  BATCH_SIZE: 32
```

### Registry System

Component registration for easy extension:

```python
from base import TRAINER_REGISTRY

@TRAINER_REGISTRY.register()
class CustomTrainer(TrainerBase):
    def __init__(self, config):
        super().__init__(config)
        # Custom initialization
```

## 📊 Supported Models

| Model | Type | Paper | Status |
|-------|------|-------|--------|
| VQ-VAE | Vector Quantization | [Esser et al., 2021] | ✅ |
| CodeTalker | Audio-driven | [Xing et al., 2023] | ✅ |
| DiffPoseTalk | Diffusion-based | - | 🚧 |
| StyleEncoder | Style-based | - | 🚧 |

## 📈 Datasets

- **VOCASET**: Speech-driven 3D facial animation
- **HDTF-TFHP**: High-definition talking face dataset with 3D head pose

## 🛠️ Advanced Features

### Distributed Training

```bash
python -m torch.distributed.launch \
    --nprocs_per_node=4 \
    main/train.py --config-file config.yaml
```

### Experiment Tracking

Built-in support for:
- **TensorBoard**: Real-time training visualization
- **WandB**: Cloud-based experiment tracking

```python
# Automatic logging
self.write_scalar("train/loss", loss, step)
```

### Model Checkpointing

```python
# Automatic best model saving
# Resume from checkpoint
trainer.resume_model_if_exist("./checkpoint_dir")
```

## 📝 Adding New Components

### Add a New Model

```python
from base import BaseModel

class YourModel(BaseModel):
    def __init__(self, cfg):
        super().__init__()
        # Initialize your model
    
    def forward(self, x):
        # Forward pass
        return output
```

### Add a New Trainer

```python
from base import TrainerBase, TRAINER_REGISTRY

@TRAINER_REGISTRY.register()
class YourTrainer(TrainerBase):
    def build_model(self):
        # Build your model
        pass
    
    def forward_backward(self, batch):
        # Training step logic
        pass
```

### Add a New Dataset

```python
from base import DatasetBase, DATASET_REGISTRY

@DATASET_REGISTRY.register()
class YourDataset(DatasetBase):
    def __init__(self, cfg):
        # Initialize dataset
        pass
```

## 🔧 Development Guide

### Project Philosophy

This codebase follows a **registry-based modular design** where:
- All major components (models, trainers, datasets, evaluators) are registered
- Configuration is centralized and hierarchical
- Training pipeline is fully automated through trainer classes
- Easy to extend with new models and experiments

### Key Design Patterns

1. **Base Classes**: All components inherit from base classes in `base/`
2. **Registry Pattern**: Use `@REGISTRY.register()` for component discovery
3. **Configuration-Driven**: All hyperparameters managed through YACS config
4. **Decoupled Training**: Trainer handles all training logic separately from model

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- YACS for configuration management
- PyTorch team for the deep learning framework
- The talking head research community

## 📧 Contact

For questions and feedback, please open an issue or contact the maintainers.

---

<div align="center">

**⭐ Star us on GitHub if this project helps your research!**

</div>