# Structured Coding for 3D Talking Head Codebase

<div align="center">

**A Modular and Extensible Framework for 3D Talking Head Generation Research**

**⭐ Star us on GitHub if this project helps your research!**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

## 🎯 Overview
This repository provides a foundational framework for any AI model training project. It serves as a base for **accumulating and reusing essential model code, enabling rapid development of custom modules and avoiding reinventing the wheel**. 

---

The framework adopts a decoupled trainer architecture that automatically manages the entire pipeline—from data loading to model evaluation—with a robust configuration management system. 

---

By embracing structured programming, **complex code is divided into independent modules, greatly improving code standardization, maintainability, and readability**.

---

**Key Features:**
- 🔧 **Modular Architecture**: Decoupled components for easy extension and customization
- 🎨 **DiffPoseTalk Model**: Implements diffusion-based talking head generation with style encoding
- 📊 **Unified Training Framework**: Trainer-based system with full pipeline automation
- ⚙️ **Flexible Configuration**: YACS-based hierarchical configuration management
- 📈 **Experiment Tracking**: Built-in TensorBoard and WandB support
- 🚀 **Production Ready**: Comprehensive logging, checkpointing, and evaluation tools

> [!NOTE]
>
> This project currently implements state-of-the-art (SOTA) methods for 3D talking head generation, specifically the **DiffPoseTalk** model. We are actively developing our own research methods to further advance the field.

> [!NOTE]
>
> This project is modified from **Dassl**, making it more user-friendly and structured. It includes additional modules tailored for 3D Talking Head research, such as datasets for 3D Talking Head studies and FLAME-based rendering components.

## 🗒️ TODO Plan

- [ ] Support for audio_visual dataset collection module
    - Design and implement audio-visual data collection workflow
    - Provide data annotation and preprocessing tools
    - Integrate with existing data management and training pipeline

## 📁 Project Structure

```
3DTalkingHeadCodeBase/
├── base/                      # Core base classes
│   ├── base_config.py         # Configuration base class
│   ├── base_dataset.py        # Dataset base class
│   ├── base_datamanager.py    # Data manager base class
│   ├── base_model.py          # Model base class
│   ├── base_trainer.py        # Trainer base class
│   └── base_evaluator.py      # Evaluator base class
│
├── config/                     # Configuration files
│   ├── difftalk_trainer_config.yaml  # DiffPoseTalk trainer config
│   └── style_trainer_config.yaml     # Style encoder trainer config
│
├── dataset/                   # Dataset implementations
│   └── HDTF_TFHP.py           # HDTF-TFHP dataset
│
├── models/                    # Model implementations
│   ├── diffposetalk.py        # DiffPoseTalk model
│   ├── avatar/                # Avatar related modules
│   │   ├── flame.py           # FLAME head model
│   │   └── lbs.py             # Linear blend skinning
│   └── lib/                   # Model components
│       ├── base_models.py     # Transformer, Attention, etc.
│       ├── common.py          # Common utilities
│       ├── quantizer.py       # Vector quantization
│       ├── audio/             # Audio feature extractors
│       ├── head/              # Head model components
│       └── network/           # Network architectures
│
├── trainers/                   # Training logic
│   └── diffposetalk_trainer.py # DiffPoseTalk trainer
│
├── evaluator/                # Evaluators
│   └── TalkerEvaluator.py     # Talking head evaluator
│
├── utils/                      # Utility functions
│   ├── optim/                 # Optimizers and schedulers
│   ├── tools.py               # General utilities
│   ├── meters.py              # Metric tracking
│   ├── registry.py            # Component registration
│   ├── loss.py                # Loss functions
│   ├── media.py               # Media utilities
│   └── renderer.py            # Rendering utilities
│
├── scripts/                    # Shell scripts
│   ├── style_train.sh         # Style encoder training script
│   └── talker_train.sh        # Talker training script
│
├── data/                       # Data directory
│   └── HDTF_TFHP/             # HDTF-TFHP dataset files
│
├── output/                     # Training outputs
│   └── HDTF_TFHP/             # Output for HDTF-TFHP experiments
│
├── pretrained/                 # Pretrained models
├── train.py                    # Main training entry point
├── environment.yml            # Conda environment file
└── requirements.txt           # Python dependencies
```

## 📁 Trainer Architecture

```
TrainerBase
├── config
│   ├── check_cfg
│   └── system_init
├── data
│   ├── build_data_loader
│   ├── DataManager
│   │   ├── DatasetBase
│   │   ├── DatasetWrapper
│   │   ├── show_dataset_summary
│   │   └── data_analysis
├── model
│   ├── build_model
│   ├── get_model_names
│   ├── register_model
│   └── set_model_mode
├── writer
│   ├── init_writer
│   ├── write_scalar
│   └── close_writer
├── train
│   ├── parse_batch_train
│   ├── before_train
│   ├── train_epoch
│   │   ├── before_epoch
│   │   ├── run_epoch
│   │   └── after_epoch
│   ├── train_iter
│   │   ├── before_iter
│   │   ├── run_iter
│   │   └── after_iter
│   ├── forward_backward
│   └── after_train
├── optim
│   ├── build_optimizer
│   ├── build_lr_scheduler
│   ├── model_backward_and_update
│   │   ├── model_zero_grad
│   │   ├── model_backward
│   │   └── model_update
│   ├── update_lr
│   └── get_current_lr
├── test
│   └── parse_batch_test
├── evaluator
│   └── build_loss_metrics
├── save_load
│   ├── save_model
│   ├── save_checkpoint
│   ├── load_model
│   ├── load_checkpoint
│   ├── load_pretrained_weights
│   ├── resume_model_if_exist
│   └── resume_from_checkpoint
└── tools
    ├── detect_anomaly
    └── count_num_param
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/LZHMS/3DTalkingHeadCodeBase.git
cd 3DTalkingHeadCodeBase

# Create conda environment
conda create -n talkinghead python=3.9
conda activate talkinghead

# Or use the provided environment file
conda env create -f environment.yml
conda activate talkinghead

# Install PyTorch (adjust for your CUDA version)
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Train DiffPoseTalk with default configuration
python train.py --config-file config/difftalk_trainer_config.yaml

# Train Style Encoder
python train.py --config-file config/style_trainer_config.yaml

# Train with custom settings
python train.py \
    --config-file config/difftalk_trainer_config.yaml \
    --gpu 0,1 \
    OPTIM.LR 0.0001
```

### Using the Training Scripts

```bash
# Train style encoder
bash scripts/style_train.sh

# Train talking head model
bash scripts/talker_train.sh
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
  NAME: VOCASET
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
|-------|------|-------|---------|
| DiffPoseTalk | Diffusion + Style | [Sun et al., 2024] | ✅ |

## 📈 Datasets

| Dataset | Description | Subjects | Status |
|---------|-------------|----------|---------|
| HDTF-TFHP | High-definition talking face with 3D head pose | - | ✅ |

## 🛠️ Advanced Features

### Distributed Training

```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    train.py --config-file config/difftalk_trainer_config.yaml
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
from base import BaseModel, MODEL_REGISTRY

@MODEL_REGISTRY.register()
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

## 📖 Citation

If you find this codebase useful for your research, please consider citing:

```bibtex
@misc{3DTalkingHeadCodeBase,
  author       = {Zhihao Li},
  title        = {3DTalkingHeadCodeBase: A Modular Framework for 3D Talking Head Generation},
  year         = {2025},
  publisher    = {GitHub},
  journal      = {GitHub Repository},
  howpublished = {\url{https://github.com/LZHMS/3DTalkingHeadCodeBase}}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch) for the foundational training framework architecture
- [DiffPoseTalk](https://github.com/DiffPoseTalk/DiffPoseTalk) for diffusion-based methods
- YACS for configuration management
- PyTorch team for the deep learning framework
- The talking head research community

## 📧 Contact

For questions and feedback, please open an issue or contact the maintainers.