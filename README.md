# SoilNet-Hybrid: Soil Image Classification with MambaOut and TinyViT Fusion

## 📖 Overview

SoilNet-Hybrid is an advanced deep learning model specifically designed for soil image classification tasks. This project provides a complete training pipeline that integrates MambaOut's gated convolution mechanism with TinyViT's attention advantages, featuring innovative data augmentation strategies and comprehensive performance monitoring.

## 🎯 Key Features

### 🌟 Model Innovations
- **Dual-Branch Hybrid Architecture**: Local features (MambaOut) + Global context (TinyViT)
- **Feature Selection Module (FSM)**: Dynamic selection of important feature channels and spatial regions
- **Gated Fusion Mechanism**: Adaptive weighted fusion of multi-scale features
- **Lightweight Design**: Optimized parameter efficiency for resource-constrained environments

### 🚀 Training Advantages
- **Ready-to-Run**: Complete training code that works directly in Jupyter Notebook
- **Advanced Data Augmentation**: Mixup, CutMix, and Random Erasing strategies
- **Mixed Precision Training**: Faster training with GPU optimization
- **Comprehensive Metrics**: Real-time monitoring of multiple performance indicators
- **Automatic Checkpointing**: Best model saving and detailed logging

## 📊 Performance Highlights

| Model | Parameters | Accuracy | Key Features |
|-------|------------|----------|--------------|
| Mambaout-T | ~24.25M | 95.5%(data1) 94.9%(data2) | Traditional convolutional network |
| **SoilNet-Hybrid** | **~1.6M** | **95.5%(data1) 96.4%(data2)** | Lightweight and efficient |
| TinyViT | ~5.4M | 93.7%(data1) 94.6%(data2) | Computationally intensive |

## 🚀 Quick Start

### Prerequisites

```bash
# Install required packages
pip install umap-learn  
pip install seaborn
pip3 install tqdm==4.66.4 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install pandas -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install pandas==1.5.3 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -U scikit-learn
pip install wandb -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install numpy==1.26.4 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install plotly
pip install torchsummary
pip install onnxsim
pip install timm
pip install fpdf2
pip install omegaconf
pip install fvcore
pip install thop
pip install ptflops
pip install einops
Data Preparation
Organize your dataset in the following structure:

复制代码
soil/
├── train/
│   ├── class_1/
│   ├── class_2/
│   └── ...
└── val/
    ├── class_1/
    ├── class_2/
    └── ...
Direct Execution
Simply run the training script in your Jupyter Notebook:


# The training code is designed to run directly
# Just execute the main() function in train.py
Or run from command line:


python train.py
⚙️ Training Configuration
The model uses the following optimized hyperparameters:


config = {
    'data_path': 'soil',           # Dataset directory
    'num_classes': 6,              # Number of soil classes
    'epochs': 300,                 # Training epochs
    'batch_size': 32,              # Batch size
    'lr': 3e-4,                    # Learning rate
    'weight_decay': 1e-4,          # Weight decay for regularization
    'mixup_alpha': 0.8,            # Mixup augmentation parameter
    'cutmix_alpha': 1.0,           # CutMix augmentation parameter
    'mix_prob': 0.8,               # Probability of applying augmentation
    'erase_prob': 0.25,            # Random erasing probability
}
🔧 Advanced Features
Data Augmentation Strategies
Mixup & CutMix Implementation:

Mixup: Linear interpolation between images and labels
CutMix: Patch replacement with label mixing
Adaptive Switching: Random selection between augmentation methods
Random Erasing:

Random occlusion of image regions
Configurable scale and ratio parameters
Enhances model robustness to occlusions
Performance Optimization
Mixed Precision Training: Accelerated training with GPU support
Cosine Annealing LR: Optimal learning rate scheduling
Comprehensive Metrics: Loss, Accuracy, F1-score, Precision, Recall
📊 Results and Logging
The training script automatically generates:

Model Checkpoints: Best and final models
Training Metrics: CSV files with epoch-wise performance
Classification Reports: Detailed per-class analysis
Confusion Matrices: Visual error analysis
Time Statistics: Performance benchmarks
📧 Data Availability Statement
The soil image dataset used in this study is currently not publicly available due to ongoing research and privacy considerations.

For academic collaboration or data access requests, please contact:

Email: 1971777601@qq.com
We are open to collaboration and data sharing for legitimate academic research purposes. Please include your affiliation and research objectives in your inquiry.

📝 Citation
If you use this code in your research, please acknowledge the source:



@software{SoilNetHybrid,
  title = {SoilNet-Hybrid: Soil Image Classification with MambaOut and TinyViT Fusion},
  author = {ZhangYan},
  year = {2025},
  url = {https://github.com/your-repo/soilnet-hybrid}
}


🤝 Contributing
Contributions are welcome! Please feel free to submit issues and pull requests to improve this project.

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

Note: This implementation combines MambaOut and TinyViT architectures with our novel Feature Selection Module, achieving superior performance with significantly fewer parameters compared to existing methods.

