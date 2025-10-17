# LightDepth - Simple Depth Estimation

A simplified depth estimation project for CS7180 using ResNet18 encoder-decoder architecture.

## Features

- **Simple Architecture**: ResNet18 encoder with U-Net style decoder
- **Easy Training**: Straightforward training script with minimal configuration
- **L1 Loss**: Simple and effective loss function
- **RMSE Metric**: Clear evaluation metric

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or use PDM (Recommended)
pdm install
```

## Dataset

Place the NYU Depth v2 dataset in `data/nyu/` directory with the following structure:

```text
data/nyu/
├── nyu2_train.csv
├── nyu2_test.csv
├── nyu2_train/
│   ├── *.jpg
│   └── *.png
└── nyu2_test/
    ├── *.jpg
    └── *.png
```

## Usage

### Training

Basic training with default config:

```bash
python scripts/train.py --config config.yaml

# or with PDM
pdm train
```

The model will save the best checkpoint to `checkpoints/best_model.pth`.

### Evaluation

Evaluate a trained model:

```bash
python scripts/eval.py --checkpoint checkpoints/best_model.pth
```

With custom config:

```bash
python scripts/eval.py --checkpoint checkpoints/best_model.pth --config config.yaml

# or with PDM
pdm eval
```

### Inference

Run inference on a single image:

```bash
python scripts/infer.py --checkpoint checkpoints/best_model.pth --input image.jpg --output depth.png

# or with PDM
pdm infer -- --checkpoint checkpoints/best_model.pth --input image.jpg --output depth.png
```

Options:

- `--colormap`: Choose colormap (plasma, viridis, magma, etc.)
- `--device`: Use 'cuda' or 'cpu'

## Configuration

Edit `config.yaml` to change training settings:

```yaml
# Data settings
data_root: data/nyu
img_height: 480
img_width: 640

# Training settings
batch_size: 16
num_epochs: 50
learning_rate: 0.0001

# System settings
num_workers: 4
device: cuda
```

## Project Structure

```text
lightdepth/
├── src/lightdepth/
│   ├── models/          # Model architectures
│   │   ├── encoder.py   # ResNet18 encoder
│   │   ├── decoder.py   # U-Net decoder
│   │   └── depth_net.py # Complete model
│   ├── data/            # Data loading
│   │   ├── dataset.py   # NYU dataset
│   │   ├── dataloader.py
│   │   └── transforms.py
│   └── utils/           # Utilities
│       ├── config.py    # Configuration
│       ├── losses.py    # L1 loss
│       └── metrics.py   # RMSE metric
├── scripts/
│   ├── train.py         # Training script
│   ├── eval.py          # Evaluation script
│   └── infer.py         # Inference script
└── config.yaml          # Sample configuration
├── requirements.txt     # Python dependencies for pip
└── pyproject.toml       # Python dependencies for PDM
```

## Model Architecture

- **Encoder**: ResNet18 (pretrained on ImageNet)
- **Decoder**: 4-stage upsampling with skip connections
- **Channels**: [64, 64, 128, 256, 512] → [256, 128, 64, 32] → 1
- **Loss**: L1 (Mean Absolute Error)
- **Metric**: RMSE (Root Mean Squared Error)
