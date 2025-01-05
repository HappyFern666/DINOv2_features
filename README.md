# DINOv2_features

This repository provides a feature extraction tool for DINOv2. The implementation is inspired by discussions and suggestions from [this GitHub issue](https://github.com/facebookresearch/dinov2/issues/23).

## Environment Setup

To set up the environment, you can use the `requirements.txt` file provided in this repository. It contains all the necessary Python packages.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/HappyFern666/DINOv2_features.git
   cd DINOv2_features
   ```

2. Install the required packages:

   ```bash
   conda create -n dinov2 python=3.9
   conda activate dinov2
   pip install -r requirements.txt
   ```

## Usage

To extract features, run the `extract_feature.py` script:

```bash
python dinov2_feature/extract_feature.py
```

And I refer to advice in [Github issue](https://github.com/facebookresearch/dinov2/issues/23#issuecomment-1587650712), and get another code.

```bash
python dinov2_feature/2extract_feature.py
```
