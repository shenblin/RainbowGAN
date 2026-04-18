# 🌈 RainbowGAN
### Deep Learning Framework for Multiplex virtual histology and immunohistochemistry from a single H&E slide

![Python](https://img.shields.io/badge/python-3.6-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.4-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-research-orange.svg)

RainbowGAN is a **deep learning framework for multi-stain virtual histology translation**, enabling the conversion of a single **H&E stained image into multiple IHC or special stains** using a modified CycleGAN architecture.

The model supports **multi-style stain translation** by concatenating multiple stain channels and learning stain-to-stain mappings simultaneously.
---

# 🧠 Overview

Conventional pathology workflows require **multiple physical staining procedures** (IHC, PAS, Masson's trichrome, etc.), which consume additional tissue sections and time.

RainbowGAN enables:

➡ **Single H&E → multiple virtual stains**

using a **multi-channel GAN translation model**.

Key features:

- 🌈 Multi-stain translation in a single model
- 🔄 Cycle-consistency learning
- 🧬 Suitable for histopathology datasets
- ⚡ Efficient inference on whole slide tiles

---

# 🏗 Architecture

RainbowGAN extends the **CycleGAN framework** with a **multi-style representation**.

Core components:

| Component | Description |
|-----------|-------------|
| Generator \(G_{AB}\) | H&E → multi-stain translation |
| Generator \(G_{BA}\) | reverse mapping |
| Discriminator \(D_A\) | real vs generated H&E |
| Discriminator \(D_B\) | real vs generated stains |

The model optimizes: Adversarial Loss, Cycle Consistency Loss, Identity Loss, allowing **structure-preserving stain translation**.


# ⚙️ Installation

## 1️⃣ Clone repository

```bash
git clone https://github.com/yourusername/RainbowGAN.git
cd RainbowGAN
```

## 2️⃣ Create environment
Conda (recommended)
```bash
conda env create -f environment.yml
conda activate RainbowGAN

pip install -r requirements.txt
```

## 📊 Dataset Structure

Datasets follow a multi-style folder format:

datasets/
- train_dataset/
  - multi_stain_colorectum/
    - trainA/
      - Style_1/
      - Style_2/
      - Style_3/
    - trainB/
      - Style_1/
      - Style_2/
      - Style_3/

Where:

| Folder | Meaning                         |
|--------|---------------------------------|
| trainA | input stain (e.g., H&E)     |
| trainB | output (IHC or special stains)|

# 🚀 Training

Example: Colorectal multi-stain model (6 stains)
```bash
python train.py \
--dataroot datasets/train_dataset/multi_stain_colorectum/Other \
--name multi_stain_colorectum_split \
--model rainbow_gan \
--dataset_mode one2onestyle \
--num_style 6 \
--input_nc 18 \
--output_nc 18
```
Explanation:

```markdown
| Parameter | Meaning              | Value (example) |
|-----------|----------------------|------------------|
| `num_style` | number of stains   | 4                |
| `input_nc`  | channels = 3 × num_style | 12           |
| `output_nc` | channels = 3 × num_style | 12           |
```

# 🔬 Inference

Generate virtual stains:
```bash
python test.py \
--dataroot datasets/test_dataset/multi_stain_colorectum/Other \
--name multi_stain_colorectum_split \
--model rainbow_gan \
--phase test \
--dataset_mode one2onestyle \
--num_style 6 \
--input_nc 18 \
--output_nc 18 \
--results_dir result_tiles/
```

If you use RainbowGAN in your research, please cite:
@article{rainbowgan_virtual_staining,
  title={Virtual multiplex staining of tissue and cytology specimens via RainbowGAN},
  author={Shen, Binglin and others},
  journal={},
  year={2026}
}

CycleGAN reference:
@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={ICCV},
  year={2017}
}
