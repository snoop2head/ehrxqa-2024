# EHRXQA 2024

### Introduction
- The model was trained on 650,000 image-caption pairs from various sources including MIMIC-CXR, NIH ChestX-ray8, ROCO, CheXpert, and OPEN-I.
- We employed a multimodal masked reconstruction training objective, following the methodology described in [M3AE (Geng and Liu et al., preprint 2022)](https://arxiv.org/abs/2205.14204).
- **The model underwent training for 1600 epochs on TPUv4-64 using JAX/FLAX. The weights I am sharing have been converted to PyTorch format for your convenience.**
- The model utilizes a 12-layer Transformer architecture with 100M parameters.
- After additional fine-tuning, we observed a positive RS10 score on both EHRXQA validation set and test set.
- DISCLAIMER: We release our model weight on [<img width="18" alt="image" src="./assets/googledrive.png">Google Drive](https://drive.google.com/file/d/1aTgCZdMYlq0HbAMDEKJ9X_K9Nm2iMIsd/view?usp=sharing).


### Load Weight
```python
import torch
from src.modeling_pt import ViT, ViTConfig
ckpt_path = "PATH_TO_CHECKPOINT.pth"
config = ViTConfig() # To use the default configuration
model = ViT(config)
model.load_state_dict(torch.load(ckpt_path))
```
```
<All keys matched successfully>
```

### Feature Extraction
```python
import torch
from transformers import AutoTokenizer

from src.dataset import CXR_Dataset, Normalize
from model import ViT, ViTConfig

# Dataset and DataLoader
CXR_DEFAULT_MEAN = 0.4756
CXR_DEFAULT_STD = 0.3029

valid_transforms = T.Compose([
  T.Grayscale(num_output_channels=1),
  T.Resize((384, 384)),
  T.PILToTensor(),
  Normalize(CXR_DEFAULT_MEAN, CXR_DEFAULT_STD)
])

tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
valid_dataset = CXR_Dataset(
  "MIMIC-CXR-VQA/valid.json",
  image_folder="resized_ratio_short_side_768",
  transform=valid_transforms,
  tokenizer=tokenizer,
)
valid_loader = DataLoader(valid_dataset, batch_size=512, shuffle=False)

# Load model
config = ViTConfig()
model = ViT(config)
model.load_state_dict(torch.load(ckpt_path))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Feature extraction
model.eval()

with torch.no_grad():
    for img, text, _ in tqdm(val_loader):
        img = img.to(device) # torch.Size([512, 1, 384, 384])
        text = text.to(device) # torch.Size([512, 64])
        z = model(img, text)
        print(z.shape) # torch.Size([512, 768])
```

### Training


### Acknowledgements
Thanks to the TPU Research Cloud program for providing resources. Models are trained on the TPU v4-64 or TPU v4-32 pod slice.

```
@article{geng2022multimodal,
  title={Multimodal Masked Autoencoders Learn Transferable Representations},
  author={Geng, Xinyang and Liu, Hao and Lee, Lisa and Schuurams, Dale and Levine, Sergey and Abbeel, Pieter},
  journal={arXiv preprint arXiv:2205.14204},
  year={2022}
}

@inproceedings{he2022masked,
  title={Masked autoencoders are scalable vision learners},
  author={He, Kaiming and Chen, Xinlei and Xie, Saining and Li, Yanghao and Doll{\'a}r, Piotr and Girshick, Ross},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={16000--16009},
  year={2022}
}
```