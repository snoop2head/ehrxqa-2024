# EHRXQA 2024

### Introduction
- Dataset: The model was trained on 650,000 image-caption pairs from various sources including MIMIC-CXR, NIH ChestX-ray8, ROCO, CheXpert, and OPEN-I.
- Pretraining: We employed a multimodal masked reconstruction training objective, following the methodology described in M3AE (Geng and Liu et al., preprint 2022).
- Training Specifics: The model underwent training for 1600 epochs on TPUv4-64. The weights I am sharing have been converted to PyTorch format for your convenience.
- Model Architecture: The model utilizes a 12-layer Transformer architecture with 100M parameters.
- Performance: After additional fine-tuning, we observed a positive RS5 score on the EHRXQA validation set.
- DISCLAIMER: Please note that the weights provided are pretrained and not fine-tuned for VQA. Additionally, the model does not include a classification head.

### Load Weight
```python
import torch
from model import ViT, ViTConfig
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

from dataset import CXR_Dataset, Normalize
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
