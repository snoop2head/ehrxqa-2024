# Copyright 2024 Jungwoo Park (affjljoo3581) and Young Jin Ahn (snoop2head)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from dataclasses import dataclass, fields

import numpy as np

@dataclass
class ViTConfig:
    layers: int = 12
    dim: int = 768
    heads: int = 12

    patch_size: int = 16
    image_size: int = 384 # 24 x 24 patches

    dropout: float = 0.0
    droppath: float = 0.0

    max_len: int = 420 # pretrained max text length
    truncation_len: int = 64 # truncation required for VQA to save memory. can be set lower, but should match padded input text length.
    text_vocab_size: int = 28996 # dmis-lab/biobert-base-cased-v1.1 vocab size (equivalent to bert-base-cased vocab size)

    @property
    def head_dim(self) -> int:
        return self.dim // self.heads

    @property
    def hidden_dim(self) -> int:
        return 4 * self.dim

    @property
    def num_patches(self) -> tuple[int, int]:
        return (self.image_size // self.patch_size,) * 2


def fixed_sincos2d_embeddings(ncols: int, nrows: int, dim: int) -> torch.Tensor:
    freqs = 1 / (10000 ** np.linspace(0, 1, dim // 4))
    x = np.outer(np.arange(0, nrows, dtype=np.float32), freqs)
    y = np.outer(np.arange(0, ncols, dtype=np.float32), freqs)

    x = np.broadcast_to(x[None, :, :], (ncols, nrows, dim // 4))
    y = np.broadcast_to(y[:, None, :], (ncols, nrows, dim // 4))
    return torch.tensor(np.concatenate((np.sin(x), np.cos(x), np.sin(y), np.cos(y)), axis=2), dtype=torch.float32)

def fixed_sincos1d_embeddings(max_len: int, dim: int) -> torch.Tensor:
    freqs = 1 / (10000 ** np.linspace(0, 1, dim // 2))
    pos = np.arange(0, max_len, dtype=np.float32)[:, None]
    pos_enc = np.dot(pos, freqs[None, :])

    sin_enc = np.sin(pos_enc)
    cos_enc = np.cos(pos_enc)
    embeddings = np.concatenate([sin_enc, cos_enc], axis=1)
    return torch.tensor(embeddings, dtype=torch.float32)


class PatchEmbed(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.wte = nn.Conv2d(in_channels=1, out_channels=config.dim, kernel_size=config.patch_size, stride=config.patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.dim))
        self.wpe = nn.Parameter(fixed_sincos2d_embeddings(*config.num_patches, config.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.wte(x).flatten(2).transpose(1, 2)
        x += self.wpe.flatten(0, 1)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        return x

class TextEmbed(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.wte = nn.Embedding(config.text_vocab_size, config.dim)
        self.pos_emb = nn.Parameter(fixed_sincos1d_embeddings(config.max_len, config.dim))
        self.truncation_len = config.truncation_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.truncation_len == x.size(1), f"Truncation length mismatch: {self.truncation_len} != {x.size(1)}"
        return self.wte(x) + self.pos_emb[:x.size(1), :]

class Attention(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config
        self.wq = nn.Linear(config.dim, config.dim)
        self.wk = nn.Linear(config.dim, config.dim)
        self.wv = nn.Linear(config.dim, config.dim)
        self.wo = nn.Linear(config.dim, config.dim)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.wq(x).unflatten(2, (self.config.heads, -1)).permute(0, 2, 1, 3)
        k = self.wk(x).unflatten(2, (self.config.heads, -1)).permute(0, 2, 1, 3)
        v = self.wv(x).unflatten(2, (self.config.heads, -1)).permute(0, 2, 1, 3)
        x = F.scaled_dot_product_attention(q, k, v).permute(0, 2, 1, 3).flatten(2)
        return self.drop(self.wo(x))

class FeedForward(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.hidden_dim)
        self.w2 = nn.Linear(config.hidden_dim, config.dim)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w1(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.w2(x)
        x = self.drop(x)
        return x

class ViTLayer(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.attn = Attention(config)
        self.ff = FeedForward(config)

        self.norm1 = nn.LayerNorm(config.dim)
        self.norm2 = nn.LayerNorm(config.dim)
        self.drop = nn.Dropout(config.droppath)

        self.scale1 = self.scale2 = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.scale1 * self.attn(self.norm1(x)))
        x = x + self.drop(self.scale2 * self.ff(self.norm2(x)))
        return x

class ViT(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.image_embed = PatchEmbed(config)
        self.text_embed = TextEmbed(config)
        self.drop = nn.Dropout(config.dropout)

        self.encoder_image_type_embedding = nn.Parameter(torch.zeros(1, 1, config.dim))
        self.encoder_text_type_embedding = nn.Parameter(torch.zeros(1, 1, config.dim))

        self.layer = nn.ModuleList([ViTLayer(config) for _ in range(config.layers)])
        self.norm = nn.LayerNorm(config.dim)

    def forward(self, images: torch.Tensor, texts: torch.Tensor) -> torch.Tensor:
        # Image Embedding
        images_x = self.image_embed(images)
        cls_token, images_x = images_x[:, :1, :], images_x[:, 1:, :]
        images_x += self.encoder_image_type_embedding

        # Text Embedding
        texts_x = self.text_embed(texts)
        texts_x += self.encoder_text_type_embedding

        x = torch.cat((cls_token, images_x, texts_x), dim=1)

        x = self.drop(x)
        for layer in self.layer:
            x = layer(x)
        x = self.norm(x)

        return x
