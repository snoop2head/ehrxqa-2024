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

from __future__ import annotations

from dataclasses import dataclass, fields
from functools import partial
from typing import Any, Literal

import flax.linen as nn
import flax.linen.initializers as init
import jax
import jax.numpy as jnp
from chex import Array

from utils import fixed_sincos2d_embeddings, fixed_sincos1d_embeddings
from utils_mae import (
  mask_union, mask_intersection, mask_not, mask_select, no_mask, all_mask, # masking related
  extract_patches, merge_patches, index_sequence, random_masking # patch related
)

DenseGeneral = partial(nn.DenseGeneral, kernel_init=init.truncated_normal(0.02))
Dense = partial(nn.Dense, kernel_init=init.truncated_normal(0.02))
Conv = partial(nn.Conv, kernel_init=init.truncated_normal(0.02))
Embed = partial(nn.Embed, embedding_init=init.truncated_normal(0.02))

@dataclass
class ViTBase:
    layers: int = 12
    dim: int = 768
    heads: int = 12
    labels: int | None = 1000
    layerscale: bool = False

    patch_size: int = 16
    image_size: int = 224
    posemb: Literal["learnable", "sincos2d"] = "learnable"
    pooling: Literal["cls", "gap"] = "cls"

    dropout: float = 0.0
    droppath: float = 0.0
    grad_ckpt: bool = False
    
    # MAE related
    image_mask_ratio: float = 0.75
    linear_probing: bool = False
    batch_norm: bool = False

    # M3AE related
    max_len: int = 420
    truncation_len: int = 64
    text_mask_ratio: float = 0.75
    text_vocab_size: int = 30522 # bert-base-uncased vocab size

    @property
    def kwargs(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(ViTBase)}

    @property
    def head_dim(self) -> int:
        return self.dim // self.heads

    @property
    def hidden_dim(self) -> int:
        return 4 * self.dim

    @property
    def num_patches(self) -> tuple[int, int]:
        return (self.image_size // self.patch_size,) * 2

    # M3AE Related
    @property
    def max_patch_num(self) -> int:
        return self.num_patches[0] * self.num_patches[1]

    @property
    def image_keep_length(self) -> int:
        return int(self.max_patch_num * (1.0 - self.image_mask_ratio))

    @property
    def text_keep_length(self) -> int:
        return int(self.max_len * (1.0 - self.text_mask_ratio))

    @property
    def masked_patch_num(self) -> int:
        return self.max_patch_num - self.image_keep_length

    @property
    def masked_len(self) -> int:
        return self.max_len - self.text_keep_length



@dataclass
class MAEDecoderBase:
    dec_layers: int = 6
    dec_dim: int = 512
    dec_heads: int = 8
    dec_layerscale: bool = False

    dec_posemb: Literal["learnable", "sincos2d"] = "learnable"

    dec_dropout: float = 0.0
    dec_droppath: float = 0.0
    grad_ckpt: bool = False

    patch_size: int = 16
    image_size: int = 224

    # M3AE related
    max_len: int = 64

    @property
    def kwargs(self) -> dict[str, Any]:
        decoder_kwargs = {f.name: getattr(self, f.name) for f in fields(MAEDecoderBase)}
        return {k.replace("dec_", ""): v for k, v in decoder_kwargs.items()} # replace "dec_" with "" for the decoder kwargs
    
    @property
    def head_dim(self) -> int:
        return self.dec_dim // self.dec_heads

    @property
    def hidden_dim(self) -> int:
        return 4 * self.dec_dim

    @property
    def num_patches(self) -> tuple[int, int]:
        return (self.image_size // self.patch_size,) * 2

    @property
    def max_patch_num(self) -> int:
        return self.num_patches[0] * self.num_patches[1]

class PatchEmbed(ViTBase, nn.Module):
    def setup(self):
        self.wte = Conv(
            self.dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
        )
        if self.pooling == "cls":
            self.cls_token = self.param(
                "cls_token", init.truncated_normal(0.02), (1, 1, self.dim)
            )

        if self.posemb == "learnable":
            self.wpe = self.param(
                "wpe", init.truncated_normal(0.02), (*self.num_patches, self.dim)
            )
        elif self.posemb == "sincos2d":
            self.wpe = fixed_sincos2d_embeddings(*self.num_patches, self.dim)
            if self.wpe.shape != (24, 24, self.dim): # 384 // 16 = 24
                self.wpe = jax.image.resize(self.wpe, (*self.num_patches, self.dim), method="bicubic")
                print(f"Resized positional embeddings to {self.num_patches}")
                
    def __call__(self, x: Array) -> Array:
        x = (self.wte(x) + self.wpe).reshape(x.shape[0], -1, self.dim)
        if self.pooling == "cls":
            cls_token = jnp.repeat(self.cls_token, x.shape[0], axis=0)
            x = jnp.concatenate((cls_token, x), axis=1)
        return x

class TextEmbed(ViTBase, nn.Module):
    def setup(self):
        self.wte = Embed(self.text_vocab_size, self.dim)
        self.pos_emb = fixed_sincos1d_embeddings(self.max_len, self.dim)
        if self.max_len != self.truncation_len:
            print(f"Truncating positional embeddings to {self.truncation_len}")
            self.pos_emb = self.pos_emb[:self.truncation_len, :]

    def __call__(self, x: Array) -> Array:
        return self.wte(x) + self.pos_emb

class Attention(ViTBase, nn.Module):
    def setup(self):
        self.wq = DenseGeneral((self.heads, self.head_dim))
        self.wk = DenseGeneral((self.heads, self.head_dim))
        self.wv = DenseGeneral((self.heads, self.head_dim))
        self.wo = DenseGeneral(self.dim, axis=(-2, -1))
        self.drop = nn.Dropout(self.dropout)

    def __call__(self, x: Array, det: bool = True) -> Array:
        z = jnp.einsum("bqhd,bkhd->bhqk", self.wq(x) / self.head_dim**0.5, self.wk(x))
        z = jnp.einsum("bhqk,bkhd->bqhd", self.drop(nn.softmax(z), det), self.wv(x))
        return self.drop(self.wo(z), det)


class FeedForward(ViTBase, nn.Module):
    def setup(self):
        self.w1 = Dense(self.hidden_dim)
        self.w2 = Dense(self.dim)
        self.drop = nn.Dropout(self.dropout)

    def __call__(self, x: Array, det: bool = True) -> Array:
        return self.drop(self.w2(self.drop(nn.gelu(self.w1(x)), det)), det)


class ViTLayer(ViTBase, nn.Module):
    def setup(self):
        self.attn = Attention(**self.kwargs)
        self.ff = FeedForward(**self.kwargs)

        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.drop = nn.Dropout(self.droppath, broadcast_dims=(1, 2))

        self.scale1 = self.scale2 = 1.0
        if self.layerscale:
            self.scale1 = self.param("scale1", init.constant(1e-4), (self.dim,))
            self.scale2 = self.param("scale2", init.constant(1e-4), (self.dim,))

    def __call__(self, x: Array, det: bool = True) -> Array:
        x = x + self.drop(self.scale1 * self.attn(self.norm1(x), det), det)
        x = x + self.drop(self.scale2 * self.ff(self.norm2(x), det), det)
        return x


class LinearCLS(nn.Module):
    @nn.compact
    def __call__(self, x: Array, det: bool) -> Array:
        x = nn.BatchNorm(use_running_average=det, axis_name="batch")(x)
        x = nn.Dense(1000, kernel_init=init.truncated_normal(0.02))(x)
        return x

class ViT(ViTBase, nn.Module):
    def setup(self):
        self.image_embed = PatchEmbed(**self.kwargs)
        self.text_embed = TextEmbed(**self.kwargs)
        self.drop = nn.Dropout(self.dropout)

        self.encoder_image_type_embedding = self.param("encoder_image_type_embedding", init.truncated_normal(0.02), (1, 1, self.dim))
        self.encoder_text_type_embedding = self.param("encoder_text_type_embedding", init.truncated_normal(0.02), (1, 1, self.dim))

        # The layer class should be wrapped with `nn.remat` if `grad_ckpt` is enabled.
        layer_fn = nn.remat(ViTLayer) if self.grad_ckpt else ViTLayer
        self.layer = [layer_fn(**self.kwargs) for _ in range(self.layers)]

        self.norm = nn.LayerNorm()
        
        # head        
        if self.labels > 0:
            if self.batch_norm:
                self.head = LinearCLS()
            else:
                self.head = Dense(self.labels)
        else:
            self.head = None


    def __call__(self, images: Array, texts: Array, validation: bool = False, det: bool = True) -> Array:
        # Image Embedding
        images_x = self.image_embed(images)
        cls_token, images_x = images_x[:, :1, :], images_x[:, 1:, :] # x[:, 0, :] decreases dimension
        images_x += self.encoder_image_type_embedding

        # Text Embedding
        texts_x = self.text_embed(texts)
        texts_x += self.encoder_text_type_embedding

        # M3AE + PreTraining
        if self.head is None:

            # Image random masking
            images_x, image_mask, image_ids_restore = random_masking(images_x, self.make_rng("image_noise"), self.image_keep_length)                
            
            # Text random masking
            texts_x, text_mask, text_ids_restore = random_masking(texts_x, self.make_rng("text_noise"), self.text_keep_length)
            
            x = jnp.concatenate((cls_token, images_x), axis=1) if validation else jnp.concatenate((cls_token, images_x, texts_x), axis=1)
        elif self.head is not None:
            x = jnp.concatenate((cls_token, images_x, texts_x), axis=1)

        x = self.drop(x, det)
        for layer in self.layer:
            x = layer(x, det)
        x = self.norm(x)

        # If the classification head is not defined, then return the output of all
        # tokens instead of pooling to a single vector and then calculate class logits.
        if self.head is None: # MAE for both training and inference, forward_encoder
            return x, image_mask, text_mask, image_ids_restore, text_ids_restore

        if self.linear_probing:
            x = jax.lax.stop_gradient(x)

        if self.pooling == "cls":
            x = x[:, 0, :]
        elif self.pooling == "gap":
            x = x.mean(1)
        
        if self.batch_norm:
            return self.head(x, det)
        else:
            return self.head(x)

class MAEDecoder(MAEDecoderBase, nn.Module):
    def setup(self):
        layer_fn = nn.remat(ViTLayer) if self.grad_ckpt else ViTLayer
        self.dec_layer = [layer_fn(**self.kwargs) for _ in range(self.dec_layers)]
        self.dec_norm = nn.LayerNorm()

        self.dec_img_wpe = fixed_sincos2d_embeddings(*self.num_patches, self.dec_dim)
        self.dec_text_wpe = fixed_sincos1d_embeddings(self.max_len, self.dec_dim)
    
    def __call__(self, x: Array, det: bool = True) -> Array:
        # Add positional embeddings
        cls_token, x = x[:, :1, :], x[:, 1:, :]
        images_x = x[:, :self.max_patch_num, :]
        texts_x = x[:, self.max_patch_num:, :]
        images_x = images_x + self.dec_img_wpe.reshape(-1, self.dec_dim)
        texts_x = texts_x + self.dec_text_wpe

        # Decoder blocks
        x = jnp.concatenate((cls_token, images_x, texts_x), axis=1)
        for layer in self.dec_layer:
            x = layer(x, det)
        x = self.dec_norm(x)
        return x