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

import argparse
from functools import partial
from typing import Callable

import flax
import flax.linen as nn
import flax.linen.initializers as init

import jax
import jax.numpy as jnp
import optax
from chex import Array, ArrayTree, PRNGKey
from flax.training import train_state
from flax.training.common_utils import shard_prng_key
from jax.tree_util import tree_map_with_path

from transformers import AutoTokenizer

from dataset import CXR_DEFAULT_MEAN, CXR_DEFAULT_STD
from modeling import ViT, MAEDecoder
from utils import Mixup, get_layer_index_fn, load_pretrained_params, modified_lamb, load_pretrained_mlm_embedding
from utils_mae import (
  mask_intersection, mask_not, # masking related
  extract_patches, index_sequence, # patch related
  patch_mse_loss, cross_entropy_loss_and_accuracy
)

Dense = partial(nn.Dense, kernel_init=init.truncated_normal(0.02))

CRITERION_COLLECTION = {
    "ce": optax.softmax_cross_entropy,
    "bce": lambda x, y: optax.sigmoid_binary_cross_entropy(x, y > 0).mean(-1),
    "focal": lambda x, y: optax.sigmoid_focal_loss(x, y > 0).mean(-1),
}
OPTIMIZER_COLLECTION = {
    "adamw": optax.adamw,
    "lamb": modified_lamb,
    "lars": optax.lars,
    "sgd": optax.sgd,
}


class TrainState(train_state.TrainState):
    mixup_rng: PRNGKey
    dropout_rng: PRNGKey
    image_noise_rng: PRNGKey
    text_noise_rng: PRNGKey

    micro_step: int = 0
    micro_in_mini: int = 1
    grad_accum: ArrayTree | None = None

    def split_rngs(self) -> tuple[ArrayTree, ArrayTree]:
        mixup_rng, new_mixup_rng = jax.random.split(self.mixup_rng)
        dropout_rng, new_dropout_rng = jax.random.split(self.dropout_rng)
        image_noise_rng, new_image_noise_rng = jax.random.split(self.image_noise_rng)
        text_noise_rng, new_text_noise_rng = jax.random.split(self.text_noise_rng)

        rngs = {"mixup": mixup_rng, "dropout": dropout_rng, "image_noise": image_noise_rng, "text_noise": text_noise_rng}
        updates = {"mixup_rng": new_mixup_rng, "dropout_rng": new_dropout_rng, "image_noise_rng": new_image_noise_rng, "text_noise_rng": new_text_noise_rng}
        return rngs, updates

    def replicate(self) -> TrainState:
        return flax.jax_utils.replicate(self).replace(
            mixup_rng=shard_prng_key(self.mixup_rng),
            dropout_rng=shard_prng_key(self.dropout_rng),
            image_noise_rng=shard_prng_key(self.image_noise_rng),
            text_noise_rng=shard_prng_key(self.text_noise_rng),
        )


class FinetuneModule(nn.Module):
    model: ViT
    mixup: Mixup
    label_smoothing: float = 0.0
    criterion: Callable[[Array, Array], Array] = CRITERION_COLLECTION["ce"]
    test_threshold: float = 0.5  # label is 1.0, so higher the number, the stricter the threshold
    mask_label: bool = False

    def __call__(self, images: Array, texts: Array, labels: Array, det: bool = True) -> ArrayTree:
        # Normalize the pixel values in TPU devices, instead of copying the normalized
        # float values from CPU. This may reduce both memory usage and latency.
        images = jnp.moveaxis(images, 1, 3).astype(jnp.float32) / 0xFF
        images = (images - CXR_DEFAULT_MEAN) / CXR_DEFAULT_STD

        labels = nn.one_hot(labels, self.model.labels) if labels.ndim == 1 else labels
        labels = labels.astype(jnp.float32)

        if not det:
            labels = optax.smooth_labels(labels, self.label_smoothing)
            images, labels = self.mixup(images, labels)

        # Compute logits from the model
        logits = self.model(images, texts, det=det)

        # Calculate the loss
        loss = self.criterion(logits, labels)

        if self.mask_label:
            # make one only mask with shape of [B, 110] equivalent to the size of logits
            mask = jnp.ones_like(logits)
            # mask 65th index and 109th index to zero
            mask = mask.at[:, 65].set(0.0)
            mask = mask.at[:, 109].set(0.0)
            # multiply by negative infinity to make the logits to zero
            mask = mask * -1e9
            logits = logits + mask

            diff_mask = jnp.zeros_like(logits)
            diff_mask = diff_mask.at[:, 65].set(1.0)
            diff_mask = diff_mask.at[:, 109].set(1.0)
            
            no_mask = jnp.zeros_like(logits)
            no_mask = no_mask.at[:, 65].set(1.0)
            
            yes_mask = jnp.zeros_like(logits)
            yes_mask = yes_mask.at[:, 109].set(1.0)

        # Predictions are considered positive if logits are greater than the test_threshold
        preds = jax.nn.sigmoid(logits) > self.test_threshold
        preds = jax.nn.one_hot(jnp.argmax(logits, axis=1), self.model.labels) # only one maxiumum value label is considered as positive
        diff = jnp.abs(labels - jax.nn.sigmoid(logits))
        diff = diff * diff_mask
        yes_diff = diff * yes_mask
        no_diff = diff * no_mask

        # Calculate multi-label accuracy metrics
        true_positives = jnp.sum(labels * preds, axis=1) # Higher the better
        false_positives = jnp.sum((1 - labels) * preds, axis=1) # Lower the better, very important on medical context
        false_negatives = jnp.sum(labels * (1 - preds), axis=1) # Not important

        precision = true_positives / (true_positives + false_positives + 1e-8) # HIGHER THE BETTER
        recall = true_positives / (true_positives + false_negatives + 1e-8)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

        # Calculate exact match and RS10
        correct = jnp.all(labels == preds, axis=1) # Exact match
        wrong = jnp.any(labels != preds, axis=1)
        abstained = jnp.all(preds == 0, axis=1)
        RS10 = correct - 10 * (wrong & ~abstained)

        # Return metrics
        return {
            "loss": loss,
            "precision": jnp.mean(precision),
            "recall": jnp.mean(recall),
            "f1_score": jnp.mean(f1_score),
            "acc1": jnp.mean(correct), # Exact Match
            "RS10": jnp.mean(RS10), # RS10
            "YN_DIFF_MEAN": jnp.mean(diff), # Difference between true and false
            "YN_DIFF_STD": jnp.std(diff), # Difference between true and false
            "YN_DIFF_MAX": jnp.max(diff), # Difference between true and false
            "YN_DIFF_MIN": jnp.min(diff), # Difference between true and false
            "YES_DIFF_MEAN": jnp.mean(yes_diff), # Difference between true and false
            "YES_DIFF_STD": jnp.std(yes_diff), # Difference between true and false
            "YES_DIFF_MAX": jnp.max(yes_diff), # Difference between true and false
            "YES_DIFF_MIN": jnp.min(yes_diff), # Difference between true and false
            "NO_DIFF_MEAN": jnp.mean(no_diff), # Difference between true and false
            "NO_DIFF_STD": jnp.std(no_diff), # Difference between true and false
            "NO_DIFF_MAX": jnp.max(no_diff), # Difference between true and false
            "NO_DIFF_MIN": jnp.min(no_diff), # Difference between true and false
        }


class PretrainModule(nn.Module):
    """ 
    Role: Receives images and texts input arrays 
    - Encoder: ViT
    - Projection & Reassembling with Mask Tokens
    - Decoder: MAEDecoder
    """
    model: ViT
    decoder_model: MAEDecoder
    image_size: int
    image_loss_weight: float = 1.0
    text_loss_weight: float = 0.5
    norm_pix_loss: bool = False
    pad_token_id: int = 0

    def setup(self):
        self.image_mask_embedding = self.param("image_mask_embedding", init.truncated_normal(0.02), (1, 1, self.decoder_model.dec_dim))
        self.text_mask_embedding = self.param("text_mask_embedding", init.truncated_normal(0.02), (1, 1, self.decoder_model.dec_dim))

        self.decoder_image_type_embedding = self.param("decoder_image_type_embedding", init.truncated_normal(0.02), (1, 1, self.decoder_model.dec_dim))
        self.decoder_text_type_embedding = self.param("decoder_text_type_embedding", init.truncated_normal(0.02), (1, 1, self.decoder_model.dec_dim))

        self.decoder_proj = Dense(self.decoder_model.dec_dim)
        
        self.decoder_image_output = Dense(self.model.patch_size ** 2 * 1)
        self.decoder_text_output = Dense(self.model.text_vocab_size)

    def __call__(self, images: Array, texts: Array, det: bool = True) -> ArrayTree:
        
        # Normalize the pixel values in TPU devices, instead of copying the normalized
        # float values from CPU. This may reduce both memory usage and latency.
        images = jnp.moveaxis(images, 1, 3).astype(jnp.float32) / 0xFF
        images = (images - CXR_DEFAULT_MEAN) / CXR_DEFAULT_STD

        flag_validation = True if texts is None else False
        if texts is None:
            # fill with self.pad_token_id with (bs, max_len) shape array
            texts = jnp.full((images.shape[0], self.model.max_len), self.pad_token_id, dtype=jnp.int32)
        text_padding_mask = texts == self.pad_token_id

        # ENCODER: ViT Model feedforward (No deterministic mode due to random masking)
        x, image_mask, text_mask, image_ids_restore, text_ids_restore = self.model(images, texts, validation=flag_validation, det=det)
        
        # Linear projection before reassembling the input representation
        x = self.decoder_proj(x)

        # split kept vectors before reassmbling
        cls_token, x = x[:, :1, :], x[:, 1:, :]
        image_x = x[:, :self.model.image_keep_length, :]
        text_x = x[:, self.model.image_keep_length:, :]
        
        # repeat masked vectors before reassembling
        bs, _, _ = cls_token.shape
        masked_image_x = jnp.broadcast_to(self.image_mask_embedding, (bs, self.model.masked_patch_num, self.decoder_model.dec_dim))
        masked_text_x = jnp.broadcast_to(self.text_mask_embedding, (bs, self.model.masked_len, self.decoder_model.dec_dim))

        # Reassemble kept vectors and masked vectors back to original order referring image_ids_restore
        image_x = index_sequence(jnp.concatenate([image_x, masked_image_x], axis=1), image_ids_restore)
        text_x = index_sequence(jnp.concatenate([text_x, masked_text_x], axis=1), text_ids_restore)

        # Add type embeddings
        image_x += self.decoder_image_type_embedding
        text_x += self.decoder_text_type_embedding

        # DECODER: MAEDecoder Model feedforward
        x = jnp.concatenate([cls_token, image_x, text_x], axis=1)
        x = self.decoder_model(x, det=det)

        # split concatenated[kept+masked] vectors
        cls_token, x = x[:, :1, :], x[:, 1:, :]
        image_x = x[:, :self.model.max_patch_num, :]
        text_x = x[:, self.model.max_patch_num:, :]

        # Make logits
        image_output = self.decoder_image_output(image_x)
        text_output = self.decoder_text_output(text_x)

        # Make label
        image_patches = extract_patches(images, self.model.patch_size)
        text_mask = mask_intersection(text_mask, mask_not(text_padding_mask))

        # Get loss
        if self.norm_pix_loss:
            mean = jnp.mean(image_patches, axis=-1, keepdims=True)
            var = jnp.var(image_patches, axis=-1, keepdims=True)
            image_patches = (image_patches - mean) / jnp.sqrt(var + 1e-6)

        image_loss = patch_mse_loss(image_output, image_patches, image_mask)
        text_loss, text_accuracy = cross_entropy_loss_and_accuracy(text_output, texts, text_mask)
        loss = self.image_loss_weight * image_loss + self.text_loss_weight * text_loss

        return {"loss": loss, "image_loss": image_loss, "text_loss": text_loss, "text_accuracy": text_accuracy}
        

@partial(jax.pmap, axis_name="batch", donate_argnums=0)
def training_step(state: TrainState, batch: ArrayTree) -> tuple[TrainState, ArrayTree]:
    def loss_fn(params: ArrayTree) -> ArrayTree:
        metrics = state.apply_fn({"params": params}, *batch, det=False, rngs=rngs)
        metrics = jax.tree_map(jnp.mean, metrics)
        return metrics["loss"], metrics

    def update_fn(state: TrainState) -> TrainState:
        # Collect a global gradient from the accumulated gradients and apply actual
        # parameter update with resetting the accumulations to zero.
        grads = jax.tree_map(lambda g: g / state.micro_in_mini, state.grad_accum)
        return state.apply_gradients(
            grads=jax.lax.pmean(grads, axis_name="batch"),
            grad_accum=jax.tree_map(jnp.zeros_like, state.grad_accum),
            micro_step=state.micro_step % state.micro_in_mini,
        )

    rngs, updates = state.split_rngs()
    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    metrics = jax.lax.pmean(metrics, axis_name="batch")

    # Update parameters with the gradients. If the gradient accumulation is enabled,
    # then the parameters will be updated at the end of each mini-batch step. In every
    # micro steps, the gradients will be accumulated.
    if state.grad_accum is None:
        state = state.apply_gradients(grads=jax.lax.pmean(grads, axis_name="batch"))
    else:
        state = state.replace(
            grad_accum=jax.tree_map(lambda ga, g: ga + g, state.grad_accum, grads),
            micro_step=state.micro_step + 1,
        )
        state = jax.lax.cond(
            state.micro_step == state.micro_in_mini, update_fn, lambda x: x, state
        )
    return state.replace(**updates), metrics | state.opt_state.hyperparams


@partial(jax.pmap, axis_name="batch")
def validation_step(state: TrainState, batch: ArrayTree) -> ArrayTree:
    if len(batch) < 3: # pretraining
        rngs, _ = state.split_rngs()
        metrics = state.apply_fn({"params": state.params}, *batch, det=False, rngs=rngs)
    else: # finetuning
        metrics = state.apply_fn({"params": state.params}, *batch, det=True)
    return jax.lax.pmean(metrics, axis_name="batch")


def create_train_state(args: argparse.Namespace) -> TrainState:

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    model = ViT(
        layers=args.layers,
        dim=args.dim,
        heads=args.heads,
        labels=args.labels,
        layerscale=args.layerscale,
        patch_size=args.patch_size,
        image_size=args.image_size,
        posemb=args.posemb,
        pooling=args.pooling,
        dropout=args.dropout,
        droppath=args.droppath,
        grad_ckpt=args.grad_ckpt,
        max_len=args.max_len,
        truncation_len=args.truncation_len if args.mode == "finetune" else args.max_len,
        text_vocab_size=tokenizer.vocab_size,
        text_mask_ratio=args.text_mask_ratio if args.mode == "pretrain" else None,
        image_mask_ratio=args.image_mask_ratio if args.mode == "pretrain" else None,
        linear_probing=True if args.mode == "linear" else False,
    )

    if args.mode == "pretrain":
        decoder_model = MAEDecoder(
            dec_layers=args.dec_layers,
            dec_dim=args.dec_dim,
            dec_heads=args.dec_heads,
            dec_layerscale=args.dec_layerscale,
            dec_posemb=args.dec_posemb,
            dec_dropout=args.dec_dropout,
            dec_droppath=args.dec_droppath,
            grad_ckpt=args.grad_ckpt,
            patch_size=args.patch_size,
            image_size=args.image_size,
            max_len=args.max_len,
        )
        module = PretrainModule(
            model=model,
            decoder_model=decoder_model,
            image_size=args.image_size,
            norm_pix_loss=args.norm_pix_loss,
            image_loss_weight=args.image_loss_weight,
            text_loss_weight=args.text_loss_weight,
            pad_token_id=tokenizer.pad_token_id,
        )
        example_inputs = {
            "images": jnp.zeros((1, 1, args.image_size, args.image_size), dtype=jnp.uint8),
            "texts": jnp.zeros((1, args.max_len), dtype=jnp.int32),
        }
    elif args.mode == "finetune" or args.mode == "linear":
        module = FinetuneModule( 
            model=model,
            mixup=Mixup(args.mixup, args.cutmix),
            label_smoothing=args.label_smoothing if args.criterion == "ce" else 0,
            criterion=CRITERION_COLLECTION[args.criterion],
            test_threshold=args.test_threshold,
            mask_label=args.mask_label,
        )

        # Initialize the model weights with dummy inputs. Using the init RNGS and inputs, we
        # will tabulate the summary of model and its parameters. Furthermore, empty gradient
        # accumulation arrays will be prepared if the gradient accumulation is enabled.
        example_inputs = {
            "images": jnp.zeros((1, 1, args.image_size, args.image_size), dtype=jnp.uint8),
            "texts": jnp.zeros((1, args.truncation_len), dtype=jnp.int32),
            "labels": jnp.zeros((1, args.labels), dtype=jnp.int32),
        }
    init_rngs = {"params": jax.random.PRNGKey(args.init_seed)}
    print(module.tabulate(init_rngs, **example_inputs)) if "debug" not in args.name else None

    params = module.init(init_rngs, **example_inputs)["params"]
    if args.pretrained_ckpt is not None:
        params = load_pretrained_params(args, params, finetune=True if args.mode == "finetune" else False)
    elif args.pretrained_ckpt is None and args.mode == "pretrain":
        if args.pretrained_mlm is not None:
            params = load_pretrained_mlm_embedding(args, params)
    if args.grad_accum > 1:
        grad_accum = jax.tree_map(jnp.zeros_like, params)


    # Create learning rate scheduler and optimizer with gradient clipping. The learning
    # rate will be recorded at `hyperparams` by `optax.inject_hyperparameters`.
    @partial(optax.inject_hyperparams, hyperparam_dtype=jnp.float32)
    def create_optimizer_fn(
        learning_rate: optax.Schedule,
    ) -> optax.GradientTransformation:
        if args.optimizer == "adamw" or args.optimizer == "lamb":
            tx = OPTIMIZER_COLLECTION[args.optimizer](
                learning_rate=learning_rate,
                b1=args.adam_b1,
                b2=args.adam_b2,
                eps=args.adam_eps,
                weight_decay=args.weight_decay,
                mask=partial(tree_map_with_path, lambda kp, *_: kp[-1].key == "kernel"),
            )
        elif args.optimizer == "lars" or args.optimizer == "sgd":
            tx = OPTIMIZER_COLLECTION[args.optimizer](
                learning_rate=learning_rate,
                momentum=0.9,
            )
        if args.lr_decay < 1.0:
            layerwise_scales = {
                i: optax.scale(args.lr_decay ** (args.layers - i))
                for i in range(args.layers + 1)
            }
            label_fn = partial(get_layer_index_fn, num_layers=args.layers)
            label_fn = partial(tree_map_with_path, label_fn)
            tx = optax.chain(tx, optax.multi_transform(layerwise_scales, label_fn))
        if args.clip_grad > 0:
            tx = optax.chain(optax.clip_by_global_norm(args.clip_grad), tx)
        return tx

    # https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/PRETRAIN.md
    # https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/main_pretrain.py#L40-L41
    # https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/main_pretrain.py#L156-L184
    lr_peak_value = args.learning_rate * args.train_batch_size / 256 if args.mode == "pretrain" else args.learning_rate
    print(f"Peak learning rate: {lr_peak_value:.1e}")
    
    learning_rate = optax.warmup_cosine_decay_schedule(
        init_value=1e-6,
        peak_value=lr_peak_value,
        warmup_steps=args.warmup_steps,
        decay_steps=args.training_steps,
        end_value=1e-5,
    )
    return TrainState.create(
        apply_fn=module.apply,
        params=params,
        tx=create_optimizer_fn(learning_rate),
        mixup_rng=jax.random.PRNGKey(args.mixup_seed + jax.process_index()),
        dropout_rng=jax.random.PRNGKey(args.dropout_seed + jax.process_index()),
        image_noise_rng=jax.random.PRNGKey(args.image_noise_seed + jax.process_index()),
        text_noise_rng=jax.random.PRNGKey(args.text_noise_seed + jax.process_index()),
        micro_step=0,
        micro_in_mini=args.grad_accum,
        grad_accum=grad_accum if args.grad_accum > 1 else None,
    )
