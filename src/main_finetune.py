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
import random
import warnings

import jax
import numpy as np
import tqdm
import wandb
from flax.jax_utils import unreplicate
from flax.serialization import msgpack_serialize
from flax.training.common_utils import shard
from torch.utils.data import DataLoader

from dataset import create_dataloaders
from training import TrainState, create_train_state, training_step, validation_step
from utils import AverageMeter, save_checkpoint_in_background

warnings.filterwarnings("ignore")


def evaluate(state: TrainState, dataloader: DataLoader, test=False) -> dict[str, float]:
    average_meter = AverageMeter()
    for batch in tqdm.tqdm(dataloader, leave=False, dynamic_ncols=True):
        metrics = validation_step(state, shard(jax.tree_map(np.asarray, batch)))
        average_meter.update(**jax.device_get(unreplicate(metrics)))
    
    subscript = "test/" if test else "val/"
    metrics = average_meter.summary(subscript)
    return metrics


def main(args: argparse.Namespace):
    train_dataloader, valid_dataloader, test_dataloader = create_dataloaders(args)
    train_dataloader_iter = iter(train_dataloader)
    state = create_train_state(args).replicate()

    # SANITATION CHECK
    print("EVALUATION")
    metrics = evaluate(state, valid_dataloader) if valid_dataloader is not None else {}
    test_metrics = evaluate(state, test_dataloader, test=True) if test_dataloader is not None else {}

    if jax.process_index() == 0:
        wandb.init(name=args.name, project=args.project, config=args)
    average_meter, max_val_acc1, max_val_RS10 = AverageMeter(use_latest=["learning_rate"]), 0.0, 0.0
    max_test_acc1, max_test_RS10 = 0.0, 0.0
    min_loss, min_test_loss = 1e9, 1e9

    print("TRAINING")
    for step in tqdm.trange(1, args.training_steps + 1, dynamic_ncols=True):
        for _ in range(args.grad_accum):
            batch = shard(jax.tree_map(np.asarray, next(train_dataloader_iter)))
            state, metrics = training_step(state, batch)
            average_meter.update(**unreplicate(metrics))

        if (
            jax.process_index() == 0
            and args.log_interval > 0
            and step % args.log_interval == 0
        ):
            metrics = average_meter.summary(prefix="train/")
            metrics["processed_samples"] = step * args.train_batch_size
            wandb.log(metrics, step)

        if args.eval_interval > 0 and (
            step % args.eval_interval == 0 or step == args.training_steps
        ):
            if jax.process_index() == 0:
                params_bytes = msgpack_serialize(unreplicate(state.params))
                save_checkpoint_in_background(args, params_bytes, postfix="last")
            
            if jax.process_index() == 0:
                if valid_dataloader is not None:
                    metrics = evaluate(state, valid_dataloader)
        
                    if metrics["val/acc1"] > max_val_acc1:
                        max_val_acc1 = metrics["val/acc1"]
                        save_checkpoint_in_background(args, params_bytes, postfix="best")
                        print(f"Best acc1: {max_val_acc1}")
                        
                    if metrics["val/RS10"] > max_val_RS10:
                        max_val_RS10 = metrics["val/RS10"]
                        save_checkpoint_in_background(args, params_bytes, postfix="best_RS10")
                        print(f"Best RS10: {max_val_RS10}")
                    
                    if metrics["val/YN_DIFF_MEAN"] < min_loss:
                        min_loss = metrics["val/YN_DIFF_MEAN"]
                        save_checkpoint_in_background(args, params_bytes, postfix="best_DIFF")
                        print(f"Best loss: {min_loss}")
                
                    metrics["val/acc1/best"] = max_val_acc1
                    metrics["val/RS10/best"] = max_val_RS10
                    metrics["processed_samples"] = step * args.train_batch_size
                    wandb.log(metrics, step)

                if test_dataloader is not None:
                    test_metrics = evaluate(state, test_dataloader, test=True)

                    if jax.process_index() == 0:
                        if test_metrics["test/acc1"] > max_test_acc1:
                            max_test_acc1 = test_metrics["test/acc1"]
                            save_checkpoint_in_background(args, params_bytes, postfix="best_test") if valid_dataloader is None else None
                            print(f"Best test acc1: {max_test_acc1}")
                        
                        if test_metrics["test/RS10"] > max_test_RS10:
                            max_test_RS10 = test_metrics["test/RS10"]
                            save_checkpoint_in_background(args, params_bytes, postfix="best_test_RS10") if valid_dataloader is None else None
                            print(f"Best test RS10: {max_test_RS10}")

                        if test_metrics["test/YN_DIFF_MEAN"] < min_test_loss:
                            min_test_loss = test_metrics["test/YN_DIFF_MEAN"]
                            save_checkpoint_in_background(args, params_bytes, postfix="best_test_DIFF") if valid_dataloader is None else None
                            print(f"Best test loss: {min_test_loss}")
                        
                        test_metrics["test/acc1/best"] = max_test_acc1
                        test_metrics["test/RS10/best"] = max_test_RS10
                        wandb.log(test_metrics, step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="finetune")
    parser.add_argument("--mask_label", action="store_true", default=False)
    parser.add_argument("--pretrained-ckpt", default=None)
    parser.add_argument("--train-dataset-shards")
    parser.add_argument("--valid-dataset-shards")
    parser.add_argument("--test-dataset-shards")
    parser.add_argument("--max_len", type=int, default=420)
    parser.add_argument("--truncation_len", type=int, default=64)
    parser.add_argument("--tokenizer_name", default="bert-base-uncased")
    parser.add_argument("--train-batch-size", type=int, default=2048)
    parser.add_argument("--valid-batch-size", type=int, default=256)
    parser.add_argument("--test-batch-size", type=int, default=256)
    parser.add_argument("--train-loader-workers", type=int, default=40)
    parser.add_argument("--valid-loader-workers", type=int, default=5)
    parser.add_argument("--test-loader-workers", type=int, default=5)
    parser.add_argument("--add-choices-prompt", action="store_true", default=False)

    parser.add_argument("--random-crop", default="rrc")
    parser.add_argument("--color-jitter", type=float, default=0.0)
    parser.add_argument("--auto-augment", default="rand-m9-mstd0.5-inc1")
    parser.add_argument("--random-erasing", type=float, default=0.25)
    parser.add_argument("--augment-repeats", type=int, default=3)
    parser.add_argument("--test-crop-ratio", type=float, default=0.875)

    parser.add_argument("--mixup", type=float, default=0.8)
    parser.add_argument("--cutmix", type=float, default=1.0)
    parser.add_argument("--criterion", default="ce")
    parser.add_argument("--label-smoothing", type=float, default=0.1)

    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--labels", type=int, default=-1)
    parser.add_argument("--layerscale", action="store_true", default=False)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--posemb", default="learnable")
    parser.add_argument("--pooling", default="cls")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--droppath", type=float, default=0.1)
    parser.add_argument("--grad-ckpt", action="store_true", default=False)

    parser.add_argument("--init-seed", type=int, default=random.randint(0, 1000000))
    parser.add_argument("--mixup-seed", type=int, default=random.randint(0, 1000000))
    parser.add_argument("--dropout-seed", type=int, default=random.randint(0, 1000000))
    parser.add_argument("--image-noise-seed", type=int, default=random.randint(0, 1000000))
    parser.add_argument("--text-noise-seed", type=int, default=random.randint(0, 1000000))
    parser.add_argument("--shuffle-seed", type=int, default=random.randint(0, 1000000))
    parser.add_argument("--noise-seed", type=int, default=random.randint(0, 1000000))
    parser.add_argument("--label-mapping")
    parser.add_argument("--test-threshold", type=float, default=0.5)

    parser.add_argument("--optimizer", default="adamw")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--adam-b1", type=float, default=0.9)
    parser.add_argument("--adam-b2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--lr-decay", type=float, default=1.0)
    parser.add_argument("--clip-grad", type=float, default=0.0)
    parser.add_argument("--grad-accum", type=int, default=1)

    parser.add_argument("--warmup-steps", type=int, default=10000)
    parser.add_argument("--training-steps", type=int, default=200000)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=0)

    parser.add_argument("--project")
    parser.add_argument("--name")
    parser.add_argument("--ipaddr")
    parser.add_argument("--hostname")
    parser.add_argument("--output-dir", default=".")
    main(parser.parse_args())
