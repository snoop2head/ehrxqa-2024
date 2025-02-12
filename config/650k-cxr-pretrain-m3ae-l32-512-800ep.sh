python3 src/main_pretrain.py \
    --mode pretrain \
    --image_mask_ratio 0.75 \
    --max_len 420 \
    --text_mask_ratio 0.75 \
    --image_loss_weight 1.0 \
    --text_loss_weight 0.5 \
    --tokenizer_name "dmis-lab/biobert-large-cased-v1.1" \
    --pretrained_mlm "dmis-lab/biobert-large-cased-v1.1" \
    --output-dir "$GCS_DATASET_DIR/CKPT" \
    --train-dataset-shards "$GCS_DATASET_DIR/mimic-cxr-wds/{000..543}.tar" \
    --train-batch-size 1024 \
    --train-loader-workers 40 \
    --random-crop rrc \
    --color-jitter 0.0 \
    --auto-augment "none" \
    --random-erasing 0.0 \
    --augment-repeats 1 \
    --test-crop-ratio 0.875 \
    --mixup 0.0 \
    --cutmix 0.0 \
    --layers 24 \
    --dim 1024 \
    --heads 16 \
    --labels 0 \
    --patch-size 32 \
    --image-size 512 \
    --posemb sincos2d \
    --pooling cls \
    --dropout 0.0 \
    --droppath 0.0 \
    --dec-layers 8 \
    --dec-dim 512 \
    --dec-heads 16 \
    --dec-posemb sincos2d \
    --dec-dropout 0.0 \
    --dec-droppath 0.0 \
    --init-seed 0 \
    --mixup-seed 0 \
    --dropout-seed 0 \
    --image-noise-seed 0 \
    --text-noise-seed 0 \
    --shuffle-seed 0 \
    --optimizer adamw \
    --learning-rate 1.5e-4 \
    --weight-decay 0.05 \
    --adam-b1 0.9 \
    --adam-b2 0.95 \
    --adam-eps 1e-8 \
    --lr-decay 1.0 \
    --clip-grad 0.0 \
    --grad-accum 1 \
    --warmup-steps $((651754 * 5 / 1024)) \
    --training-steps $((651754 * 800 / 1024)) \
    --log-interval 1000 \
    --eval-interval 0 \
    --project MIMIC-CXR \
    --name $(basename $0 .sh) \
    --ipaddr $(curl -s ifconfig.me) \
    --hostname $(hostname)
