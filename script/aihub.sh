#!/bin/bash
#SBATCH -p 
#SBATCH --cpus-per-gpu=
#SBATCH --mem-per-gpu=
#SBATCH --time=

PYTHON_PATH=
OUTPUT_DIR=$4 
MASTER_NODE=$1
OMP_NUM_THREADS=1 python -m torch.distributed.launch \
    --nproc_per_node=$6 \
    --master_port $3 --nnodes=$5 \
    --node_rank=$2 --master_addr=${MASTER_NODE} \
    $PYTHON_PATH \
    --output_dir $OUTPUT_DIR \
    --log_dir $OUTPUT_DIR \
    --seed 777 \
    --model efficient \
    --dataset ai_hub \
    --part_type upper_body \
    --json_path /data/datasets/ai_hub_sketch_json/ \
    --reprob 0.0 \
    --mixup 0.0 \
    --mixup_switch_prob 0.0 \
    --cutmix 0.0 \
    --weight_decay 0.1 \
    --dropout 0.5 \
    --save_ckpt_freq 1 \
    --drop_path 0.1 \
    --dist_eval \
    --warmup_epochs 3 \
    --epochs 30 \
    --num_workers 12 \
    --data_path \
    --batch_size 256 \
    --blr 1e-3 \
    --nb_classes 5 \
    --cls_token \
    --max_acc 
