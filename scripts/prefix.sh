data_dir=/home/bashar.alhafni/personalized-gen-release/data/data

export NCCL_P2P_DISABLE=1
export NCCL_P2P_LEVEL=NVL

accelerate launch --multi_gpu run_clm.py \
    --model_name_or_path  EleutherAI/pythia-1b-deduped \
    --use_attributes \
    --add_attributes_to_tokens \
    --train_file ${data_dir}/train.json \
    --validation_file ${data_dir}/dev.json \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --remove_unused_columns False \
    --save_strategy epoch \
    --logging_strategy epoch \
    --evaluation_strategy epoch \
    --block_size 1024 \
    --do_train \
    --lr_scheduler_type reduce_lr_on_plateau \
    --num_train_epochs 10 \
    --report_to "none" \
    --output_dir pythia_1b_prefix

