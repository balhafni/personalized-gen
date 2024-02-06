model_dir=/home/bashar.alhafni/personalized-gen-release/pythia_1b_prefix
data_dir=/home/bashar.alhafni/personalized-gen-release/data/data


for checkpoint in $model_dir/checkpoint-*
do
    printf "Evaluating ${checkpoint}\n"

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --multi_gpu inference.py \
        --model_dir ${checkpoint} \
        --test_file ${data_dir}/dev.json \
        --mode prepend \
        --output_path ${checkpoint}/preds.json

    python eval/annotate.py \
        --bins_path ${data_dir}/bins.json \
        --data_path ${checkpoint}/preds.json \
        --output_path ${checkpoint}/preds_annot.json

    python eval/msr.py \
        --mode hard \
        --bins ${data_dir}/bins.json \
        --test_file ${data_dir}/dev.json \
        --prediction_file ${checkpoint}/preds_annot.json \
        --output_path  ${checkpoint}/msr.hard


done
