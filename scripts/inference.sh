model_dir=/mnt/green-efs/bashar.alhafni/pythia_1b_prefix
data_dir=/home/bashar.alhafni/personalized-gen-release/data/data


accelerate launch inference.py \
    --model_dir ${model_dir} \
    --test_file ${data_dir}/test.json \
    --mode prepend \
    --output_path ${model_dir}/test_preds.json


printf "Running the annotation on the predictions..."

# writing the preds to file for rst parsing
python eval/get_files.py \
     --prediction_file ${model_dir}/test_preds.json \
     --output_dir ${model_dir}/preds_rst_test


# running rstfinder on the predictions to get rst parses
eval "$(conda shell.bash hook)"
conda activate rstenv

cd /home/bashar.alhafni/personalized-gen/data/rstfinder

input_files=$(for k in $model_dir/preds_rst_test/*; do echo $k; done)

rst_parse -g segmentation_model.best -p  rst_parsing_model.best  $input_files > $model_dir/preds_rst_test/preds_rst_parsed.json

eval "$(conda shell.bash hook)"
conda activate base

cd /home/bashar.alhafni/personalized-gen

python eval/annotate.py \
    --bins_path ${data_dir}/bins.json \
    --data_path ${model_dir}/test_preds.json \
    --rst_parse ${model_dir}/preds_rst_test \
    --output_path ${model_dir}/test_preds_annot_w_rst.json

python eval/msr.py \
        --mode hard \
        --bins ${data_dir}/bins.json \
        --test_file ${data_dir}/test.json \
        --prediction_file ${model_dir}/test_preds_annot_w_rst.json \
        --output_path  ${model_dir}/test_msr_w_rst.hard
