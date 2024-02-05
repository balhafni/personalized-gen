# dir=/mnt/green-efs/bashar.alhafni/data_check/scaling_exps/train_35k_rst/30k

# input_files=$(for k in $dir/*; do echo $k; done)

# rst_parse -g segmentation_model.best -p  rst_parsing_model.best  $input_files > $dir/rst_parsed.json


dir=/mnt/green-efs/bashar.alhafni/data_check/author_scaling_exps_new/1000/dev_rst/60k

input_files=$(for k in $dir/*; do echo $k; done)

rst_parse -g segmentation_model.best -p  rst_parsing_model.best  $input_files > $dir/rst_parsed.json
