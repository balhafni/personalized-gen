# dir=/mnt/green-efs/bashar.alhafni/data_check/scaling_exps/train_20k_rst

# input_files=$(for k in $dir/*; do echo $k; done)

# rst_parse -g segmentation_model.best -p  rst_parsing_model.best  $input_files > $dir/rst_parsed.json


dir=/mnt/green-efs/bashar.alhafni/data_check/author_scaling_exps_new/1000/train_rst/15k

input_files=$(for k in $dir/*; do echo $k; done)

rst_parse -g segmentation_model.best -p  rst_parsing_model.best  $input_files > $dir/rst_parsed.json



# dir=/mnt/green-efs/bashar.alhafni/data_check/author_scaling_exps/100/rst/30k

# input_files=$(for k in $dir/*; do echo $k; done)

# rst_parse -g segmentation_model.best -p  rst_parsing_model.best  $input_files > $dir/rst_parsed.json



# dir=/mnt/green-efs/bashar.alhafni/data_check/author_scaling_exps/100/rst/40k

# input_files=$(for k in $dir/*; do echo $k; done)

# rst_parse -g segmentation_model.best -p  rst_parsing_model.best  $input_files > $dir/rst_parsed.json