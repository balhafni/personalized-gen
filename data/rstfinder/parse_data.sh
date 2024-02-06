dir=../data/rst_data

input_files=$(for k in $dir/*; do echo $k; done)

rst_parse -g segmentation_model.best -p  rst_parsing_model.best  $input_files > $dir/rst_parsed.json
