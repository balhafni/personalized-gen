split=train
dir=../data/rst_data/${split}_shards
for s in $dir/*
do
	echo "Parsing files in $s..."
	input_files=$(for k in $s/*txt; do echo $k; done)
	rst_parse -g segmentation_model.best -p  rst_parsing_model.best  $input_files > $s/${split}_parsed.json
	# paste $dir/${split}_files ../data/rst_data/${split}_parsed.json > ../data/rst_data/${split}_parsed_files.json
done



