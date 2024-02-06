# Downloading the blogs authorship corpus
wget http://www.cs.biu.ac.il/~koppel/blogs/blogs.zip

unzip -d blogs_unziped blogs.zip
mv blogs_unziped/blogs ./
rm -r blogs_unziped
rm blogs.zip

# Processing the blogs corpus
mkdir -p ./processed_blogs

python utils/process_blogs_corpus.py \
    --input_dir blogs \
    --output_dir processed_blogs


# Download amazon-5-core
wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/All_Amazon_Review_5.json.gz
mkdir -p amazon-reviews
mv All_Amazon_Review_5.json.gz amazon-reviews
Filter Amazon
python utils/process_amazon.py \
    --input_file amazon-reviews/All_Amazon_Review_5.json.gz \
    --output_file amazon-reviews/amazon_subset.json

rm All_Amazon_Review_5.json.gz

# Donwload IMDb62
wget https://www.dropbox.com/s/np1u1hl343gd73m/imdb62.zip
mkdir -p imdb62
unzip -d imdb62 imdb62.zip
rm imdb62.zip

# Run the Annotation Process
python utils/annotate_data.py \
    --blogs_data_dir processed_blogs \
    --blogs_domains Communications-Media Technology Internet Arts Education \
    --imdb_path imdb62 \
    --amazon_path  amazon-reviews/amazon_subset.json \
    --output_path data.json


# Integrate RST relations with the annotated data
python utils/integrate_rst_relations.py \
    --data_path data/data.json.annotated \
    --rst_rels rst_data/rst_parsed.json \
    --output_path data.json.annotated.rst


# Split and discretize
mkdir -p annotated_data
python utils/split_and_discretize.py \
    --data_path data.json.annotated.rst \
    --output_dir ./
