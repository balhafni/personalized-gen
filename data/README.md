# Data

We used the [Blogs Authorship Corpus](), [IMDb62](), [Amazon 5-core Reviews]() datasets to construct our benchmark. The description of the how the data was selected, processed, and split into train, dev, and test splits is described in our [paper](). The data we used is available in this [release]().

## Benchmark Construction:

Reproducing our benchmark can be obtained by using the `create_data.sh` script, which encapsulates the following steps:

1. Downloading each dataset separately and applying a preprocessing step if needed.
2. Annotating the data examples with fine-grained linguistic features by using the `utils/annotate_data.py` script.
3. Obtaining the RST relations by using [rstfinder](rstfinder). This can be done by 1) writing every data example to a separate file and 2) invoking the [rstfinder/parse_data.sh](rstfinder/parse_data.sh) script.
4. Spliting and descritizing the data to create train, dev, and test splits using the [utils/split_and_discretize.py](utils/split_and_discretize.py) script.