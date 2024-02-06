from utils.discretize import annotate_data_author, discretize_attributes
import argparse
import json
from glob import glob
import os


def load_bins(path):
    with open(path) as f:
        return json.load(f)['author_atts_bins']

def read_data(path):
    with open(path) as f:
        return [json.loads(x) for x in f.readlines()]

def load_rst_parses(rst_dir):
    files_interm = glob(os.path.join(rst_dir, '*'))
    files = sorted([f for f in files_interm if 'tree' not in f and 'json' not in f])
    parses = read_data(os.path.join(rst_dir, 'preds_rst_parsed.json'))
    pairs = pair_data_with_files(files, parses)
    return [x[1] for x in pairs]

def pair_data_with_files(files, data):
    assert len(files) == len(data)
    pairs = []
    for file, example in zip(files, data):
        pairs.append((int(file.split('/')[-1].split('.')[0]), example))
    sorted_pairs = sorted(pairs, key=lambda x: x[0])
    return sorted_pairs

def write_data(data, path):
    with open(path, mode='w', encoding='utf8') as f:
        for example in data:
            f.write(json.dumps(example))
            f.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bins_path', help='Bins file.')
    parser.add_argument('--data_path', help='Data that needs to be annotated.')
    parser.add_argument('--rst_parse', help='RST parses.')
    parser.add_argument('--output_path', help='Output file.')

    args = parser.parse_args()

    bins = load_bins(args.bins_path)
    data = read_data(args.data_path)

    if args.rst_parse:
        rst_parses = load_rst_parses(args.rst_parse)
    else:
        rst_parses = None

    annotated_outputs = annotate_data_author(data=data, bins=bins, rst_parses=rst_parses)
    discretized_outputs = discretize_attributes(data=annotated_outputs, bins=bins, mode='author_atts')
    write_data(discretized_outputs, args.output_path)



