import json
import os
from glob import glob
import argparse


def load_data(path):
    with open(path) as f:
        return [json.loads(l) for l in f.readlines()]


def pair_data_with_files(files, data):
    assert len(files) == len(data)
    pairs = []
    for file, example in zip(files, data):
        pairs.append((int(file.split('/')[-1].split('.')[0]), example))
    sorted_pairs = sorted(pairs, key=lambda x: x[0])
    return sorted_pairs

def intergrate_data(data, rst_data, output_path):

    for i, example in enumerate(data):
        rst_parse = rst_data[i]
        rst_tree = rst_parse['scored_rst_trees'][0]['tree']
        example['rst_tree'] = rst_tree

    with open(output_path, mode='w') as f:
        for example in data:
            f.write(json.dumps(example))
            f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help='Path to annotated data', required=True)
    parser.add_argument('--rst_rels', help='Path to parsed RST data', required=True)
    parser.add_argument('--output_path', help='Output Path', required=True)

    args = parser.parse_args()

    data = load_data(args.data_path)
    rst_data = load_data(args.rst_rels)

    intergrate_data(data=data, rst_data=rst_data, output_path=args.output_path)

