import pandas as pd
from collections import defaultdict
import argparse


def read_data(path):
    data = pd.read_json(path, lines=True, chunksize=100000)
    examples_by_authors = defaultdict(list)

    for chunk in data:
        chunk = chunk.to_dict('records')
        for example in chunk:
            author, text = example['reviewerID'], example['reviewText']
            examples_by_authors[author].append(text)

    # keep authors who wrote at least 2800 review

    subset = dict()
    for author in examples_by_authors:
        docs = examples_by_authors[author]
        if len(docs) >= 2800:
            subset[author] = docs

    # flatten data
    examples = []
    for author in subset:
        examples.extend(subset[author])
    return subset


def write_data(path, examples):
    with open(path, mode='w') as f:
        for example in examples:
            f.write(example)
            f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', help='Path to all amazon review')
    parser.add_argument('--output_file', help='Output file to write data.')
    args = parser.parse_args()

    data = read_data(args.input_file)

    write_data(data_dir=args.output_file, data=data)

