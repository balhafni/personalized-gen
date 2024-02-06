import json
import os
import argparse


def read_data(path):
    with open(path) as f:
        return [json.loads(x) for x in f.readlines()]


def write_files(preds, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, pred in enumerate(preds):
        with open(os.path.join(output_dir, str(i)), mode='w') as f:
            f.write(str(pred['text']))
            f.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_file', help='Path to prediction file.')
    parser.add_argument('--output_dir', help='Directory to write files to.')
    args = parser.parse_args()

    preds = read_data(args.prediction_file)
    write_files(preds, args.output_dir)

