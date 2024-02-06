from glob import glob
import os
from msr import msr_hard, load_data
import argparse


def get_best_checkpoint(model_path, bins, gold):
    checkpoints = glob(os.path.join(model_path, 'checkpoint-*/'))
    checkpoint_scores = []

    gold_atts = [doc['author_atts'] for doc in gold]

    for checkpoint in checkpoints:
        preds = load_data(os.path.join(checkpoint, 'preds_annot.json'))
        pred_atts = [doc['author_atts'] for doc in preds]

        msr_metrics = msr_hard(pred_atts=pred_atts, gold_atts=gold_atts, bins=bins)

        checkpoint_scores.append({'checkpoint': checkpoint,
                                'msr': msr_metrics}
                                )

    best_checkpoint = max(checkpoint_scores, key=lambda x: x['msr']['discounted_median'])
    print(best_checkpoint['checkpoint'])
    return best_checkpoint

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir')
    parser.add_argument('--gold_data')
    parser.add_argument('--bins')

    args = parser.parse_args()

    gold = load_data(args.gold_data)
    bins = load_data(args.bins)[0]['author_atts_bins']

    best_checkpoint = get_best_checkpoint(model_path=args.model_dir,
                                          bins=bins, gold=gold
                                          )

    print('Best checkpoint')
    print(best_checkpoint)

