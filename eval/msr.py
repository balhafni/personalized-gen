import numpy as np
import argparse
import json
from collections import defaultdict
from copy import deepcopy


def msr_hard(pred_atts, gold_atts, bins):
    assert len(pred_atts) == len(gold_atts)

    # we want to evaluate against attributes we annotated
    c_gold_atts = deepcopy(gold_atts)

    for ex in c_gold_atts:
        for att in gold_atts[0].keys():
            if att not in pred_atts[0].keys():
                del ex[att]


    assert pred_atts[0].keys() == c_gold_atts[0].keys()

    evals = {key: {'c': 0, 'i': 0} for key in c_gold_atts[0].keys()}
    accuracy = {}
    discounted = {}

    for i in range(len(pred_atts)):
        pred = pred_atts[i]
        gold = c_gold_atts[i]

        assert len(pred) == len(gold)


        for key in gold:
            if gold[key] == pred[key]:
                evals[key]['c'] +=1
            else:
                evals[key]['i'] +=1

    for key in evals:
        accuracy[key] = evals[key]['c'] / (evals[key]['c'] + evals[key]['i'])

    all_accs = []

    for key in evals:
        if key != 'domain':
            num_bins = len(bins[key]) - 1
            acc = float(f'{accuracy[key]:4f}')
            all_accs.append(acc)
            if num_bins != 1:
                discounted[key] = 100 * ((acc - (1 / num_bins)) / (1 / num_bins))


    return {'att_accuracy': accuracy, 'mean_accuracy': np.array([v for v in all_accs]).mean(),
            'discounted': discounted, 'discounted_median': np.median([v for k, v in discounted.items()])}

def load_data(path, mode='json'):
    if mode == 'json':
        with open(path) as f:
            return [json.loads(l) for l in f.readlines()]
    else:
        with open(path) as f:
            return [l.strip() for l in f.readlines()]

def msr_soft(pred_atts, gold_atts):
    gold_atts_vocab = defaultdict(set)
    for example in gold_atts:
        for key in example:
            gold_atts_vocab[key].add(example[key])

    gold_atts_vocab = {k: list(v) for k, v in gold_atts_vocab.items()}
    sorted_gold_atts_vocab = {}

    # sorting the bins
    for att, vals in gold_atts_vocab.items():
        if att != 'domain':
            first = [x for x in vals if '<=' in x or '<' in x]
            last = [x for x in vals if '>=' in x or '>' in x]
            others = [x for x in vals if '>=' not in x and '<=' not in x and '>' not in x and '<' not in x]
            sorted_o = sorted(others, key=lambda x: int(x.split('-')[0]))
            sorted_gold_atts_vocab[att] = first + sorted_o + last
        else:
            sorted_gold_atts_vocab[att] = vals

    assert len(pred_atts) == len(gold_atts)
    assert pred_atts[0].keys() == gold_atts[0].keys()

    evals = {key: {'c': 0, 'i': 0} for key in gold_atts[0].keys()}
    accuracy = {}

    for i in range(len(pred_atts)):
        pred = pred_atts[i]
        gold = gold_atts[i]

        assert len(pred) == len(gold)

        for key in gold:
            if is_match(gold[key], pred[key], sorted_gold_atts_vocab[key]):
                evals[key]['c'] +=1
            else:
                evals[key]['i'] +=1
        
    for key in evals:
        accuracy[key] = evals[key]['c'] / (evals[key]['c'] + evals[key]['i'])

    return {'att_accuracy': accuracy, 'mean_accuracy': np.array([v for k, v in accuracy.items() if k !='domain']).mean()}


def is_match(gold_att, pred_att, att_list):
    if gold_att == pred_att:
        return True

    if len(att_list) == 1:
        return att_list[0] == pred_att

    idx = att_list.index(gold_att)

    if idx == len(att_list) - 1:
        return att_list[-2] == pred_att

    elif idx == 0:
        return att_list[1] == pred_att

    else:
        return pred_att == att_list[idx - 1] or pred_att == att_list[idx + 1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode')
    parser.add_argument('--bins')
    parser.add_argument('--test_file')
    parser.add_argument('--prediction_file')
    parser.add_argument('--output_path')
    args = parser.parse_args()

    bins = load_data(args.bins)[0]['author_atts_bins']
    gold = load_data(args.test_file)
    preds = load_data(args.prediction_file)

    pred_atts = [doc['author_atts'] for doc in preds]
    gold_atts = [doc['author_atts'] for doc in gold]

    if args.mode == 'soft':
        msr_metrics = msr_soft(pred_atts=pred_atts,
                               gold_atts=gold_atts)
    elif args.mode == 'hard':
        msr_metrics = msr_hard(pred_atts=pred_atts,
                              gold_atts=gold_atts, bins=bins)

    print(msr_metrics)

    with open(args.output_path, mode='w') as f:
        for k, v in msr_metrics['att_accuracy'].items():
            if k != 'domain':
                f.write(f'{k}\t{len(bins[k]) - 1}\t{v*100:.2f}%\t{msr_metrics["discounted"][k]}')
                f.write('\n')
        f.write(f'\t\t{msr_metrics["mean_accuracy"]*100:.2f}\t{msr_metrics["discounted_median"]}\n')
