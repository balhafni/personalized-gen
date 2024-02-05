import json
import numpy as np
from collections import defaultdict
import argparse
import os
import random
from data_utils import  add_attributes
from copy import deepcopy

def load_data(path):
    with open(path) as f:
        data = [json.loads(l) for l in f.readlines()]
    return data


def write_data(data, path):
    with open(path, mode='w', encoding='utf8') as f:
        for example in data:
            example['title'] = str(example['title'])
            f.write(json.dumps(example))
            f.write('\n')

def write_bins(bins, path):
    doc_atts_to_remove = [k for k, v in bins['doc_atts_bins'].items() if len(v) - 1 == 1]
    author_att_to_remove = [k for k, v in bins['author_atts_bins'].items() if len(v) - 1 == 1]

    new_bins = deepcopy(bins)
    for att in bins['doc_atts_bins'].keys():
        if att in doc_atts_to_remove:
            del new_bins['doc_atts_bins'][att]

    for att in bins['author_atts_bins'].keys():
        if att in author_att_to_remove:
            del new_bins['author_atts_bins'][att]

    with open(path, mode='w') as f:
        json.dump(new_bins, f)

def discretize_attributes(data, doc_bins, author_bins):
    """
    Args:
        - data: list of dicts
        - doc_bins: dict of lists where the keys are the atts and the vals are the bins
        - author_bins: dict of lists where the keys are the atts and the vals are the bins

    Returns:
        - list of dicts
    """
    disc_data = []

    for example in data:
        doc_atts = example['doc_atts']
        author_atts = example['author_atts']
        disc_doc_atts = dict()
        disc_author_atts = dict()

        assert doc_atts.keys() == author_atts.keys()

        for att in doc_atts:
            if att != 'domain':
                doc_att_bins = doc_bins[att]
                author_att_bins = author_bins[att]
                # discritize each attribute value based on the attriubte bins
                doc_binned_val = np.digitize([doc_atts[att]], bins=doc_att_bins)
                author_binned_val = np.digitize([author_atts[att]], bins=author_att_bins)

                if doc_binned_val[0] == len(doc_att_bins): # if the attribute is bigger than the largest bin
                    disc_doc_atts[att] = f'>={int(doc_att_bins[-1])}'
                    assert doc_atts[att] >= int(doc_att_bins[-1])

                elif doc_binned_val[0] == 0: # if the attribute is smaller than the smallest bin
                    disc_doc_atts[att] = f'<{int(doc_att_bins[doc_binned_val[0]])}'
                    assert doc_atts[att] < int(doc_att_bins[doc_binned_val[0]])

                else:
                    disc_doc_atts[att] = f'{int(doc_att_bins[doc_binned_val[0] - 1])}-{int(doc_att_bins[doc_binned_val[0]])}'
                    assert doc_att_bins[doc_binned_val[0] - 1] <= doc_atts[att] < doc_att_bins[doc_binned_val[0]]



                if author_binned_val[0] == len(author_att_bins): # if the attribute is bigger than the largest bin
                    disc_author_atts[att] = f'>={int(author_att_bins[-1])}'
                    assert author_atts[att] >= int(author_att_bins[-1])

                elif author_binned_val[0] == 0: # if the attribute is smaller than the smallest bin
                    disc_author_atts[att] = f'<{int(author_att_bins[author_binned_val[0]])}'
                    assert author_atts[att] < int(author_att_bins[author_binned_val[0]])

                else:
                    disc_author_atts[att] = f'{int(author_att_bins[author_binned_val[0] - 1])}-{int(author_att_bins[author_binned_val[0]])}'
                    assert author_att_bins[author_binned_val[0] - 1] <= author_atts[att] < author_att_bins[author_binned_val[0]]

            else:
                disc_doc_atts[att] = doc_atts[att]
                disc_author_atts[att] = author_atts[att]

        disc_data.append({'doc_atts': disc_doc_atts, 'author_atts': disc_author_atts, 'text': example['text'],
                          'author': example['author'], 'title': example['title']}
                        )

    return disc_data


def create_bins(data):
    """
    Args:
        - data: list of dicts
    Returns:
        - dict contating the doc atts bins and the author atts bins
    """

    doc_attributes = defaultdict(list)
    author_attributes = defaultdict(list)

    # grouping the values of each attribute
    for example in data:
        for att in example['doc_atts']:
            if att != 'domain':
                doc_attributes[att].append(int(round(example['doc_atts'][att])))
                author_attributes[att].append(int(round(example['author_atts'][att])))
            else:
                doc_attributes[att].append(example['doc_atts'][att])
                author_attributes[att].append(example['author_atts'][att])


    doc_atts_bins = defaultdict(list)
    author_atts_bins = defaultdict(list)

    # creating doc atts bins
    for att, vals in doc_attributes.items():
        if att != 'domain':
            # create bins based on deciles
            bins = list(np.unique(np.percentile(vals, np.arange(0, 100, 10))))
            if bins == [0]: # making sure we have at least two bins
                bins = [0, 1]
            doc_atts_bins[att] = bins
        else:
            doc_atts_bins[att] = sorted(list(set(vals)))

    # creating author atts bins
    for att, vals in author_attributes.items():
        if att != 'domain':
            # create bins based on deciles
            bins = list(np.unique(np.percentile(vals, np.arange(0, 100, 10))))
            if bins == [0]: # making sure we have at least two bins
                bins = [0, 1]
            author_atts_bins[att] = bins
        else:
            author_atts_bins[att] = sorted(list(set(vals)))

    return {'doc_atts_bins': doc_atts_bins, 'author_atts_bins': author_atts_bins}


def build_vocab(dataset, atts='doc_atts'):
    docs_atts = [ex[atts] for ex in dataset]
    atts_vals = defaultdict(set)

    for doc_atts in docs_atts:
        for att in doc_atts:
            atts_vals[att].add(doc_atts[att])

    vocab = dict()
    for att, vals in atts_vals.items():
        vocab[att] = {k: i for i, k in enumerate(list(vals))}

    return vocab


def verify(train, dev, test):
    train_vocab_doc = build_vocab(train)
    dev_vocab_doc = build_vocab(dev)
    test_vocab_doc = build_vocab(test)

    # sanity check that all the dev and test discritized value are present in train
    assert train_vocab_doc.keys() == dev_vocab_doc.keys() == test_vocab_doc.keys()

    for attribute in dev_vocab_doc.keys():
        for k in dev_vocab_doc[attribute].keys():
            if not k in train_vocab_doc[attribute].keys():
                print(f'{k} for {attribute} in train att doc not in dev')

    for attribute in test_vocab_doc.keys():
        for k in dev_vocab_doc[attribute].keys():
            if not k in train_vocab_doc[attribute].keys():
                print(f'{k} for {attribute} in train att doc not in test')

    train_vocab_author = build_vocab(train, atts='author_atts')
    dev_vocab_author  = build_vocab(dev, atts='author_atts')
    test_vocab_author  = build_vocab(test, atts='author_atts')

    # sanity check that all the dev and test discritized value are present in train
    assert train_vocab_author.keys() == dev_vocab_author.keys() == test_vocab_author.keys()

    for attribute in dev_vocab_author.keys():
        for k in dev_vocab_author[attribute].keys():
            if not k in train_vocab_author[attribute].keys():
                print(f'{k} for {attribute} in train att author not in dev')


    for attribute in test_vocab_author.keys():
        for k in test_vocab_author[attribute].keys():
            if not k in train_vocab_author[attribute].keys():
                print(f'{k} for {attribute} in train att author not in test')


def split_data(data):
    """
    Args:
        - data: dict where the key is the author and values are the docs with attributes

    Returns:
        - splits: dict containing train, dev, and test splits
    """
    # we will split the data using the same proportions for each author:
    # 80% train, 10% dev, 10% test
    train = defaultdict(list)
    dev = defaultdict(list)
    test = defaultdict(list)

    random.seed(42)

    for author in data:
        # shuffle the data
        shuffled_data = random.sample(data[author], len(data[author]))
        assert len(shuffled_data) == len(data[author])

        train_size = int(0.8 * len(shuffled_data))
        dev_size = int(0.1 * len(shuffled_data))

        train[author] = shuffled_data[:train_size]
        dev[author] = shuffled_data[train_size: train_size+dev_size]
        test[author] = shuffled_data[train_size+dev_size: ]


    # collating data for all authors
    train_all = []
    dev_all = []
    test_all = []

    for author in train:
        train_all.extend(train[author])

    for author in dev:
        dev_all.extend(dev[author])

    for author in dev:
        test_all.extend(test[author])

    splits = {'train': train_all, 'dev': dev_all, 'test': test_all}

    for split in splits:
        print(f'\t{split}: {len(splits[split])}')
        print()

    return splits


def remove_bins(data, bins):
    """
    Removes bins of size one from the data
    """

    doc_atts_to_remove = [k for k, v in bins['doc_atts_bins'].items() if len(v) - 1 == 1]
    author_att_to_remove = [k for k, v in bins['author_atts_bins'].items() if len(v) - 1 == 1]

    new_data = []

    for example in data:
        n_example = deepcopy(example)
        doc_atts = n_example['doc_atts']
        author_atts = n_example['author_atts']
        text = n_example['text']

        for att in example['doc_atts'].keys():
            if att in doc_atts_to_remove:
                del doc_atts[att]

        for att in example['author_atts'].keys():
            if att in author_att_to_remove:
                del author_atts[att]

        new_data.append({'doc_atts': doc_atts, 'author_atts': author_atts,
                         'text': text, 'author': n_example['author'],
                         'title': n_example['title']
                         })

    return new_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help='Path to annotated data', required=True)
    parser.add_argument('--output_dir', help='Dir to write splits too', required=True)

    args = parser.parse_args()

    annotated_data = load_data(args.data_path)

    print('Adding attributes to data')
    data_w_attributes = add_attributes(annotated_data)

    print('Creating splits..')
    splits = split_data(data_w_attributes)

    train, dev, test = splits['train'], splits['dev'], splits['test']

    write_data(data=train, path=os.path.join(args.output_dir, 'train_raw.json'))
    write_data(data=dev, path=os.path.join(args.output_dir, 'dev_raw.json'))
    write_data(data=test, path=os.path.join(args.output_dir, 'test_raw.json'))

    bins  = create_bins(train)

    print('Discretizing data\n\n')
    train_disc  = discretize_attributes(train, doc_bins=bins['doc_atts_bins'],
                                        author_bins=bins['author_atts_bins'])

    dev_disc = discretize_attributes(dev, doc_bins=bins['doc_atts_bins'],
                                     author_bins=bins['author_atts_bins'])

    test_disc = discretize_attributes(test, doc_bins=bins['doc_atts_bins'],
                                      author_bins=bins['author_atts_bins'])


    train_disc_new = remove_bins(data=train_disc, bins=bins)
    dev_disc_new = remove_bins(data=dev_disc, bins=bins)
    test_disc_new = remove_bins(data=test_disc, bins=bins)

    write_data(data=train_disc_new, path=os.path.join(args.output_dir, 'train.json'))
    write_data(data=dev_disc_new, path=os.path.join(args.output_dir, 'dev.json'))
    write_data(data=test_disc_new, path=os.path.join(args.output_dir, 'test.json'))

    write_bins(bins=bins, path=os.path.join(args.output_dir, 'bins.json'))

    verify(train_disc_new, dev_disc_new, test_disc_new)


    print('Done!')
