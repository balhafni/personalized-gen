from .data_utils import Corpus, doc_attributes
from collections import defaultdict
import numpy as np
import re
from copy import deepcopy

def annotate_data_doc(data):
    """
    Args:
        - data: list of strings
    Returns:
        - atts: dict containing the attributes
    """
    annotated_data = []

    for text in data:
        annotated_example = Corpus(author=None, title=None, domain=None, text=text)
        attributes = doc_attributes(annotated_example.to_dict())
        annotated_data.append({'doc_atts': attributes, 'text': text})

    print('Done!')
    return annotated_data


def annotate_data_author(data, bins, rst_parses):
    """
    Args:
        - data: list of strings
    Returns:
        - atts: dict containing the attributes
    """
    annotated_data = []

    data_by_authors = defaultdict(list)

    if rst_parses == None:
        rst_parses = [None for _ in range(len(data))]

    assert len(data) == len(rst_parses)

    for pred_example, rst_parse in zip(data, rst_parses):

        annotated_example = Corpus(author=pred_example['author'], title=pred_example['title'],
                                   domain=pred_example['domain'],
                                   text=str(pred_example['text']).strip())
        annotated_example = annotated_example.to_dict()

        if rst_parse:
            # add the rst parse of every example
            rst_tree = rst_parse['scored_rst_trees'][0]['tree']
        else:
            rst_tree = None

        annotated_example['rst_tree'] = rst_tree

        attributes = doc_attributes(annotated_example)

        # taking out the attributes that are not in bins
        c_attributes = deepcopy(attributes)

        for att in attributes:
            if att not in bins:
                del c_attributes[att]

        annotated_data.append({'doc_atts': c_attributes, 'text': str(pred_example['text']).strip(),
                               'author': pred_example['author'], 'title': pred_example['title']}
                             )

    # grouping the annotated data by author
    for example in annotated_data:
        data_by_authors[example['author']].append(example)

    att_keys = annotated_data[0]['doc_atts'].keys()

    # averaging the doc attributes to get author attributes
    for example in annotated_data:
        author_examples = data_by_authors[example['author']]
        example['author_atts'] = {k: sum([x['doc_atts'][k] for x in author_examples]) / len(author_examples) for k in att_keys if k != 'domain'}
        example['author_atts']['domain'] = author_examples[0]['doc_atts']['domain']

    print('Done!')
    return annotated_data


def discretize_attributes(data, bins, mode='doc_atts'):
    """
    Args:
        - data: list of dicts
        - bins: dict of lists where the keys are the atts and the vals are the bins
    Returns:
        - list of dicts
    """
    disc_data = []

    for example in data:
        atts = example[mode]
        disc_atts = dict()

        for att in atts:
            if att != 'domain' and att in bins:
                att_bins = bins[att]
                # discritize each attribute value based on the attriubte bins
                att_binned_val = np.digitize([atts[att]], bins=att_bins)

                if att_binned_val[0] == len(att_bins): # if the attribute is bigger than the largest bin
                    disc_atts[att] = f'>={int(att_bins[-1])}'
                    assert atts[att] >= int(att_bins[-1])

                elif att_binned_val[0] == 0: # if the attribute is smaller than the smallest bin
                    disc_atts[att] = f'<{int(att_bins[att_binned_val[0]])}'
                    assert atts[att] < int(att_bins[att_binned_val[0]])

                else:
                    disc_atts[att] = f'{int(att_bins[att_binned_val[0] - 1])}-{int(att_bins[att_binned_val[0]])}'
                    assert att_bins[att_binned_val[0] - 1] <= atts[att] < att_bins[att_binned_val[0]]

            else:
                disc_atts[att] = atts[att]

        disc_data.append({mode: disc_atts, 'text': example['text']})

    return disc_data