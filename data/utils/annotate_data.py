import os
from collections import defaultdict
from data_utils import Corpus
import json
import argparse
import re


def read_data_imdb(path):
    """
    Args:
        - path (path): path to imdb txt file containing data

    Returns:
        - dict where keys are the authors and values are the textual data
    """
    data = defaultdict(list)

    with open(path) as f:

        for line in f.readlines():
            line = line.strip().split('\t')
            example_id, author_id, text = line[0], line[1], line[5]
            data[author_id].append({'file': example_id, 'text': text, 'domain': 'movie'})

    print('\nIMDB dataset statistics')
    print(f'\t# Authors: {len(data)}')
    print(f'\t# Docs: {sum(len(v) for v in data.values())}')
    return data


def read_data_blogs(data_dir, domains):
    """
    Args:
        - data_dir (path): path to dir containing the domains folder
        - domains (list): list of domains we want to select

    Returns:
        - dict where keys are the domains and values are the textual data in each domain
    """
    data = defaultdict(list)

    for domain in domains:
        path = os.path.join(data_dir, domain)
        files = os.listdir(path)
        for file in files:
            file_path = os.path.join(path, file)
            author = file.split('_')[0]

            with open(file_path, encoding='utf8') as f:
                lines = [x.strip() for x in f.readlines()]
                assert len(lines) == 1
                text = lines[0]
                data[author].append({'file': file, 'text': text, 'domain': 'blog'})

    print('\nBlogs dataset statistics')
    print(f'\t# Authors: {len(data)}')
    print(f'\t# Docs: {sum(len(v) for v in data.values())}')

    return data


def read_data_amazon(path):
    """
    Args:
        - path (path): path to json file containing data

    Returns:
        - dict where keys are the authors and values are the textual data
    """
    data = defaultdict(list)

    with open(path) as f:
        for example_id, line in enumerate(f.readlines()):
            example = json.loads(line.strip())
            author, text = example['author'], example['text']
            data[author].append({'file': str(example_id), 'text': text, 'domain': 'retail'})

    print('\nAmazon dataset statistics')
    print(f'\t# Authors: {len(data)}')
    print(f'\t# Docs: {sum(len(v) for v in data.values())}')
    return data


def load_data(path):
    with open(path) as f:
        return [json.loads(ex) for ex in f.readlines()]


def filter_data(data, doc_threshold, doc_word_threshold, author_max_word):

    total_authors = len(data)
    total_docs = sum(len(x) for x in data.values())

    filtered_data = defaultdict(list)

    for author in data:
        author_word_num = 0

        for doc in data[author]:
            try:
                text = doc['text'].strip()
            except:
                import pdb; pdb.set_trace()
                x = 10
            text = text.replace('\n', ' ')
            text = text.replace('\x00', '')
            text = re.sub(' +', ' ', text)

            words = text.split(' ')

            # we would like to keep docs with at least doc_word_threshold words

            if len(words) >= doc_word_threshold:
                author_word_num += len(words)

                # if we reach the author_max_word limit, we stop adding docs
                if author_word_num > author_max_word:
                    break

                doc['text'] = text
                filtered_data[author].append(doc)

    total_docs_fil = sum(len(x) for x in filtered_data.values())

    print(f'{total_docs - total_docs_fil} docs have less than {doc_word_threshold} words')

    # we would like to keep authors who wrote at least doc_threshold docs
    filtered_data = {author: docs for author, docs in filtered_data.items() if len(docs) >= doc_threshold}

    total_authors_fil = len(filtered_data)
    num_docs = sum(len(v) for v in filtered_data.values())

    print(f'{total_authors - total_authors_fil} authors have less than {doc_threshold} docs')
    print(f'\t# Authors: {total_authors_fil}')
    print(f'\t# Docs: {num_docs}')

    docs = [doc for v in filtered_data.values() for doc in v]
    doc_avg_words = sum([len(doc['text'].split(' ')) for doc in docs]) / len(docs)
    author_avg_words = sum([len(doc['text'].split(' ')) for doc in docs]) / total_authors_fil

    print(f'\tAvg Words per Doc: {doc_avg_words}')
    print(f'\tAvg Words per Author: {author_avg_words}')

    return filtered_data


def combine_datasets(blogs, imdb, amazon, output_path):
    data = []

    for author, docs in blogs.items():

        for doc in docs:
            doc['author'] = author

        data.extend(docs)

    for author, docs in imdb.items():
        for doc in docs:
            doc['author'] = author

        data.extend(docs)

    for author, docs in amazon.items():

        for doc in docs:
            doc['author'] = author

        data.extend(docs)

    with open(output_path, mode='w') as f:
        for doc in data:
            f.write(json.dumps(doc))
            f.write('\n')

    return data


def annotate_data(data):
    """
    Args:
        - example: list of examples (dicts)
    Returns:
        - atts: dict containing the attributes
    """
    annotated_data = list()
    cnt = 0

    print(f'\nStarting the annotation process...')
    for example in data:
        annotated_example = Corpus(author=example['author'], title=example['file'],
                                   domain=example['domain'], text=example['text'])

        annotated_data.append(annotated_example)
        cnt += 1

        if cnt % 1000 == 0:
            print(cnt)

    print('Finished the annotation process!')
    return annotated_data

def write_annotated_data(data, path):
    with open(path, mode='w', encoding='utf8') as f:
        for example in data:
            f.write(json.dumps(example.to_dict()))
            f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--blogs_data_dir', help='Data dir with all domains for blogs')
    parser.add_argument('--blogs_domains', nargs='+', help='Blogs domains to annotated')
    parser.add_argument('--imdb_path', help='Path to imdb text file')
    parser.add_argument('--amazon_path', help='Path to amazon json file')
    parser.add_argument('--output_path', help='Path to aggregated data')

    args = parser.parse_args()

    blogs_corpus = read_data_blogs(data_dir=args.blogs_data_dir, domains=args.blogs_domains)
    blogs_corpus = filter_data(blogs_corpus, doc_threshold=30, doc_word_threshold=50, author_max_word=200000)

    imdb62 = read_data_imdb(path=args.imdb_path)
    imdb62 = filter_data(imdb62, doc_threshold=100, doc_word_threshold=50, author_max_word=200000)

    amazon = read_data_amazon(path=args.amazon_path)
    amazon = filter_data(amazon, doc_threshold=100, doc_word_threshold=50, author_max_word=200000)

    data = combine_datasets(blogs=blogs_corpus, imdb=imdb62, amazon=amazon, output_path=args.output_path)

    data_annotated = annotate_data(data)
    write_annotated_data(data_annotated, f'{args.output_path}.annotated')

    # data_annotated = annotate_data(load_data('/mnt/green-efs/bashar.alhafni/data_check/scaling_exps/train_1k.raw.json'))
    # write_annotated_data(data_annotated, '/mnt/green-efs/bashar.alhafni/data_check/scaling_exps/train_1k.raw.json.annotated')

    # data_annotated = annotate_data(load_data('/mnt/green-efs/bashar.alhafni/data_check/scaling_exps/train_5k.raw.json'))
    # write_annotated_data(data_annotated, '/mnt/green-efs/bashar.alhafni/data_check/scaling_exps/train_5k.raw.json.annotated')

    # data_annotated = annotate_data(load_data('/mnt/green-efs/bashar.alhafni/data_check/scaling_exps/train_10k.raw.json'))
    # write_annotated_data(data_annotated, '/mnt/green-efs/bashar.alhafni/data_check/scaling_exps/train_10k.raw.json.annotated')

    # data_annotated = annotate_data(load_data('/mnt/green-efs/bashar.alhafni/data_check/scaling_exps/train_20k.raw.json'))
    # write_annotated_data(data_annotated, '/mnt/green-efs/bashar.alhafni/data_check/scaling_exps/train_20k.raw.json.annotated')

    # data_annotated = annotate_data(load_data('/mnt/green-efs/bashar.alhafni/data_check/scaling_exps/train_35k.raw.json'))
    # write_annotated_data(data_annotated, '/mnt/green-efs/bashar.alhafni/data_check/scaling_exps/train_35k.raw.json.annotated')

