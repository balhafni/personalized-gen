import spacy
import copy
import json
import re
from fkgl import corpus_fkgl
from collections import Counter, defaultdict
import numpy as np

spacy_en = spacy.load("en_core_web_sm")
POS_TAGS = spacy.parts_of_speech.IDS
DEP_RELS = spacy_en.get_pipe('parser').labels
RST_RELS = ['attribution', 'background', 'cause', 'comparison', 'condition',
            'contrast', 'elaboration', 'enablement', 'evaluation', 'explanation',
            'mannermeans', 'summary', 'temporal', 'topicchange', 'topiccomment'
            ]

class Corpus:
    def __init__(self, author, title, domain, text):
        self.author = author
        self.title = title 
        self.text = text
        self.domain = domain
        self.get_corpus_info()

    def get_corpus_info(self):
        self.tokens = []
        self.pos_tags = []
        self.lemmas = []
        self.dep_rels = []
        self.sents = []
        self.text = self.text.replace('\n', ' ').replace('\x00', '')
        self.text = re.sub(' +', ' ', self.text)

        data = spacy_en(self.text)

        self.sents = [sent.text for sent in data.sents]

        for token in data:
            self.tokens.append(token.text)
            if token.pos_ in ['SPACE', 'EOL', '']:
                self.pos_tags.append('X')
            elif token.pos_ == 'CCONJ':
                self.pos_tags.append('CONJ')
            else:
                self.pos_tags.append(token.pos_)
            self.lemmas.append(token.lemma_)
            self.dep_rels.append(token.dep_)

        self.fkgl = corpus_fkgl([self.text], tokenizer=spacy_en)

    def __repr__(self):
        return str(self.to_json_str())

    def to_json_str(self):
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

class InputExample:
    def __init__(self, text):
        self.text = text

    @classmethod
    def create_example(cls, path):
        with open(path, encoding='utf8') as f:
            data = [x.strip() for x in f.readlines()]
            assert len(data) == 1

        return cls(data[0])

    def __repr__(self):
        return self.text


def extract_rst_relations(rst_tree):
    rst_relations = []

    if rst_tree:
        rst_tree = re.sub('\s+ ', ' ', rst_tree.replace('(', ' ').replace(')', ' ')).strip()
        rst_tree = rst_tree.split()
        rst_relations = [x.split(':')[1] for x in rst_tree if x.startswith('satellite:') #or x.startswith('nucleus:')
                        ]

    return rst_relations


def doc_attributes(annotated_example):

    if len(set(annotated_example['tokens'])) == 0:
        import pdb; pdb.set_trace()
    # Lexical Variety: token / type ratio
    lexical_variety = len(annotated_example['tokens']) / len(set(annotated_example['tokens']))

    # Lexical Variety (lemma): token / type ratio
    lexical_variety_lem = len(annotated_example['lemmas']) / len(set(annotated_example['lemmas']))

    # Number of tokens
    num_tokens = len(annotated_example['tokens'])
    
    # Number of sentences
    num_sents = len(annotated_example['sents'])

    # Avg number of word length
    avg_word_length = round(np.array([len(token) for token in annotated_example['tokens']]).mean(), 2)

    fkgl = annotated_example['fkgl']

    features = {'domain': annotated_example['domain'], 'FKGL': fkgl, 'lex_variety': lexical_variety,
                'lexical_variety_lem': lexical_variety_lem,
                'num_tokens': num_tokens, 'num_sents': num_sents,
                'avg_word_length': avg_word_length}

    pos_tags = {pos: 0 for pos in POS_TAGS if pos not in ['SPACE', 'EOL', '', 'CCONJ']}
    pos_tags_cnts  = Counter(annotated_example['pos_tags'])
    for pos, cnt in pos_tags_cnts.items():
        pos_tags[pos] += cnt
    # normalizing the pos tags
    # pos_tags = {tag: v / sum(pos_tags.values()) for tag, v in pos_tags.items()}

    dep_rels = {rel: 0 for rel in DEP_RELS}
    dep_rels_cnts = Counter(annotated_example['dep_rels'])
    for rel, cnt in dep_rels_cnts.items():
        dep_rels[rel] += cnt

    # normalizing the dep rels
    # dep_rels = {tag: v / sum(dep_rels_cnts.values()) for tag, v in dep_rels_cnts.items()}

    rst_rels = {rel: 0 for rel in RST_RELS}
    rst_rels_cnts = Counter(extract_rst_relations(annotated_example['rst_tree']))
    for rel, cnt in rst_rels_cnts.items():
        rst_rels[rel] += cnt


    for pos, cnt in pos_tags.items():
        features[pos] = cnt

    for rel, cnt in dep_rels.items():
        features[rel] = cnt

    for rel, cnt in rst_rels.items():
        features[rel] = cnt

    # sort features by key
    sorted_keys = sorted(list(features.keys()))

    features = {k: features[k] for k in sorted_keys}

    return features


def add_attributes(annotated_examples):
    data_by_authors = defaultdict(list)
    # grouping data by authors
    for example in annotated_examples:
        data_by_authors[example['author']].append(example)

    dataset_with_atts = defaultdict(list)
    outliers_removed = defaultdict(int)

    # creating author-level attributes
    for author in data_by_authors:
        num_authors = 0 
        tmp = []
        domain = data_by_authors[author][0]['domain']

        for doc in data_by_authors[author]:

            doc_atts = doc_attributes(doc)

            tmp.append({'doc_atts': doc_atts, 'text': doc['text'], 'author': doc['author'], 'title': doc['title']})

            assert doc['domain'] == domain

        if len(tmp) == 0:
            continue

        # remove the outliers from the examples of each author
        author_examples = remove_outliers(tmp)

        outliers_removed[domain] += len(tmp) - len(author_examples)
        assert len(set([len(ex['doc_atts']) for ex in author_examples])) == 1

        att_keys = author_examples[0]['doc_atts'].keys()

        # creating author atts by averaging the docs atts for that author
        avg_atts = {k: sum([x['doc_atts'][k] for x in author_examples]) / len(author_examples) for k in att_keys if k != 'domain'}
        avg_atts['domain'] = domain
        num_authors += 1

        for example in author_examples:
            example['author_atts'] = avg_atts
            assert example['author'] == author
            dataset_with_atts[author].append(example)


    total_docs = sum([len(v) for v in dataset_with_atts.values()])
    print(f'There are {total_docs} docs in total!')
    print(f'{sum(outliers_removed.values())} outlier docs were removed!\n')

    return dataset_with_atts


def remove_outliers(author_examples, outlier_removal_threshold=.1):
    atts = author_examples[0]['doc_atts'].keys()
    attribute_zscore = dict()

    for att in atts:
        if att != 'domain':
            attribute_zscore[att] = np.array([example['doc_atts'][att] for example in author_examples])
            # compute the zscore for each attribute
            attribute_zscore[att] = (attribute_zscore[att] - attribute_zscore[att].mean()) / attribute_zscore[att].std()

    # for each example, get the attributes that are 2 std away from the mean
    outlier_atts = {}
    for att in atts:
        if att != 'domain':
            outlier_examples = np.where(attribute_zscore[att] > 2)[0]
            outlier_atts[att] = outlier_examples

    # count the number of attributes in each example that are outliers
    ex_outlier_cnt = Counter([ex_num for ex_nums in outlier_atts.values() for ex_num in ex_nums])

    # remove examples where the number of outlier attributes are more than
    # the outlier_removal_threshold

    new_examples = []
    removed_examples = []

    for id, example in enumerate(author_examples):

        if id in ex_outlier_cnt:
            if ex_outlier_cnt[id] / (len(atts) - 1) > outlier_removal_threshold:
                removed_examples.append(example)
            else:
                new_examples.append(example)
        else:
            new_examples.append(example)

    assert len(removed_examples) + len(new_examples) == len(author_examples)

    return new_examples