import re
import glob
import os
from collections import defaultdict
import argparse

POST_PATTERN = re.compile('<post>[\s\S]*?<\/post>')

def read_data(path):
    files = glob.glob(f'{path}/*.xml')
    # take off docs belonging to 'Student' and 'indUnk'
    files = [f for f in files if 'Student' not in f and 'indUnk' not in f]
    text_by_author = defaultdict(list)

    for i, file in enumerate(files):
        
        with open(file, encoding='utf-8') as f: # only focusing on utf-8 files
            file = file.split('/')[-1]
            author, domain = file.split('.')[0], file.split('.')[3]
            try:
                text = f.read()
            except:
                print(f'Error reading: {file}')

            posts = POST_PATTERN.findall(text)

            for post in posts:
                post = post.replace('<post>','').replace('</post>', '').strip()
                post = post.replace('\n', ' ')
                post = re.sub(' +', ' ', post)

                if post != "": # take out empty posts
                    text_by_author[author].append({'text': post, 'domain': domain})

        if i > 0 and i % 100 == 0:
            print(i)

    print(f'Done reading data!\n')
    num_docs = sum(len(v) for v in text_by_author.values())
    print(f'There were {num_docs} blogs in total written by {len(text_by_author)} Authors\n')
    return text_by_author

def write_data(data, data_dir):
    # organizing data by domain
    print(f'Organizing data by domain....')
    data_by_domain = defaultdict(lambda: defaultdict(list))

    for author, corpora in data.items():
        for corpus in corpora:
            data_by_domain[corpus['domain']][author].append(corpus)

    print(f'There are {len(data_by_domain)} domains in total.')
    top_five = dict(sorted(data_by_domain.items(), key=lambda x: len(x[1]), reverse=True)[:5])
    print(f'The top 5 domains with the most docs are:')

    for domain  in top_five:
        num_authors = len(top_five[domain])
        num_docs = sum(len(v) for v in top_five[domain].values())

        print(f'\t{domain}: {num_docs} blogs written by {num_authors} authors')

    if not os.path.exists(data_dir):
       os.makedirs(data_dir)

    for domain, authors_data in data_by_domain.items():
       os.makedirs(os.path.join(data_dir, domain))

       for author, corpora in authors_data.items():
           for i, corpus in enumerate(corpora):
               with open(os.path.join(data_dir, domain, f'{author}_{i}.txt'), mode='w') as f:
                   f.write(corpus['text'])
                   f.write('\n')

    print(f'Done writing data to {data_dir}!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help='Directory of blogs.')
    parser.add_argument('--output_dir', help='Output dir to write data.')
    args = parser.parse_args()

    data = read_data(args.input_dir)

    write_data(data_dir=args.output_dir, data=data)
