import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
import argparse
import spacy
import re

spacy_en = spacy.load("en_core_web_sm")


class MyDataset(Dataset):
    def __init__(self, tokenizer, config, examples, mode='prepend'):
        self.tokenizer = tokenizer
        self.examples = examples
        self.config = config
        self.mode = mode
        self.sents = []

        if self.mode == 'prepend':
            self.tokenizer.padding_side = 'left'
            for ex in examples:
                ex_sents = [sent.text for sent in spacy_en(ex['text']).sents]
                self.sents.append(ex_sents[0])

            self.prompts = [''.join([f'{k}:{v}' for k, v in ex['author_atts'].items()]) + sent
                            for ex, sent in zip(examples, self.sents)]

            self.feats = self.tokenizer(self.prompts, padding=True, truncation=True, max_length=1024, return_tensors='pt')

        elif self.mode == 'baseline':
            self.tokenizer.padding_side = 'left'
            for ex in examples:
                ex_sents = [sent.text for sent in spacy_en(ex['text']).sents]
                self.sents.append(ex_sents[0])

            self.feats = self.tokenizer(self.sents, padding=True, truncation=True, max_length=1024, return_tensors='pt')

    def __getitem__(self, idx):
        input_ids = self.feats['input_ids'][idx]
        attention_mask = self.feats['attention_mask'][idx]

        feat = {'input_ids': input_ids, 'attention_mask': attention_mask}

        return feat

    def __len__(self):
        return len(self.examples)


class Collator:
    def __init__(self, attributes):
        self.attributes = attributes

    def __call__(self, batch):
        input_ids = [x['input_ids'] for x in batch]
        attention_mask = [x['attention_mask'] for x in batch]
        attributes = dict()
        for attribute in self.attributes:
            attributes[attribute] = torch.tensor([x[attribute] for x in batch])
        return {'input_ids': torch.stack(input_ids), 'attention_mask': torch.stack(attention_mask), 'attributes': attributes}


def load_data(path):
    with open(path) as f:
        data = [json.loads(l) for l in f.readlines()]
    return data


def generate(model, tokenizer, test_dataset, output_path):
    loader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=False)

    accelerator = Accelerator()
    model, loader = accelerator.prepare(model, loader)

    all_outputs = []

    samples_seen = 0

    with torch.no_grad():
        for step, batch in enumerate(tqdm(loader)):
            preds = accelerator.unwrap_model(model).generate(**batch, max_length=1024,
                                                             pad_token_id=tokenizer.pad_token_id,
                                                             no_repeat_ngram_size=2)
            preds = accelerator.pad_across_processes(preds, dim=1, pad_index=tokenizer.pad_token_id)
            preds = accelerator.gather(preds).cpu().numpy()

            decoded_preds = tokenizer.batch_decode(preds[:, batch['input_ids'].shape[1]:], skip_special_tokens=True)

            if accelerator.num_processes > 1:
                if step == len(loader) - 1:
                    decoded_preds = decoded_preds[: len(loader.dataset) - samples_seen]
                else:
                    samples_seen += len(decoded_preds)

            all_outputs.extend(decoded_preds)


    accelerator.wait_for_everyone()
    assert len(test_dataset) == len(all_outputs)

    # adding the first sentence to the predictions
    final_outputs = []
    for r, sent in zip(all_outputs, test_dataset.sents):
        clear_r = r.strip().replace('\n', ' ')
        clear_r = re.sub(' +', ' ', clear_r)
        final_outputs.append(sent + ' ' + clear_r)

    assert len(final_outputs) == len(test_dataset)
    authors = [x['author'] for x in test_dataset.examples]
    titles = [x['title'] for x in test_dataset.examples]
    domains = [x['author_atts']['domain'] for x in test_dataset.examples]

    with open(output_path, mode='w', encoding='utf8') as f:
        for pred, author, title, domain in zip(final_outputs, authors, titles, domains):
            out_ex = {'text': pred.strip(), 'author': author,
                      'domain': domain, 'title': title
                     }
            f.write(json.dumps(out_ex))
            f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir')
    parser.add_argument('--test_file')
    parser.add_argument('--mode')
    parser.add_argument('--output_path')
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(args.model_dir)

    data = load_data(args.test_file)

    test_dataset = MyDataset(tokenizer, examples=data, config=config, mode=args.mode)

    generate(model, tokenizer, test_dataset,
             output_path=args.output_path,
             )
