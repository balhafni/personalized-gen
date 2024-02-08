# Personalized Text Generation with Fine-Grained Linguistic Control

This repo contains code and pretrained models to reproduce the results in our paper [Personalized Text Generation with Fine-Grained Linguistic Control](https://arxiv.org/abs/2402.04914).

# Requirements:
The code was written for Python >=3.9, pytorch 2.1.2, and transformers 4.36.2. You will need a few additional packages. Here's how you can set up the environment using Conda (assuming you have Conda and Cuda installed):

```bash
conda create -n person-gen python=3.9
conda activate person-gen

pip install -r requirements.txt
```

# Experiments and Reproducibility:
We make the data we use to train and test our models publicly available in this [release](https://github.com/balhafni/personalized-gen/releases/tag/data). Details on how the data was obtained are described [here](data).

## Training:
Replicating our 1B Pythia baseline and Prefix models can be done using the [scripts/baseline.sh](scripts/baseline.sh) and [scripts/prefix.sh](scripts/prefix.sh) scripts, respectively. Both scripts can be used to also replicate the smaller Pythia models we report on in the paper.

## Inference and Evaluation:
Once the models are trained, we run the inference on the Dev set using all the models' checkpoints can be done using the [scripts/inference_checkpoints.sh](scripts/inference_checkpoints.sh). We pick the best checkpoint using the [eval/get_best_checkpoint.py](eval/get_best_checkpoint.py) based on the performance on the Dev set. We then run the inference on the Test set using the best checkpoint by using the [scripts/infernece.sh](scripts/inference.sh) script.


# Hugging Face Integration:
We make our best multi-attribute controlled model publicly available on [Hugging Face](https://huggingface.co/balhafni/personalized-gen):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained('balhafni/personalized-gen')
tokenizer = AutoTokenizer.from_pretrained('balhafni/personalized-gen')


ling_atts = {"ADJ": "5-8", "ADP": "10-11", "ADV": "6-8", "AUX": "9-11",
             "CONJ": "2-4", "DET": "7-10", "FKGL": "5-6", "NOUN": "11-18",
             "NUM": "2-3", "PART": "4-5", "PRON": "14-17", "PROPN": "8-11",
             "PUNCT": "22-25", "ROOT": "9-10", "SCONJ": "3-4", "VERB": "16-20",
             "acl": "0-1", "acomp": "1-2", "advcl": "2-3", "advmod": "7-9",
             "amod": "3-6", "appos": "0-1", "attr": "1-2", "attribution": "2-3",
             "aux": "6-7", "auxpass": "0-1", "case": "0-1", "cc": "2-4",
             "comp": "3-4", "compound": "5-6", "conj": "2-4", "contrast": "0-1",
             "det": "7-10", "dobj": "6-7", "domain": "blog",
             "elaboration": "10-12", "mark": "2-3", "neg": "2-3", "nmod": "0-1",
             "npadvmod": "1-2", "subj": "13-16", "nsubjpass": "0-1",
             "num_sents": "9-10", "num_tokens": "118-139", "nummod": "1-2",
             "pcomp": "0-1", "obj": "8-10", "poss": "2-3", "prep": "9-10"
             }

prompt = ("Today's lunch was a layered entree, consisting of, "
          "shredded lettuce and popcorn chicken.")

inputs = [''.join([f'{k}:{v}' for k, v in ling_atts.items()]) + prompt]
inputs = tokenizer(inputs, return_tensors='pt')

preds = model.generate(**inputs,
                       max_length=1024,
                       pad_token_id=tokenizer.pad_token_id,
                       no_repeat_ngram_size=2
                      )

decoded_preds = tokenizer.batch_decode(preds[:, inputs['input_ids'].shape[1]:],
                                       skip_special_tokens=True)[0]
output = prompt + ' ' + decoded_preds.strip()
print(output)
```

# License:
This repo is available under the MIT license. See the [LICENSE](LICENSE) for more info.


# Citation:
If you find the code or data in this repo helpful, please cite [our paper](https://arxiv.org/abs/2402.04914):

```BibTeX
@inproceedings{alhafni-etal-2024-personalized,
    title = "Personalized Text Generation with Fine-Grained Linguistic Control",
    author = "Alhafni, Bashar  and
      Kulkarni, Vivek  and
      Kumar, Dhurv  and
      Raheja, Vipul",
    month = march,
    year = "2024",
    address = "Malta",
    publisher = "Association for Computational Linguistics",
    abstract = "As the text generation capabilities of large language models become increasingly prominent, recent studies have focused on controlling particular aspects of the generated text to make it more personalized. However, most research on controllable text generation focuses on controlling the content or modeling specific high-level/coarse-grained attributes that reflect authors’ writing styles, such as formality, domain, or sentiment. In this paper, we focus on controlling fine-grained attributes spanning multiple linguistic dimensions, such as lexical and syntactic attributes. We introduce a novel benchmark to train generative models and evaluate their ability to generate personalized text based on multiple fine-grained linguistic attributes. We systematically investigate the performance of various large language models on our benchmark and draw insights from the factors that impact their performance. We make our code, data, and pretrained models publicly available.",
}
