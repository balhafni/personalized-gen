# Personalized Text Generation with Fine-Grained Linguistic Control

This repo contains code and pretrained models to reproduce the results in our paper [Personalized Text Generation with Fine-Grained Linguistic Control]().

# Requirements:
The code was written for Python >=3.9, pytorch 2.1.2, and transformers 4.36.2. You will need a few additional packages. Here's how you can set up the environment using Conda (assuming you have Conda and Cuda installed):

```bash
conda create -n person-gen python=3.9
conda activate person-gen

pip install -r requirements.txt
```

# Experiments and Reproducibility:
We make the data we use to train and test our models publicly available in this [release](). Details on how the data was obtained are described [here]().

## Training:
Replicating our 1B Pythia baseline and Prefix models can be done using the [scripts/baseline.sh](scripts/baseline.sh) and [scripts/prefix.sh](scripts/prefix.sh) scripts, respectively.

## Inference:

## Evaluation:


# Hugging Face Integration:
We make our models publicly available on [Hugging Face]()


# License:
This repo is available under the MIT license. See the [LICENSE](LICENSE) for more info.


# Citation:
If you find the code or data in this repo helpful, please cite [our paper]():

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
