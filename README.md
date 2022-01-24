# Invariant Language Modeling
Implementation of the training for invariant language models.

## Motivation

Modern pretrained language models are critical components of NLP pipelines. Yet, they suffer from spurious correlations, poor out-of-domain generalization, and biases.
Inspired by recent progress in causal machine learning, we propose __invariant language modeling__, a framework to learn invariant representations that should generalize across training environments.
In particular, we adapt [IRM-games](https://arxiv.org/abs/2002.04692) to language models, where the invariance emerges from a specific training schedule in which environments compete to optimize their environment-specific loss by updating subsets of the model in a round-robin fashion.

## Model Description

The data is assumed to come as `n` distinct environments and we aim to learn a language model that focusing on correlations that generalize across environments.

The model is decomposed into two components:
* `ϕ` the main body of the transformer language model,
* `w` the language modeling head that predicts the missing token.

In our implementation, there are now as many heads as environments: `n`.
For each data point, all heads make their predictions and they are averaged.
However, during training we sample one batch from each environment in a round-robin fashion.
When seeing a batch from environment `e` only the head `w_e` and the main body `ϕ` receive a batch update.

## Usage

To get started with the code:
```
pip install -r requirements.txt
```

PyTorch with a CUDA installation is required to run this framework.
Please find all useful installation information [here](https://pytorch.org/)

Then, to continue the training of a language model from a [huggingface](https://huggingface.co/models) checkpoint:
```
CUDA_VISIBLE_DEVICES=0 python3 run_invariant_mlm.py \
    --model_name_or_path roberta-base \
    --validation_file data-folder/validation_file.txt \
    --do_train \
    --do_eval \
    --nb_steps 5000 \
    --learning_rate 1e-5 \
    --output_dir folder-to-save-model \
    --seed 123 \
    --train_file data-folder/training-environments \
    --overwrite_cache
```
If the machine on which the code is executed has several GPUs, we recommand to use the `CUDA_VISIBLE_DEVICE` command to 
restrict to one GPU as the multiple GPUs are currently not supported by the implementation.

Currently, the supported base models are:
* `roberta`: [checkpoints](https://huggingface.co/models?sort=downloads&search=roberta)
* `distilbert`: [checkpoints](https://huggingface.co/models?sort=downloads&search=distilbert)

## Implementation

To train language models according to the [IRM-games](https://arxiv.org/abs/2002.04692), one needs to modify:
* the training schedule to perform batch updates according to each environment in a round-robin fashion.
This logic is implemented by the `InvariantTrainer` in `invariant_trainer.py', a class inherited from the `Trainer` from huggingface.
* the language modeling heads in the model.
It needs one head per environment.
This is done by creating variations of the base model classes. It is implemented in `invariant_roberta.py` for `roberta` and in `invariant_distilbert.py` for `distilbert`.

#### Contact

Maxime Peyrard, maxime.peyrard@epfl.ch
