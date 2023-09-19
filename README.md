## Saturn Llama2

<p align="center">
    <img src="assets/saturn_llama.png" width="300" height="300", alt="Saturn Llama">
</p>

Building upon Andrej's LLAMA2.C, I've crafted this streamlined prototype to train the LLM in PyTorch. This endeavor has enriched my comprehension of the LLM mechanism, delving deep into its tokenization and data loading intricacies. This prototype mirrors the LLM architecture, offering the flexibility to train from the ground up. Owing to the shared architectural design, it's feasible to load and make inferences using Meta's LLM models.

I've managed to smoothly run the training procedure on my M2 MacBook Pro.

## Files Description:

## `model.py`
It defines the Llama2-like architechture, including attention blocks and etc.

## `export.py`
It loads models (including Meta Llama2) checkpoints and construct the model.

## `tokenizer.py`
It uses sentencepiece to load Meta pretrained tokenizer or train your own tokenizer.

## `tinystories.py`
It downloads and process a tiny dataset created from GPT-4.
To tokenize data with Llama2 tokenizer:
```bash
python tinystories.py download
python tinystories.py pretokenize
```

To train a vocab: 
```bash
python tinystories.py download
python tinystories.py train_vocab --vocab_size=2048
python tinystories.py pretokenize
```

## `train.py`
It is used to train the model.
```bash
python train.py
```
