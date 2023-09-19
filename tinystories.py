import argparse
import glob
import os
import requests
from tqdm import tqdm
import json
import random

import numpy as np
import torch
import torch.distributed as dist
from functools import partial
import sentencepiece as spm
from concurrent.futures import ProcessPoolExecutor

from tokenizer import Tokenizer

DATA_CACHE_DIR = "data"


def download_file(url: str, fname: str, chunk_size=1024):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as f, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = f.write(data)
            bar.update(size)


def download():
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    data_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
    data_filename = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data.tar.gz")
    if not os.path.exists(data_filename):
        print(f"Downloading data from {data_url}")
        download_file(data_url, data_filename)
    else:
        print(f"Data already downloaded at {data_filename}")

    # unpack the tar.gz file into all the data shards (json files)
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        print(f"Unpacking data into {data_dir}")
        os.system(f"tar -xzf {data_filename} -C {data_dir}")
    else:
        print(f"Data already unpacked at {data_dir}")

    # print a single example just for debuging
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    with open(shard_filenames[0], "r") as f:
        data = json.load(f)
    print("Download done")
    print(f"Number of shards: {len(shard_filenames)}")
    print(f"Example data: {data[0]}")


def get_tokenizer_model_path(vocab_size):
    if vocab_size == 0:
        return None
    else:
        return os.path.join(DATA_CACHE_DIR, f"tokenized_{vocab_size}.model")


def process_shard(args, vocab_size):
    shard_id, shard = args
    tokenizer_model = get_tokenizer_model_path(vocab_size)
    enc = Tokenizer(tokenizer_model)
    with open(shard, "r") as f:
        data = json.load(f)
    all_tokens = []
    for example in tqdm(data, position=shard_id):
        text = example["story"]
        text = text.strip()
        tokens = enc.encode(text, bos=True, eos=False)
        all_tokens.extend(tokens)
    all_tokens = np.array(all_tokens, dtype=np.uint16)

    if vocab_size == 0:  # Llama2
        tokenized_filename = shard.replace(".json", ".bin")
    else:
        bin_dir = os.path.join(DATA_CACHE_DIR, f"tokenized_{vocab_size}")
        shard_basename = os.path.basename(shard)
        bin_basename = shard_basename.replace(".json", ".bin")
        tokenized_filename = os.path.join(bin_dir, bin_basename)

    with open(tokenized_filename, "wb") as f:
        f.write(all_tokens.tobytes())

    avg_seq_len = all_tokens.size / ((all_tokens == 1).sum())
    print(f"Shard {shard_id} done. Average sequence length: {avg_seq_len}")
    print(f"saved to {tokenized_filename}")


def pretokenize(vocab_size):
    # iterate the shards and tokenize them
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    if vocab_size > 0:
        bin_dir = os.path.join(DATA_CACHE_DIR, f"tokenized_{vocab_size}")
        os.makedirs(bin_dir, exist_ok=True)

    # process all the shards
    func = partial(process_shard, vocab_size=vocab_size)
    with ProcessPoolExecutor() as executor:
        executor.map(func, enumerate(shard_filenames))

    print("done")


def train_vocab(vocab_size):
    assert vocab_size > 0

    prefix = os.path.join(DATA_CACHE_DIR, f"tokenized_{vocab_size}")

    num_shards = 10

    tiny_file = os.path.join(DATA_CACHE_DIR, "tiny.txt")
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))

    # export a chunk of text as a single text file tiny.txt for training.
    print(f"Writing to {tiny_file} with {num_shards} shards")
    with open(tiny_file, "w", encoding="utf-8") as of:
        for shard in tqdm(shard_filenames[:num_shards]):
            with open(shard, "r") as f:
                data = json.load(f)
            for example in data:
                text = example["story"]
                text = text.strip()
                of.write(text + "\n")
    print(f"Size is {os.path.getsize(tiny_file) / 1024 / 1024:.2f} MB")

    # train the sentencepiece model
    print(f"Training sentencepiece model with vocab size {vocab_size}")
    spm.SentencePieceTrainer.train(input=tiny_file,
                                   model_prefix=prefix,
                                   model_type="bpe",
                                   vocab_size=vocab_size,
                                   self_test_sample_size=0,
                                   input_format="text",
                                   character_coverage=1.0,
                                   num_threads=4,
                                   split_digits=True,
                                   allow_whitespace_only_pieces=True,
                                   byte_fallback=True,
                                   unk_surface=" \342\201\207",
                                   normalization_rule_name="identity",
    )

    print(f"Training done. Model saved to {prefix}.model")


class PretokDataset(torch.utils.data.IterableDataset):
    def __init__(self, split, max_seq_len, vocab_size, vocab_source):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.vocab_source = vocab_source

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        # get DDP rank info
        rank = dist.get_rank() if dist.is_initialized() else 0
        # create a random seed
        seed = 1984 + worker_id + rank * 1000
        rng = random.Random(seed)
        print(f"created a PretokDataset with seed {seed}")
        if self.vocab_source == "llama2":
            bin_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
            shard_filenames = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))
        elif self.vocab_source == "custom":
            bin_dir = os.path.join(DATA_CACHE_DIR, f"tokenized_{self.vocab_size}")
            shard_filenames = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))
        shard_filenames = shard_filenames[1:] if self.split == "train" else shard_filenames[:1]
        assert len(shard_filenames) > 0, f"no shards found in {bin_dir}"

        while True:
            rng.shuffle(shard_filenames)
            for shard in shard_filenames:
                # open the dataset for reading but keep it on disk with memmap
                m = np.memmap(shard, dtype=np.uint16, mode="r")
                num_batches = len(m) // self.max_seq_len
                num_batches -= 1
                assert num_batches > 0, "this shard is way too small?"
                ixs = list(range(num_batches))
                rng.shuffle(ixs)
                for ix in ixs:
                    start = ix * self.max_seq_len
                    end = start + self.max_seq_len + 1
                    chunk = torch.from_numpy(m[start:end].astype(np.int64))
                    x = chunk[:-1]
                    y = chunk[1:]
                    yield x, y


class Task:
    @staticmethod
    def iter_batches(batch_size, device, num_workers=0, **dataset_kwargs):
        ds = PretokDataset(**dataset_kwargs)
        dl = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=num_workers,
        )
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yield x, y


if __name__ == "__main__":
    """
    These stages are run in order.

    To tokenize data with Llama2 tokenizer:
    python tinystories.py download
    python tinystories.py pretokenize

    To train a vocab: 
    python tinystories.py download
    python tinystories.py train_vocab --vocab_size=2048
    python tinystories.py pretokenize
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, choices=["download", "pretokenize", "train_vocab"])
    parser.add_argument("--vocab_size", type=int, default=0, help="0 for llama2")
    args = parser.parse_args()

    if args.stage == "download":
        download()
    elif args.stage == "train_vocab":
        train_vocab(vocab_size=args.vocab_size)
    elif args.stage == "pretokenize":
        pretokenize(vocab_size=args.vocab_size)
    else:
        raise ValueError(f"Unknown stage {args.stage}")