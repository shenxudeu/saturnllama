import os
from typing import List
import argparse

from sentencepiece import SentencePieceProcessor

TOKENIZER_MODEL = "tokenizer.model"

class Tokenizer:
    def __init__(self, tokenizer_model=None) -> None:
        model_path = tokenizer_model or os.path.join(os.path.dirname(__file__), TOKENIZER_MODEL)
        assert os.path.exists(model_path), f"Tokenizer model {model_path} does not exist"
        self.sp = SentencePieceProcessor(model_file=model_path)
        self.model_path = model_path

        self.n_words: int = self.sp.vocab_size()
        self.bos_id: int = self.sp.bos_id()
        self.eos_id: int = self.sp.eos_id()
        self.pad_id: int = self.sp.pad_id()
        assert self.sp.vocab_size() == self.sp.get_piece_size()

    def encode(self, text: str, bos: bool, eos: bool) -> List[int]:
        assert type(text) is str
        t = self.sp.encode(text)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t
    
    def decode(self, t: List[int]) -> str:
        return self.sp.decode(t)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-model", type=str, default=None)
    args = parser.parse_args()

    tokenizer = Tokenizer(args.tokenizer_model)
    test_str = "hello world, Shen!"
    token_ids = tokenizer.encode(test_str, bos=True, eos=True)
    print(test_str)
    print(token_ids)
    print(tokenizer.decode(token_ids))
