from typing import Tuple
import os
import json
import regex as re
from collections import Counter, defaultdict
from typing import Dict, List, Set, DefaultDict
import multiprocessing
from multiprocessing import Process, Pool
from collections import Counter

from pretokenization_example import find_chunk_boundaries


class Tokenizer:
    def __init__(self):
        self.num_processes = 16 # arbitrarily set to 16 here
        self.vocabulary = {} # Mapping from token ID to bytestring token
        self.token_ID = 0 # token ID that increments with each new token that's been added
        self.pretokenized_counts = defaultdict(int) # Map from tuple of bytes of each pretoken, to frequency of that pretoken, for merging
        
        for i in range(256):
            self.vocabulary[i] = bytes([i])
            self.token_ID += 1
        
        self.vocabulary[self.token_ID] = "<|endoftext|>".encode("utf-8")
        self.token_ID += 1


    def pretokenize(self, args):
        chunk, pattern, PAT_COMPILED = args

        # First, before running pre-tokenization, we have to split on all special tokens. 
        segment_splits = re.split(pattern, chunk)
        local_counts = Counter()
        for segment in segment_splits:
            # Pretokenize each segment
            iter = PAT_COMPILED.finditer(segment) # iter now contains an iterable of matches within each segment
            for x in iter:
                utf_x = tuple(x.group(0).encode("utf-8"))
                local_counts[utf_x] += 1
        
        return local_counts


    def train_bpe(self, input_path: str, vocab_size: str, special_tokens: list[str]) -> Tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

        with open(input_path, "rb") as bpe_train_data:  # raw text data to be tokenized
            boundaries = find_chunk_boundaries(bpe_train_data, self.num_processes, "<|endoftext|>".encode("utf-8"))

            pattern = "|".join(re.escape(token) for token in special_tokens) # We use re.escape since | could appear in special tokens
            PAT_COMPILED = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

            chunks = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                bpe_train_data.seek(start)
                chunk = bpe_train_data.read(end - start).decode("utf-8", errors="ignore")
                chunks.append(chunk)

            args_list = [(chunk, pattern, PAT_COMPILED) for chunk in chunks]

            with Pool(self.num_processes) as pool:
                results = pool.map(self.pretokenize, args_list)
        
            for r in results:
                for item, count in r.items():
                    self.pretokenized_counts[item] += count

        breakpoint()
                    

if __name__ == '__main__':
    # You might need this on Windows if creating frozen executables,
    # usually safe to include, but often not strictly necessary on macOS/Linux
    # for regular script execution.
    # from multiprocessing import freeze_support
    # freeze_support()

    T = Tokenizer()
    stories_valid_path = "data/TinyStoriesV2-GPT4-valid.txt"
    stories_train_path = "data/TinyStoriesV2-GPT4-train.txt"
    T.train_bpe(stories_valid_path, 0, [""]) # Pass a real vocab size if needed