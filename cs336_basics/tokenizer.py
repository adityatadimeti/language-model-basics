from typing import Tuple
import os
import json
import re
from collections import Counter, defaultdict
from typing import Dict, List, Set, DefaultDict
import multiprocessing

from pretokenization_example import find_chunk_boundaries


class Tokenizer:
    def __init__(self):
        self.num_chunks = 16 # arbitrarily set to 16 here
        self.vocabulary = {} # Mapping from token ID to bytestring token
        self.token_ID = 0 # token ID that increments with each new token that's been added
        
        for i in range(256):
            self.vocabulary[i] = bytes([i])
            self.token_ID += 1
        
        self.vocabulary[self.token_ID] = "<|endoftext|>".encode("utf-8")
        self.token_ID += 1


    def train_bpe(self, input_path: str, vocab_size: str, special_tokens: list[str]) -> Tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        bpe_train_data = open(input_path, "r") # raw text data to be tokenized

        num_processes = 1
        with open(..., "rb") as f:
            boundaries = find_chunk_boundaries(
                f, num_processes, "<|endoftext|>".encode("utf-8"))
                
            # The following is a serial implementation, but you can parallelize this 
            # by sending each start/end pair to a set of processes.
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                # Run pre-tokenization on your chunk and store the counts for each pre-token
        
        chunk_list = find_chunk_boundaries(bpe_train_data, self.num_chunks, 


        return 
