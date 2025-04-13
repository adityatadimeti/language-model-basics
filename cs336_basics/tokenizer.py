from typing import BinaryIO, Tuple
import os
import json
import regex as re
from collections import Counter, defaultdict
from typing import Dict, List, Set, DefaultDict
import multiprocessing
from multiprocessing import Process, Pool
from collections import Counter

import heapq


class Tokenizer:
    def __init__(self):
        self.num_processes = 8  # Using 8 instead of 16 for potential speedup
        self.vocabulary = {}  # Mapping from token ID to bytestring token
        self.token_ID = 0  # token ID that increments with each new token that's been added
    

    def find_chunk_boundaries(self, 
        file: BinaryIO, 
        desired_num_chunks: int, 
        split_special_token: bytes
    ) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(split_special_token, bytes), (
            "Must represent special token as a bytestring"
        )

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))

    def pretokenize(self, args):
        """Process chunk, strip special tokens, and apply pretokenization."""
        chunk, special_tokens_pattern, regex_pattern = args
        
        # Split on special tokens
        segments = re.split(special_tokens_pattern, chunk)
        
        # Apply pretokenization to each segment
        pretokens = []
        for segment in segments:
            if not segment:  # Skip empty segments
                continue
                
            matches = regex_pattern.finditer(segment)
            for match in matches:
                pretoken = match.group(0).encode('utf-8')
                pretokens.append(pretoken)
        
        return pretokens

    def train_bpe(self, input_path: str, vocab_size: int, special_tokens: list[str]) -> Tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        # Initialize vocabulary with byte values
        self.vocabulary = {i: bytes([i]) for i in range(256)}
        self.token_ID = 256
        
        # Add special tokens to vocabulary
        for token_str in special_tokens:
            token_bytes = token_str.encode("utf-8")
            self.vocabulary[self.token_ID] = token_bytes
            self.token_ID += 1
        
        # List of merges to return
        merges = []
        
        with open(input_path, "rb") as bpe_train_data:
            # Find chunk boundaries using special token
            special_token_bytes = "<|endoftext|>".encode("utf-8")
            boundaries = self.find_chunk_boundaries(bpe_train_data, self.num_processes, special_token_bytes)
            
            # Create pattern for splitting on special tokens
            special_tokens_pattern = "|".join(re.escape(token) for token in special_tokens)
            
            # Compile regex pattern for pretokenization
            regex_pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
            
            # Process each chunk
            chunks = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                bpe_train_data.seek(start)
                chunk_data = bpe_train_data.read(end - start)
                chunk = chunk_data.decode("utf-8", errors="ignore")
                chunks.append(chunk)
            
            # Process chunks in parallel
            args_list = [(chunk, special_tokens_pattern, regex_pattern) for chunk in chunks]
            
            with Pool(self.num_processes) as pool:
                all_pretokens = []
                for result in pool.map(self.pretokenize, args_list):
                    all_pretokens.extend(result)
            
            # Convert each pretoken to a list of single-byte tokens
            # and count frequencies
            words = {}
            for pretoken in all_pretokens:
                # Convert to list of single bytes
                word = tuple(bytes([b]) for b in pretoken)
                words[word] = words.get(word, 0) + 1
            
            # BPE training loop
            while len(self.vocabulary) < vocab_size:
                # Count pair frequencies
                pairs = {}
                for word, freq in words.items():
                    for i in range(len(word) - 1):
                        pair = (word[i], word[i+1])
                        pairs[pair] = pairs.get(pair, 0) + freq
                
                if not pairs:
                    break
                
                # Find most frequent pair
                max_freq = -1
                best_pair = None
                for pair, freq in pairs.items():
                    if freq > max_freq or (freq == max_freq and pair > best_pair):
                        max_freq = freq
                        best_pair = pair
                
                # Add to merges list
                merges.append(best_pair)
                
                # Create new token and add to vocabulary
                first, second = best_pair
                new_token = first + second
                self.vocabulary[self.token_ID] = new_token
                self.token_ID += 1
                
                # Apply merge to all words
                new_words = {}
                for word, freq in words.items():
                    # Convert to list for easier manipulation
                    word_list = list(word)
                    i = 0
                    while i < len(word_list) - 1:
                        if word_list[i] == first and word_list[i+1] == second:
                            word_list[i] = new_token
                            del word_list[i+1]
                        else:
                            i += 1
                    
                    # Convert back to tuple and update frequency
                    word_tuple = tuple(word_list)
                    new_words[word_tuple] = new_words.get(word_tuple, 0) + freq
                
                words = new_words
        
        return self.vocabulary, merges