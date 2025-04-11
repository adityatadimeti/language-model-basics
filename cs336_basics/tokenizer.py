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
        self.num_processes = 16 # arbitrarily set to 16 here
        self.vocabulary = {} # Mapping from token ID to bytestring token
        self.token_ID = 0 # token ID that increments with each new token that's been added
        self.pretokenized_counts = defaultdict(int) # Map from tuple of bytes of each pretoken, to frequency of that pretoken, for merging
        
        for i in range(256):
            self.vocabulary[i] = bytes([i])
            self.token_ID += 1
        
        self.vocabulary[self.token_ID] = "<|endoftext|>".encode("utf-8")
        self.token_ID += 1
        
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


    def train_bpe(self, input_path: str, vocab_size: int, special_tokens: list[str]) -> Tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        merges = [] # tuple of bytes

        with open(input_path, "rb") as bpe_train_data:  # raw text data to be tokenized
            boundaries = self.find_chunk_boundaries(bpe_train_data, self.num_processes, "<|endoftext|>".encode("utf-8"))

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
        
            for r_val in results:
                for item, count in r_val.items():
                    self.pretokenized_counts[item] += count
            
            while (len(self.vocabulary) < vocab_size): # Keep merging until we hit the desired vocab_size
                max_heap = []
                for r_val in results:
                    # Convert Counter items to a list of (token, count) pairs
                    r_items = list(r_val.items())
                    for i in range(len(r_items) - 1):
                        # Each item is a tuple of (token, count)
                        token1, count1 = r_items[i]
                        token2, count2 = r_items[i + 1]
                        count = count1 + count2
                        heapq.heappush(max_heap, (-count, token1, token2))

                neg_freq, token1, token2 = heapq.heappop(max_heap)
                max_freq = -neg_freq

                while max_heap and (item := heapq.heappop(max_heap))[0] == neg_freq:
                    popped_tuple = item[1], item[2] # Tuple of token1, token2
                    # If we find an item with higher lexicographical value
                    if popped_tuple > (token1, token2):
                        token1, token2 = popped_tuple
                
                # Now merge
                
                new_key = token1 + token2
                for r_val in results:
                    r_val[new_key] = r_val[token1] + r_val[token2]
                    del r_val[token1]
                    del r_val[token2]

                # Finally, add to self.vocabulary and merges
                # Convert the merged tokens into bytes and add to vocabulary
                merged_token = bytes(new_key)  # new_key is already token1 + token2
                self.vocabulary[self.token_ID] = merged_token
                merges.append((bytes(token1), bytes(token2)))
                self.token_ID += 1
    
        return self.vocabulary, merges
                    

if __name__ == '__main__':
    # You might need this on Windows if creating frozen executables,
    # usually safe to include, but often not strictly necessary on macOS/Linux
    # for regular script execution.
    # from multiprocessing import freeze_support
    # freeze_support()

    T = Tokenizer()
    stories_valid_path = "data/TinyStoriesV2-GPT4-valid.txt"
    stories_train_path = "data/TinyStoriesV2-GPT4-train.txt"
    T.train_bpe(stories_valid_path, 300, [""]) 