
from typing import BinaryIO, Tuple
import os
import json
import regex as re
from collections import Counter, defaultdict
from typing import Dict, List, Set, DefaultDict
import multiprocessing
from multiprocessing import Process, Pool
from collections import Counter
import time
import cProfile

import heapq


class Tokenizer:
    def __init__(self):
        self.num_processes = 8  
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
        #chunk, special_tokens_pattern, regex_pattern = args
        input_path, boundary_start, boundary_end, special_tokens_pattern, regex_pattern = args

        process_id = os.getpid()
        # Split on special tokens

        word_counter = Counter()
        # Open the file and read the chunk
        with open(input_path, "rb") as f:
            f.seek(boundary_start)
            chunk_data = f.read(boundary_end - boundary_start)
            chunk = chunk_data.decode("utf-8", errors="ignore")
            segments = re.split(special_tokens_pattern, chunk)

            #print(f"Process {process_id}: Split into {len(segments)} segments")
            # Apply pretokenization to each segment
            for segment in segments:
                if not segment:  # Skip empty segments
                    continue
                    
                matches = regex_pattern.finditer(segment)
                for match in matches:
                    pretoken = match.group(0).encode('utf-8')
                    word = tuple(bytes([b]) for b in pretoken)
                    word_counter[word] += 1
        return word_counter

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
            
            # Process chunks in parallel
            print(f"Starting pretokenization of chunks...")
            #args_list = [(chunk, special_tokens_pattern, regex_pattern) for chunk in chunks]
            args_list = [(input_path, boundary_start, boundary_end,  special_tokens_pattern, regex_pattern) for (boundary_start, boundary_end) in zip(boundaries[:-1], boundaries[1:])]
            
            
            #Profile the parallel processing
            # profiler = cProfile.Profile()
            # profiler.enable()
            
            with Pool(self.num_processes) as pool:
                chunk_counters = pool.map(self.pretokenize, args_list)
            
            words = sum(chunk_counters, Counter())

            # profiler.disable()
            # print("\nPretokenization profiling results:")
            # profiler.print_stats(sort='cumulative')

            
            #print("Finished collecting")
            # Convert each pretoken to a list of single-byte tokens
            # and count frequencies

            
            print("Started merging.")
            # At the beginning of the train_bpe method, before the loop:
            last_progress_time = time.time()
            
            pairs = {}
            for word, freq in words.items():
                for i in range(len(word) - 1):
                    pair = (word[i], word[i+1])
                    pairs[pair] = pairs.get(pair, 0) + freq
            
            # BPE training loop
            while len(self.vocabulary) < vocab_size:
                #current_time = time.time()
                # if current_time - last_progress_time >= 30:
                #     print(f"\rProgress: {len(self.vocabulary)}/{vocab_size} tokens ({len(self.vocabulary)/vocab_size*100:.1f}%)", end="", flush=True)
                #     last_progress_time = current_time
                # Inside the BPE training loop:
                #print(f"\rProgress: {len(self.vocabulary)}/{vocab_size} tokens ({len(self.vocabulary)/vocab_size*100:.1f}%)", end="")
                
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
                            # Remove the old pairs from the count
                            if i > 0:
                                pairs[(word_list[i-1], word_list[i])] -= freq
                            
                            # Remove the pair being merged
                            pairs[(word_list[i], word_list[i+1])] -= freq
                            
                            if i < len(word_list) - 2:
                                pairs[(word_list[i+1], word_list[i+2])] -= freq
                            
                            # Apply the merge
                            word_list[i] = new_token
                            del word_list[i+1]
                            
                            # Add the new pairs to the count
                            if i > 0:
                                pairs[(word_list[i-1], word_list[i])] = pairs.get((word_list[i-1], word_list[i]), 0) + freq
                            
                            if i < len(word_list) - 1:
                                pairs[(word_list[i], word_list[i+1])] = pairs.get((word_list[i], word_list[i+1]), 0) + freq
                        else:
                            i += 1
                    
                    # Convert back to tuple and update frequency
                    word_tuple = tuple(word_list)
                    new_words[word_tuple] = new_words.get(word_tuple, 0) + freq
                
                words = new_words
        
        return self.vocabulary, merges
