from collections.abc import Iterable
import io
import math
import pickle
from typing import BinaryIO, Iterator, Tuple
import os
import json
import regex as re
from collections import Counter, defaultdict
from typing import Dict, List, Set, DefaultDict, Tuple
import multiprocessing
from multiprocessing import Process, Pool
import time
import cProfile
from tqdm import tqdm
import heapq



class Tokenizer:
    def __init__(self, vocab: dict[int, bytes] | None = None, merges: list[tuple[bytes, bytes]] | None=None, special_tokens: list[str] | None=None):
        """Initialize the tokenizer with optional vocabulary, merges, and special tokens."""
        self.num_processes = 8  
        self.vocabulary = {} # id to pretoken bytes
        self.inv_vocab = {} # pretoken bytes to id
        self.special_tokens = []
        self.token_ID = 0  # token ID that increments with each new token that's been added
        self._merge_map = {}

        if vocab:
            self.vocabulary = vocab  # Mapping from token ID to bytestring token
            self.inv_vocab = {token: id for id, token in vocab.items()}
        if special_tokens:
            self.special_tokens = special_tokens
        if merges:
            self.merges = merges
            # in __init__ after self.merges = merges
            self._merge_ranks = { pair: i for i, pair in enumerate(self.merges) }

            self._merge_map = { (f, s): f + s for f, s in self.merges }
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        """
        Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) and (optionally) a list of special
        tokens.  
        """

        return cls(pickle.load(open(vocab_filepath, "rb")), pickle.load(open(merges_filepath, "rb")), special_tokens=special_tokens,)


    def _apply_bpe_merges(self, pretoken: bytes) -> list[bytes]:
        """
        Apply the learned BPE merges to a single pre‑token.

        Parameters
        ----------
        pretoken : bytes
            Raw UTF‑8 bytes of the pre‑token (e.g. b'the').

        Returns
        -------
        list[bytes]
            Sequence of bytes tokens after all merges (e.g. [b'the']).
        """
        # Start as list of 1‑byte tokens

        # tokens = [bytes([b]) for b in pretoken]

        # for first, second in self.merges:          # each is bytes
        #     i = 0
        #     # scan left‑to‑right, merging in place
        #     while i < len(tokens) - 1:
        #         if tokens[i] == first and tokens[i + 1] == second:
        #             tokens[i : i + 2] = [first + second]   # replace pair
        #             # stay at same index so we can catch cascaded merges
        #             if i:                                  # move back one step
        #                 i -= 1
        #         else:
        #             i += 1
        # return tokens

        # initialize as list of single‐byte tokens

        # start as list of single‐byte tokens
        tokens = [bytes([b]) for b in pretoken]
        rank = self._merge_ranks

        # keep merging until no more valid pairs
        while True:
            best_i = None
            best_rank = None

            # scan for adjacent pairs
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i+1])
                r = rank.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i

            # if we found a pair to merge, do it
            if best_i is not None:
                tokens[best_i:best_i+2] = [tokens[best_i] + tokens[best_i+1]]
                # then loop again
            else:
                break

        return tokens




    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.
        """

        """
        Step 1: Pretokenize, representing each pretoken as a sequence of UTF-8 bytes
        
        """
        desired_chunk_size = 100 * 1024 * 1024   # 100 MB
        text_size = len(text) # rough approximation

        min_chunks = math.ceil(text_size / desired_chunk_size)
        desired_num_chunks = max(self.num_processes, min_chunks)

        num_workers = self.num_processes

        # Stats
        chunks_per_worker = desired_num_chunks / num_workers
        chunk_size_gb = desired_chunk_size / (1000**3)
        gigs_per_worker = chunks_per_worker * chunk_size_gb

        print(f"File size:               {text_size / (1000**3):.2f} GB")
        print(f"Fixed chunk size:        {chunk_size_gb:.2f} GB")
        print(f"Desired chunks:          {desired_num_chunks}")
        print(f"Num workers:             {num_workers}")
        print(f"Chunks per worker:       {chunks_per_worker:.2f}")
        print(f"→ Gigs per worker (avg): {gigs_per_worker:.2f} GB")
        
        blob = text.encode("utf‑8")
        
        if not self.special_tokens:
            special_token_bytes = "<|endoftext|>".encode("utf-8")
        else:
            special_token_bytes = max(self.special_tokens, key=len).encode("utf-8")
        boundaries = self.find_chunk_boundaries_from_bytes(blob, desired_num_chunks, special_token_bytes)
        
        # Create pattern for splitting on special tokens

        sorted_tokens = sorted(self.special_tokens, key= len, reverse=True)
        # 2) Escape and join into an alternation
        escaped = [re.escape(tok) for tok in sorted_tokens]
        special_tokens_pattern = re.compile(f"({'|'.join(escaped)})")
        # 3) Use a capturing group around the whole alternation
        #special_tokens_pattern = re.compile(f"({special_tokens_pattern})")
        #special_tokens_pattern = "|".join(re.escape(token) for token in self.special_tokens)
        
        # Compile regex pattern for pretokenization
        regex_pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        args_list = [(blob, boundary_start, boundary_end, special_tokens_pattern, regex_pattern) 
                        for (boundary_start, boundary_end) in zip(boundaries[:-1], boundaries[1:])]
        
        all_pretokens = []
        with Pool(num_workers) as pool:
            iterator = pool.imap(self.pretokenize_encode, args_list, chunksize=1)
            
            for sub in tqdm(iterator,
                    total=len(args_list),
                    desc="Pretokenizing chunks",
                    dynamic_ncols=True):
                all_pretokens.extend(sub)  

        """
        Apply merging 
        """
        final_ids: list[int] = []
        for pretoken in tqdm(all_pretokens, desc="Applying BPE merges", dynamic_ncols=True):
            if pretoken.decode('utf-8', errors="replace") in self.special_tokens:
                final_ids.append(self.inv_vocab[pretoken])
                continue
            merged_tokens = self._apply_bpe_merges(pretoken)  # list[bytes]

            for tok in merged_tokens:
                try:
                    final_ids.append(self.inv_vocab[tok])
                except KeyError:
                    raise ValueError(f"Token {tok!r} not found in vocabulary")

        return final_ids
    

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Lazily tokenize lines without loading the entire text into memory.
        """
        if self.special_tokens:
            sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
            split_re = re.compile(f"({'|'.join(re.escape(tok) for tok in sorted_tokens)})")
        else:
            split_re = None

        regex_pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        from itertools import tee
        iterable, iterable_copy = tee(iterable)
        length = sum(1 for _ in iterable_copy)

        for chunk in tqdm(iterable,
                      desc="Streaming BPE-encode",
                      total=36990,          
                      dynamic_ncols=True):
            segments = split_re.split(chunk) if split_re else [chunk]
            for seg in segments:
                if not seg:
                    continue
                if seg in self.special_tokens:
                    yield self.inv_vocab[seg.encode("utf-8")]
                    continue
                for m in regex_pattern.finditer(seg):
                    pre = m.group(0).encode("utf-8")
                    for merged in self._apply_bpe_merges(pre):
                        try:
                            yield self.inv_vocab[merged]
                        except KeyError:
                            raise ValueError(f"Token {merged!r} not in vocabulary")

    
    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        output = ""
        replacement = "\uFFFD"

        byte_sequence = []
        for id in ids:
            if id in self.vocabulary:
                byte_sequence.append(self.vocabulary[id])
            else:
                pass
        
        combined_bytes = b''.join(byte_sequence)
        return combined_bytes.decode("utf-8", errors="replace")

    def find_chunk_boundaries_from_bytes(
        self,
        blob: bytes,
        desired_num_chunks: int,
        split_special_token: bytes,
        mini_chunk_size: int = 4096,
    ) -> List[int]:
        if not isinstance(split_special_token, bytes):
            raise TypeError("split_special_token must be bytes")

        file_size = len(blob)
        if file_size == 0:
            return [0]

        chunk_size = file_size // desired_num_chunks
        boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        boundaries[-1] = file_size  # ensure final boundary == EOF

        for bi in range(1, len(boundaries) - 1):
            pos = boundaries[bi]
            while True:
                if pos >= file_size:               # reached EOF
                    boundaries[bi] = file_size
                    break

                window = blob[pos : min(pos + mini_chunk_size, file_size)]
                if not window:                     # safety check
                    boundaries[bi] = file_size
                    break

                idx = window.find(split_special_token)
                if idx != -1:                      # found delimiter
                    boundaries[bi] = pos + idx
                    break

                pos += mini_chunk_size            # advance window

        return sorted(set(boundaries))

        
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
        input_path, boundary_start, boundary_end, special_tokens_pattern, regex_pattern = args

        word_counter = Counter()
        # Open the file and read the chunk
        with open(input_path, "rb") as f:
            f.seek(boundary_start)
            chunk_data = f.read(boundary_end - boundary_start)
            chunk = chunk_data.decode("utf-8", errors="ignore")
            segments = re.split(special_tokens_pattern, chunk)

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
    
    def pretokenize_encode(self, args):
        source, boundary_start, boundary_end, special_tokens_pattern, regex_pattern = args

        chunk_data = source[boundary_start:boundary_end] 
        chunk = chunk_data.decode("utf-8", errors="ignore")

        special_tokens_dict = {}
        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        special_tokens_positions = []
        
        # Find all occurrences of each special token
        for token in sorted_special_tokens:
            start_pos = 0
            while True:
                start_pos = chunk.find(token, start_pos)
                if start_pos == -1:
                    break
                end_pos = start_pos + len(token)
                special_tokens_positions.append((start_pos, end_pos, token))
                start_pos += 1  # Move forward to find overlapping instances
        
        # Sort positions by start index
        special_tokens_positions.sort(key=lambda x: x[0])

        filtered_positions = []
        last_end = -1
        
        for start, end, token in special_tokens_positions:
            if start >= last_end:  # No overlap with previous token
                filtered_positions.append((start, end, token))
                last_end = end
        
        # Process the text sequentially
        pretoken_list = []
        last_end = 0
        
        for start, end, token in filtered_positions:
            # If there's a gap before this special token, process it normally
            if start > last_end:
                segment = chunk[last_end:start]
                pretoken_list.extend(
                    m.group(0).encode("utf-8")
                    for m in regex_pattern.finditer(segment)
                )
            
            # Add the special token as a whole
            pretoken_list.append(token.encode("utf-8"))
            last_end = end
        
        # Process any remaining text after the last special token
        if last_end < len(chunk):
            segment = chunk[last_end:]
            pretoken_list.extend(
                m.group(0).encode("utf-8")
                for m in regex_pattern.finditer(segment)
            )
        
        return pretoken_list



    @staticmethod
    def build_pair_indices_chunk(args):
        """
        Given a slice of words and their frequencies, return a mapping from
        byte-pair to a dict of {global_word_index: frequency}.
        """
        start, end, words_slice, freqs_slice = args
        index_map = {}
        for offset, word in enumerate(words_slice):
            global_idx = start + offset
            count = freqs_slice[offset]
            for pos in range(len(word) - 1):
                pair = (word[pos], word[pos + 1])
                index_map.setdefault(pair, {})[global_idx] = count
        return index_map

    def _refresh_counts_after_merge(
        self,
        encoded_pretokens,
        word_idx,
        pos,
        merged_token,
        pair_counts,
        pair_to_indices,
        freq
    ):
        """
        Adjust pair_counts and pair_to_indices after merging a pair at `pos`
        in word #`word_idx` into `merged_token`.
        """
        seq = encoded_pretokens[word_idx]
        neighbors = []
        # left neighbor
        if pos > 0:
            neighbors.append((pos - 1, pos))
        # right neighbor
        if pos < len(seq) - 2:
            neighbors.append((pos + 1, pos + 2))

        for i, j in neighbors:
            old = (seq[i], seq[j])
            # decrement old count
            if old in pair_counts:
                new_cnt = pair_counts[old] - freq
                if new_cnt <= 0:
                    pair_counts.pop(old)
                else:
                    pair_counts[old] = new_cnt
            # compute new pair
            if i < pos:
                new_pair = (seq[i], merged_token)
            else:
                new_pair = (merged_token, seq[j])
            # increment new count
            pair_counts[new_pair] = pair_counts.get(new_pair, 0) + freq
            # update indices
            pair_to_indices.setdefault(new_pair, {})[word_idx] = freq

    def _prune_obsolete_indices(
        self,
        encoded_pretokens,
        word_idx,
        pos,
        pair_to_indices
    ):
        """
        Remove entries for pairs that no longer occur in the word after merge.
        """
        seq = encoded_pretokens[word_idx]
        def exists(pair):
            return any(seq[k] == pair[0] and seq[k+1] == pair[1]
                       for k in range(len(seq) - 1))

        for start in (pos - 1, pos):
            if 0 <= start < len(seq) - 1:
                bigram = (seq[start], seq[start + 1])
                indices = pair_to_indices.get(bigram)
                if indices and word_idx in indices:
                    if not exists(bigram):
                        indices.remove(word_idx)
                        if not indices:
                            # no more words contain this bigram
                            pair_to_indices.pop(bigram, None)

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
            bpe_train_data.seek(0, os.SEEK_END)
        
            desired_chunk_size = 100 * 1024 * 1024   # 100 MB
            file_size = os.path.getsize(input_path)

            min_chunks = math.ceil(file_size / desired_chunk_size)
            desired_num_chunks = max(self.num_processes, min_chunks)

            num_workers = self.num_processes

            # Stats
            chunks_per_worker = desired_num_chunks / num_workers
            chunk_size_gb = desired_chunk_size / (1000**3)
            gigs_per_worker = chunks_per_worker * chunk_size_gb

            print(f"File size:               {file_size / (1000**3):.2f} GB")
            print(f"Fixed chunk size:        {chunk_size_gb:.2f} GB")
            print(f"Desired chunks:          {desired_num_chunks}")
            print(f"Num workers:             {num_workers}")
            print(f"Chunks per worker:       {chunks_per_worker:.2f}")
            print(f"→ Gigs per worker (avg): {gigs_per_worker:.2f} GB")

            special_token_bytes = "<|endoftext|>".encode("utf-8")
            boundaries = self.find_chunk_boundaries(bpe_train_data, desired_num_chunks, special_token_bytes)
            
            # Create pattern for splitting on special tokens
            special_tokens_pattern = "|".join(re.escape(token) for token in special_tokens)
            
            # Compile regex pattern for pretokenization
            regex_pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
            args_list = [(input_path, boundary_start, boundary_end, special_tokens_pattern, regex_pattern) 
                         for (boundary_start, boundary_end) in zip(boundaries[:-1], boundaries[1:])]
            
            with Pool(num_workers) as pool:
                iterator = pool.imap(self.pretokenize, args_list, chunksize=1)
                chunk_counters = list(
                    tqdm(iterator,
                        total=len(args_list),
                        desc="Pretokenizing chunks",
                        dynamic_ncols=True)
                )
            
            # Combine all counters
            words_counter = sum(chunk_counters, Counter())
            
            # Convert tuple words to lists for easier modification
            encoded_pretokens = []
            pretoken_frequencies = []
            
            print("Converting words to lists...")
            for word, freq in tqdm(words_counter.items(), desc="Preparing words", dynamic_ncols=True):
                encoded_pretokens.append(list(word))
                pretoken_frequencies.append(freq)
            
            # Calculate initial byte pair frequencies
            byte_pair_frequencies = {}
            
            print("Computing initial pair frequencies...")
            for i, pretoken in enumerate(tqdm(encoded_pretokens, desc="Computing pairs", dynamic_ncols=True)):
                freq = pretoken_frequencies[i]
                for j in range(len(pretoken) - 1):
                    pair = (pretoken[j], pretoken[j+1])
                    byte_pair_frequencies[pair] = byte_pair_frequencies.get(pair, 0) + freq
            
            # Create token_indices mapping each pair to {word_idx: frequency}
            # Use parallel processing for building this mapping
            
            print("Building pair-to-word index mappings (multiprocessed)...")
            # Split words into chunks for parallel processing
            chunk_size = max(1, len(encoded_pretokens) // (num_workers * 4))
            chunks = []
            
            for i in range(0, len(encoded_pretokens), chunk_size):
                end_idx = min(i + chunk_size, len(encoded_pretokens))
                chunks.append((i, end_idx, encoded_pretokens[i:end_idx], pretoken_frequencies[i:end_idx]))
            
            # Process chunks in parallel
            with Pool(num_workers) as pool:
                result_iter = pool.imap(self.build_pair_indices_chunk, chunks)
                chunk_results = list(tqdm(result_iter, total=len(chunks), desc="Building indices", dynamic_ncols=True))
            
            # Merge results from all workers
            token_indices = {}
            for local_result in chunk_results:
                for pair, indices_dict in local_result.items():
                    if pair not in token_indices:
                        token_indices[pair] = {}
                    token_indices[pair].update(indices_dict)
            
            # BPE merging loop
            pbar = tqdm(total=vocab_size - len(self.vocabulary), desc="Merging step")
            
            while len(self.vocabulary) < vocab_size:
                if not byte_pair_frequencies:
                    break
                
                # Find the best pair using single-pass max() with custom key function
                # This combines frequency and lexicographic ordering in one step
                best_pair = max(byte_pair_frequencies, key=lambda x: (byte_pair_frequencies[x], x))
                
                # Add to merges list
                merges.append(best_pair)
                
                # Create new token and add to vocabulary
                first, second = best_pair
                new_token = first + second
                self.vocabulary[self.token_ID] = new_token
                self.token_ID += 1
                
                # Get the pretoken indices affected by this merge
                if best_pair not in token_indices:
                    pbar.update(1)
                    continue
                    
                affected_indices = list(token_indices[best_pair].keys())
                
                # Delete the pair from tracking structures
                del byte_pair_frequencies[best_pair]
                del token_indices[best_pair]
                
                # Apply merges to affected words
                for pretoken_idx in affected_indices:
                    # Skip invalid indices
                    if pretoken_idx >= len(encoded_pretokens):
                        continue
                        
                    # Get frequency of this word
                    freq = pretoken_frequencies[pretoken_idx]
                    
                    # Skip if this word was already processed (frequency is 0)
                    if freq == 0:
                        continue
                    
                    # Process all occurrences of the pair in this word
                    byte_idx = 0
                    while byte_idx < len(encoded_pretokens[pretoken_idx]) - 1:
                        # Check if the pair still exists at this position
                        if (byte_idx < len(encoded_pretokens[pretoken_idx]) - 1 and
                            encoded_pretokens[pretoken_idx][byte_idx] == first and 
                            encoded_pretokens[pretoken_idx][byte_idx + 1] == second):

                            seq = encoded_pretokens[pretoken_idx]
                            neighbors = []
                            # left neighbor
                            if byte_idx > 0:
                                neighbors.append((byte_idx - 1, byte_idx))
                            # right neighbor
                            if byte_idx < len(seq) - 2:
                                neighbors.append((byte_idx + 1, byte_idx + 2))

                            for i, j in neighbors:
                                old = (seq[i], seq[j])
                                # decrement old count
                                if old in byte_pair_frequencies:
                                    new_cnt = byte_pair_frequencies[old] - freq
                                    if new_cnt <= 0:
                                        byte_pair_frequencies.pop(old)
                                    else:
                                        byte_pair_frequencies[old] = new_cnt
                                # compute new pair
                                if i < byte_idx:
                                    new_pair = (seq[i], new_token)
                                else:
                                    new_pair = (new_token, seq[j])
                                # increment new count
                                byte_pair_frequencies[new_pair] = byte_pair_frequencies.get(new_pair, 0) + freq
                                # update indices
                                token_indices.setdefault(new_pair, {})[pretoken_idx] = freq
                                                
                            # Update token indices to remove invalid pairs
                            self._prune_obsolete_indices(
                                encoded_pretokens,
                                pretoken_idx,
                                byte_idx,
                                token_indices
                            )
                                                        
                            seq = encoded_pretokens[pretoken_idx]
                            seq[byte_idx] = new_token
                            seq.pop(byte_idx + 1)
                            
                        else:
                            byte_idx += 1
                
                pbar.update(1)
            
            pbar.close()
        
        return self.vocabulary, merges