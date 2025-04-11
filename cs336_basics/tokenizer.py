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

        token_bytes_sequence: List[bytes] = []

        # First, before running pre-tokenization, we have to split on all special tokens. 
        segment_splits = re.split(pattern, chunk)
        local_counts = defaultdict(int) # define local dictionary to avoid inter-process communication / locking
        for segment in segment_splits:
            # Pretokenize each segment
            iter = PAT_COMPILED.finditer(segment) # iter now contains an iterable of matches within each segment
            for match in iter:
                utf_match = tuple(match.group(0).encode("utf-8"))
                local_counts[utf_match] += 1
                token_bytes_sequence.append(utf_match)
        
        return local_counts, token_bytes_sequence # Also returning token_bytes to maintain order


    def train_bpe(self, input_path: str, vocab_size: int, special_tokens: list[str]) -> Tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        merges = [] # tuple of bytes


        self.vocabulary = {i: bytes([i]) for i in range(256)}
        self.token_ID = 256

        # NOTE: later have to check for deduplication?
        for token_str in special_tokens:
            token_bytes = token_str.encode("utf-8")
            self.vocabulary[self.token_ID] = token_bytes
            self.token_ID += 1


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
                pretokenized_results = pool.map(self.pretokenize, args_list)

            aggregated_list = [] # Ordered list of pretokenized inputs
            for local_count, token_sequence in pretokenized_results:
                aggregated_list.append(token_sequence)
                for item, count in local_count.items():
                    self.pretokenized_counts[item] += count

            # Still a bit unclear as to why we use pretokenized_counts

            while (len(self.vocabulary) < vocab_size): # Keep merging until we hit the desired vocab_size
                pair_counts = Counter()
                for chunk in aggregated_list:
                    for i in range(len(chunk) - 1):
                        # Each item is a tuple of (token, count)
                        token1 = chunk[i]
                        token2 = chunk[i + 1]
                        pair_counts[(token1, token2)] += 1
                
                # Identify pair to merge
                best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
                
                token1, token2 = best_pair
                new_token_tuple = token1 + token2 # Combining the tuples

                # Now merge

                new_aggregated_list = []
                for chunk_seq in aggregated_list:
                    new_chunk_seq = []
                    i = 0
                    while i < len(chunk_seq):
                        # Check if pair at index i matches best_pair
                        if i < len(chunk_seq) - 1 and (chunk_seq[i], chunk_seq[i+1]) == best_pair:
                            new_chunk_seq.append(new_token_tuple)
                            i += 2 # Skip both original tokens
                        else:
                            new_chunk_seq.append(chunk_seq[i])
                            i += 1
                    if new_chunk_seq:
                        new_aggregated_list.append(new_chunk_seq)

                # Update the main list for the next iteration
                aggregated_list = new_aggregated_list

                # Finally, add to self.vocabulary and merges
                # Convert the merged tokens into bytes and add to vocabulary
                merged_token = bytes(token1) + bytes(token2)  # new_key is already token1 + token2
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

    import tempfile

    # This is your mini dataset sample.
    sample_text = """u don't have to be scared of the loud dog, I'll protect you. The mole felt so safe with the little girl. She was very kind and the mole soon came to trust her. He leaned against her and she kept him safe. The mole had found his best friend.
    <|endoftext|>
    Once upon a time, in a warm and sunny place, there was a big pit. A little boy named Tom liked to play near the pit. One day, Tom lost his red ball. He was very sad.
    Tom asked his friend, Sam, to help him search for the ball. They looked high and low, but they could not find the ball. Tom said, "I think my ball fell into the pit."
    Sam and Tom went close to the pit. They were scared, but they wanted to find the red ball. They looked into the pit, but it was too dark to see. Tom said, "We must go in and search for my ball."
    They went into the pit to search. It was dark and scary. They could not find the ball. They tried to get out, but the pit was too deep. Tom and Sam were stuck in the pit. They called for help, but no one could hear them. They were sad and scared, and they never got out of the pit.
    <|endoftext|>

    Tom and Lily were playing with their toys in the living room. They liked to build towers and bridges with their blocks and cars. Tom was very proud of his tall tower. He wanted to make it even taller, so he reached for more blocks.
    "Tom, can I have some blocks too?" Lily asked. She wanted to make a bridge for her cars.
    "No, these are mine. Go find your own," Tom said. He did not want to share with his sister. He pulled the blocks closer to him.
    Lily felt sad and angry. She did not think Tom was being nice. She looked at his tower and had an idea. She decided to pull one of the blocks at the bottom of the tower.
    Suddenly, the tower fell down with a loud crash. All the blocks and cars scattered on the floor. Tom and Lily were shocked. They felt the floor shake and heard a rumble. It was an earthquake!
    "Mommy! Daddy!" they cried. They were scared and ran to their parents, who were in the kitchen.
    "Are you okay, kids?" Mommy asked. She hugged them and checked if they were hurt.
    "We're okay, Mommy. But our toys are broken," Lily said.
    "I'm sorry, Lily. But toys are not important. You are important. We are safe and together. That's what matters," Mommy said.
    Tom felt sorry for what he did. He realized he was selfish and mean to his sister. He saw how scared she was during the earthquake. He wanted to make her happy.
    "Lily, I'm sorry I did not share with you. You can have all the blocks you want. I love you, sister," Tom said.
    Lily smiled and hugged him. She forgave him and thanked him. She loved him too.
    They went back to the living room and cleaned up their toys. They decided to build something together. They made a big house with a garden and a fence. They put their cars and dolls inside. They were happy and proud of their work.
    Mommy and Daddy came to see their house. They praised them and gave them a treat. It was a lemon cake. It was sour, but they liked it. They learned that sharing is caring, and that family is sweet.
    <|endoftext|>
    """

    # Create a temporary file and write the sample_text into it.
    with tempfile.NamedTemporaryFile("w+", delete=False) as temp_file:
        temp_file.write(sample_text)
        temp_file.flush()
        temp_path = temp_file.name

    # Instantiate your Tokenizer and run the training.
    tokenizer = Tokenizer()
    vocab, merges = tokenizer.train_bpe(
        input_path=temp_path, 
        vocab_size=300, 
        special_tokens=["<|endoftext|>"]
    )

    print("Final vocabulary size:", len(vocab))
    print("First 10 merges:", merges[:10])


    # T = Tokenizer()
    # stories_valid_path = "data/TinyStoriesV2-GPT4-valid.txt"
    # stories_train_path = "data/TinyStoriesV2-GPT4-train.txt"
    # T.train_bpe(stories_valid_path, 300, ["<|endoftext|>".encode("utf-8")])  