from tqdm import tqdm
from tokenizer import Tokenizer
import json
import time
import multiprocessing
import os


import random
from tokenizer import Tokenizer

# 1) Paths to your vocab & merges
TS_VOCAB = "data/tinystories_vocab.json"
TS_MERGES = "data/tinystories_merges.txt"


OWT_VOCAB = "data/owt_vocab_train_copy.json"
OWT_MERGES = "data/owt_merges.txt"

GPT2_VOCAB = "tests/fixtures/gpt2_vocab.json"
GPT2_MERGES = "tests/fixtures/gpt2_merges.txt"

TS_VOCAB_PKL = "data/tinystories_vocab.pkl"
TS_MERGES_PKL = "data/tinystories_merges.pkl"

TS_VOCAB_VALID = "data/tinystories_vocab.json"
TS_MERGES_VALID = "data/tinystories_merges.txt"

#tiny_stories_path = "data/TinyStoriesV2-GPT4-valid copy.txt"
tiny_stories_path = "data/TinyStoriesV2-GPT4-train.txt"

def main():

    ts_tok = Tokenizer.from_files(TS_VOCAB_VALID,  TS_MERGES_VALID,  special_tokens=["<|endoftext|>"])

    print("finished loading tokenizers")
    # 3) Read & split your corpora into “documents”

    print("opened first files")
       
    with open(tiny_stories_path, "r", encoding="utf-8") as f:
        # Read all text from the file
        f.seek(0, 2)  # Seek to end
        file_size = f.tell()
        one_percent = int(file_size * 0.001)
        f.seek(0)
        text = f.read(one_percent)
        
        # Split text into documents (assuming each story is separated by double newlines)
        # You may need to adjust this depending on your file format
        ts_sample = text.split("<|endoftext|>")
        
        print(f"Loaded {len(ts_sample)} documents from TinyStories")

        # Randomly sample 10 documents from ts_sample
        # Only sample non-empty documents
        ts_sample = [doc for doc in ts_sample if doc.strip()]
        if len(ts_sample) > 10:
            ts_sample = random.sample(ts_sample, 1)

        def bytes_per_token(tok, docs):
            ratios = []
            for doc in tqdm(docs, desc="Processing documents"):
                byte_doc = doc.encode("utf-8")
                ids = tok.encode(doc)
                if len(ids) > 0:  # Avoid division by zero
                    ratios.append(len(byte_doc) / len(ids))
            return ratios

        # Compute
        ts_ratios = bytes_per_token(ts_tok, ts_sample)

        # Display
        print("TinyStories bytes/token (per doc):", ts_ratios[:5], "...")  # Show just first few
        print("TinyStories avg bytes/token:      ", sum(ts_ratios)/len(ts_ratios))
        print(f"Total documents processed: {len(ts_ratios)}")

if __name__ == "__main__":
    main()