from tqdm import tqdm
from tokenizer import Tokenizer
import time
import random
import pickle
import numpy as np
import os

# 1) Paths to your vocab & merges
TS_VOCAB = "data/tinystories_vocab.pkl"
TS_MERGES = "data/tinystories_merges.pkl"
tiny_stories_path = "data/TinyStoriesV2-GPT4-train.txt"

OWT_VOCAB = "data/owt_vocab.pkl"
OWT_MERGES = "data/owt_merges.pkl"
owt_path = "data/owt_train.txt"

def bytes_per_token_and_throughput(tok, docs):
    ratios = []
    total_bytes = 0
    start_time = time.time()

    for doc in tqdm(docs, desc="Processing sample documents"):
        byte_doc = doc.encode("utf-8")
        ids = tok.encode(doc)
        if len(ids) > 0:
            ratios.append(len(byte_doc) / len(ids))
            total_bytes += len(byte_doc)

    elapsed = time.time() - start_time
    throughput = total_bytes / elapsed if elapsed > 0 else 0
    return ratios, total_bytes, elapsed, throughput

def full_file_throughput(tok, text):
    byte_text = text.encode("utf-8")
    start_time = time.time()
    ids = tok.encode(text)
    elapsed = time.time() - start_time
    throughput = len(byte_text) / elapsed if elapsed > 0 else 0
    return len(byte_text), elapsed, throughput, ids

def main():
    # load tokenizer
    ts_tok = Tokenizer.from_files(TS_VOCAB, TS_MERGES, special_tokens=["<|endoftext|>"])
    print("Finished loading tokenizer")

    # read entire file once
    with open(tiny_stories_path, "r", encoding="utf-8") as f:
        text = f.read()

    # split into docs
    ts_docs = [doc for doc in text.split("<|endoftext|>") if doc.strip()]
    print(f"Loaded {len(ts_docs)} documents from TinyStories")

    # ----- Full-file benchmark -----
    full_bytes, full_time, full_tp, ids = full_file_throughput(ts_tok, text)
    print("\n=== Full-file Benchmark ===")
    print(f"File size:       {full_bytes} bytes")
    print(f"Elapsed:         {full_time:.2f}s")
    print(f"Throughput:      {full_tp:.2f} B/s")

    ids_array = np.array(ids, dtype=np.uint16)
    print(f"Token count:     {ids_array.size}")
    print(f"Array dtype:     {ids_array.dtype}")

    # ensure output directory exists
    out_dir = "data"
    os.makedirs(out_dir, exist_ok=True)

    # Save tokenized ids to pickle file
    output_path = os.path.join(out_dir, "tinystories_tokenized_ids.npy")
    np.save(output_path, ids_array)
    print(f"Successfully saved token IDs to {output_path}")

if __name__ == "__main__":
    main()
