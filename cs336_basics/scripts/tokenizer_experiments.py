from tqdm import tqdm
from tokenizer import Tokenizer
import time
import random
import pickle

# 1) Paths to your vocab & merges
TS_VOCAB = "data/tinystories_vocab.pkl"
TS_MERGES = "data/tinystories_merges.pkl"
tiny_stories_path = "data/TinyStoriesV2-GPT4-train.txt"

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

    # ----- Sample-based benchmark -----
    ts_sample = random.sample(ts_docs, 10) if len(ts_docs) > 10 else ts_docs
    ratios, samp_bytes, samp_time, samp_tp = bytes_per_token_and_throughput(ts_tok, ts_sample)

    print("\n=== Sample Benchmark (10 docs) ===")
    print(f"Avg bytes/token: {sum(ratios)/len(ratios):.2f}")
    print(f"Processed {len(ratios)} docs in {samp_time:.2f}s → {samp_tp:.2f} B/s")

    # ----- Full-file benchmark -----
    full_bytes, full_time, full_tp, ids = full_file_throughput(ts_tok, text)
    print("\n=== Full-file Benchmark ===")
    print(f"File size:       {full_bytes} bytes")
    print(f"Elapsed:         {full_time:.2f}s")
    print(f"Throughput:      {full_tp:.2f} B/s")
    # Save tokenized ids to pickle file
    output_path = "data/tinystories_tokenized_ids.pkl"
    print(f"Saving tokenized ids to {output_path}")
    with open(output_path, "wb") as f:
        pickle.dump(ids, f)
    print(f"Successfully saved {len(ids)} tokens")

if __name__ == "__main__":
    main()
