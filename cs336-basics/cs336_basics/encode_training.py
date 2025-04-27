#!/usr/bin/env python3
import os
import json
import numpy as np
from tokenizer import Tokenizer

BASE_DIR = "./data"
#DATASETS = ["owt", "tinystories"]
DATASETS = ["owt"]
# SPLITS   = ["train", "valid"]
SPLITS = ["valid"]
SPECIAL_TOKENS = ["<|endoftext|>"]

def main():
    os.makedirs(BASE_DIR, exist_ok=True)

    for ds in DATASETS:
        # paths to your saved vocab and merges
        vocab_path  = os.path.join(BASE_DIR, f"{ds}_vocab.pkl")
        merges_path = os.path.join(BASE_DIR, f"{ds}_merges.pkl")

        # load tokenizer
        tok = Tokenizer.from_files(vocab_path, merges_path, special_tokens=SPECIAL_TOKENS)

        for split in SPLITS:
            txt_in  = os.path.join(BASE_DIR, f"{ds}_{split}.txt")
            npy_out = os.path.join(BASE_DIR, f"{ds}_{split}_ids.npy")

            print(f"\n→ Encoding {ds} – {split}…")
            ids = []
            with open(txt_in, "r", encoding="utf-8") as f:
                # line‑by‑line (or chunk‑by‑chunk) streaming
                for tid in tok.encode_iterable(f):
                    ids.append(tid)

            arr = np.array(ids, dtype=np.uint16)
            np.save(npy_out, arr)
            print(f"   Wrote {arr.shape[0]:,} token IDs to `{npy_out}`")

if __name__ == "__main__":
    main()
