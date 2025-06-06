import pickle
from tokenizer import Tokenizer
import json
import time
import multiprocessing
import os



def main():
    # Configuration variables (change these to switch datasets and modes)
    dataset = "owt"  # "owt" or "tinystories"
    data_mode = "train"  # "train" or "valid"
    
    # Set paths based on configuration
    base_data_dir = "./data"
    input_file_mapping = {
        "owt": {
            "train": f"{base_data_dir}/owt_train.txt",
            "valid": f"{base_data_dir}/owt_valid.txt"
        },
        "tinystories": {
            "train": f"{base_data_dir}/TinyStoriesV2-GPT4-train.txt",
            "valid": f"{base_data_dir}/TinyStoriesV2-GPT4-valid.txt"
        }
    }
    
    # Get input path based on dataset and mode
    input_path = input_file_mapping[dataset][data_mode]
    
    # Output file paths
    output_base = f"{base_data_dir}/{dataset}"
    vocab_pkl = f"{output_base}_vocab.pkl"
    merges_pkl = f"{output_base}_merges.pkl"

    T = Tokenizer(vocab = None, merges = None, special_tokens=None)

    special_tokens = ["<|endoftext|>"]
    vocab_size = None
    if dataset == "owt":
        vocab_size = 32000
    else:
        vocab_size = 10000
    

    print(f"Starting BPE training on {dataset} {data_mode} dataset...")
    start_time = time.time()
    vocab, merges = T.train_bpe(input_path=input_path, vocab_size=vocab_size, special_tokens=special_tokens)
    end_time = time.time()

    with open(vocab_pkl, "wb+") as f:
        pickle.dump(vocab, f)
    with open(merges_pkl, "wb+") as f:
        pickle.dump(merges, f)

    # Ensure directory exists before saving files
    os.makedirs(base_data_dir, exist_ok=True)
    
    # Print some statistics
    print(f"Trained BPE in {end_time - start_time:.2f} seconds")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")

def analysis(json_path):
    with open(json_path, 'r') as f:
        vocab = json.load(f)
    
    # Find the longest value in the vocabulary
    longest_token = max(vocab.values(), key=lambda x: len(x.encode("utf-8")))
    longest_id = [k for k, v in vocab.items() if v == longest_token][0]
    
    print(f"Analyzing vocabulary from {json_path}")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Longest token: '{longest_token}' (length: {len(longest_token)})")
    print(f"Token ID: {longest_id}")

if __name__ == '__main__':
    main()
    # To analyze a specific vocabulary file:
    #analysis("data/owt_vocab_train_copy.json")
