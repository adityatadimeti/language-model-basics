from tokenizer import Tokenizer
import json
import time
import multiprocessing
import os



def main():
    # Configuration variables (change these to switch datasets and modes)
    dataset = "tinystories"  # "owt" or "tinystories"
    data_mode = "valid"  # "train" or "valid"
    
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
    vocab_json = f"{output_base}_vocab.json"
    merges_txt = f"{output_base}_merges.txt"
    
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

    # Ensure directory exists before saving files
    os.makedirs(base_data_dir, exist_ok=True)
    
    # Save vocabulary as JSON file
    with open(vocab_json, "w") as f:
        # Convert bytes to strings for JSON serialization
        serializable_vocab = {k: v.decode('utf-8', errors='replace') if isinstance(v, bytes) else v 
                             for k, v in vocab.items()}
        json.dump(serializable_vocab, f, indent=2)
    print(f"Vocabulary saved to {vocab_json}")

    # Save merges as a text file
    with open(merges_txt, "w") as f:
        for merge in merges:
            # Convert bytes to strings for writing to file
            first = merge[0].decode('utf-8', errors='replace') if isinstance(merge[0], bytes) else merge[0]
            second = merge[1].decode('utf-8', errors='replace') if isinstance(merge[1], bytes) else merge[1]
            f.write(f"{first} {second}\n")
    print(f"Merges saved to {merges_txt}")

    # Print some statistics
    print(f"Trained BPE in {end_time - start_time:.2f} seconds")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")

def analysis(json_path):
    with open(json_path, 'r') as f:
        vocab = json.load(f)
    
    # Find the longest value in the vocabulary
    longest_token = max(vocab.values(), key=len)
    longest_id = [k for k, v in vocab.items() if v == longest_token][0]
    
    print(f"Analyzing vocabulary from {json_path}")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Longest token: '{longest_token}' (length: {len(longest_token)})")
    print(f"Token ID: {longest_id}")

if __name__ == '__main__':
    main()
    # To analyze a specific vocabulary file:
    # analysis("data/owt_vocab.json")
