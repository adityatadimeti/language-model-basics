from tokenizer import Tokenizer
import json
import time
import multiprocessing

def main():
    T = Tokenizer()

    tiny_stories_train_path = "./data/TinyStoriesV2-GPT4-train.txt"
    #tiny_stories_train_path = "./sliced_files/slice_10percent.txt"
    #tiny_stories_train_path = "./data/TinyStoriesV2-GPT4-valid.txt"
    special_tokens = ["<|endoftext|>"]
    vocab_size = 10000

    print("Starting BPE training...")
    start_time = time.time()
    vocab, merges = T.train_bpe(input_path=tiny_stories_train_path, vocab_size=vocab_size, special_tokens=special_tokens)
    end_time = time.time()

    # Ensure directory exists before saving files
    import os
    os.makedirs("../data", exist_ok=True)
    
    # Save vocabulary as JSON file
    with open("./data/tinystories_vocab.json", "w") as f:
        # Convert bytes to strings for JSON serialization
        serializable_vocab = {k: v.decode('utf-8', errors='replace') if isinstance(v, bytes) else v 
                             for k, v in vocab.items()}
        json.dump(serializable_vocab, f, indent=2)
    print(f"Vocabulary saved to ./data/tinystories_vocab.json")

    # Save merges as a text file
    with open("./data/tinystories_merges.txt", "w") as f:
        for merge in merges:
            # Convert bytes to strings for writing to file
            first = merge[0].decode('utf-8', errors='replace') if isinstance(merge[0], bytes) else merge[0]
            second = merge[1].decode('utf-8', errors='replace') if isinstance(merge[1], bytes) else merge[1]
            f.write(f"{first} {second}\n")
    print(f"Merges saved to ./data/tinystories_merges.txt")

    # Print some statistics
    print(f"Trained BPE in {end_time - start_time:.2f} seconds")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")

if __name__ == '__main__':
    # This is the crucial part that fixes the multiprocessing issue
    multiprocessing.freeze_support()  # For Windows support
    main()