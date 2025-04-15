#!/usr/bin/env python3
import os
import argparse
from pathlib import Path

def create_file_slices(input_file="data/TinyStoriesV2-GPT4-train.txt", 
                      output_dir="sliced_files", 
                      num_slices=10):
    """
    Create progressively larger slices of the input file.
    Each slice contains the first X% of the original file.
    
    Args:
        input_file: Path to the original file
        output_dir: Directory to write the slices to
        num_slices: Number of slices to create (default: 10 slices, from 10% to 100%)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get file size
    file_size = os.path.getsize(input_file)
    print(f"Original file size: {file_size / (1024*1024):.2f} MB")
    
    # Read the input file
    with open(input_file, 'rb') as f:
        content = f.read()
    
    # Create slices
    for i in range(1, num_slices + 1):
        percentage = i * (100 / num_slices)
        slice_size = int((i / num_slices) * file_size)
        
        # Create the slice file
        output_file = os.path.join(output_dir, f"slice_{int(percentage)}percent.txt")
        with open(output_file, 'wb') as f:
            f.write(content[:slice_size])
        
        print(f"Created {output_file}: {slice_size / (1024*1024):.2f} MB ({percentage:.0f}% of original)")

if __name__ == "__main__":
    # Default to your specific file
    parser = argparse.ArgumentParser(description="Create slices of TinyStoriesV2-GPT4-train.txt file for testing")
    parser.add_argument("--input-file", default="data/TinyStoriesV2-GPT4-train.txt", 
                        help="Path to the input file (default: data/TinyStoriesV2-GPT4-train.txt)")
    parser.add_argument("--output-dir", default="sliced_files", 
                        help="Directory to write the slices to (default: sliced_files)")
    parser.add_argument("--num-slices", type=int, default=10, 
                        help="Number of slices to create (default: 10)")
    
    args = parser.parse_args()
    
    create_file_slices(args.input_file, args.output_dir, args.num_slices)
    print(f"Done! Created {args.num_slices} slices in {args.output_dir}")