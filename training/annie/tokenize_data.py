#!/usr/bin/env python3
"""tokenize_data.py — Tokenize Oscar conversation logs for Annie training.

Usage:
    python3 tokenize_data.py --input conversations.jsonl --output annie_train_data.bin

Input JSONL format (one JSON object per line):
    {"query": "Hey Oscar!", "response": "Hello! How can I help?", "sentiment": 8}

Output: binary file of uint32 token IDs with Qwen2.5 chat template applied.
Only high-sentiment (7-10) conversations are included for SFT.

Chat template:
    <|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>
"""
import argparse
import json
import struct
import numpy as np
from pathlib import Path


def get_tokenizer(model_name="Qwen/Qwen2.5-3B-Instruct"):
    """Load Qwen2.5 tokenizer. Tries tiktoken first, falls back to HF."""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print(f"  Loaded HuggingFace tokenizer: {model_name}")
        print(f"  Vocab size: {tokenizer.vocab_size}")
        return tokenizer
    except Exception as e:
        print(f"  HF tokenizer failed: {e}")
        print("  Install: pip install transformers")
        raise


def format_conversation(query, response):
    """Apply Qwen2.5 chat template."""
    return (
        f"<|im_start|>user\n{query}<|im_end|>\n"
        f"<|im_start|>assistant\n{response}<|im_end|>"
    )


def main():
    parser = argparse.ArgumentParser(description='Tokenize conversations for Annie')
    parser.add_argument('--input', required=True, help='Input JSONL file')
    parser.add_argument('--output', default='annie_train_data.bin', help='Output binary file')
    parser.add_argument('--model', default='Qwen/Qwen2.5-3B-Instruct', help='Tokenizer model')
    parser.add_argument('--min-sentiment', type=int, default=7, help='Minimum sentiment score')
    parser.add_argument('--max-seq-len', type=int, default=256, help='Maximum sequence length')
    parser.add_argument('--verify', action='store_true', help='Verify round-trip encode/decode')
    args = parser.parse_args()

    print("Loading tokenizer...")
    tokenizer = get_tokenizer(args.model)

    print(f"\nProcessing {args.input}...")
    all_tokens = []
    total_convs = 0
    filtered = 0
    truncated = 0

    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            total_convs += 1
            sentiment = obj.get('sentiment', 5)

            # Filter by sentiment
            if sentiment < args.min_sentiment:
                filtered += 1
                continue

            query = obj.get('query', '')
            response = obj.get('response', '')
            if not query or not response:
                filtered += 1
                continue

            # Format and tokenize
            text = format_conversation(query, response)
            tokens = tokenizer.encode(text)

            # Truncate if needed
            if len(tokens) > args.max_seq_len:
                tokens = tokens[:args.max_seq_len]
                truncated += 1

            all_tokens.extend(tokens)

    print(f"  Total conversations: {total_convs}")
    print(f"  Filtered (low sentiment/empty): {filtered}")
    print(f"  Truncated: {truncated}")
    print(f"  Total tokens: {len(all_tokens)}")

    if len(all_tokens) == 0:
        print("ERROR: No tokens generated!")
        return

    # Write as uint32
    token_array = np.array(all_tokens, dtype=np.uint32)
    token_array.tofile(args.output)
    file_size = token_array.nbytes
    print(f"\nWritten to {args.output}: {file_size / 1e6:.1f} MB ({len(all_tokens)} tokens)")

    # Verify round-trip
    if args.verify:
        print("\nVerification: round-trip encode→decode...")
        test_text = format_conversation("Hey Oscar!", "Hello! How can I help you today?")
        test_tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(test_tokens)
        print(f"  Original:  {test_text[:80]}...")
        print(f"  Decoded:   {decoded[:80]}...")
        print(f"  Tokens:    {test_tokens[:20]}... ({len(test_tokens)} total)")

        # Verify binary
        loaded = np.fromfile(args.output, dtype=np.uint32)
        assert len(loaded) == len(all_tokens), f"Length mismatch: {len(loaded)} vs {len(all_tokens)}"
        assert np.array_equal(loaded, token_array), "Data mismatch!"
        print("  Binary verification passed!")

        # Show vocab range
        min_tok, max_tok = token_array.min(), token_array.max()
        print(f"  Token range: [{min_tok}, {max_tok}] (vocab_size={tokenizer.vocab_size})")


if __name__ == '__main__':
    main()
