#!/usr/bin/env python3
"""Generate text using a trained Transformer model"""

import argparse
import json
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, '.')

from Transform import Module as Mo
from cs336_basics.Tokenizer.BPE_tokenizer import Tokenizer


def load_tokenizer(vocab_path, merges_path, special_tokens=None):
    """Load tokenizer with correct merges parsing"""
    with open(vocab_path, "r") as vf:
        vocab_data = json.load(vf)
        vocab = {int(i): bytes(v, "latin1") for v, i in vocab_data.items()}

    merges = []
    with open(merges_path, "r") as mf:
        for line in mf:
            line = line.rstrip('\n')
            if line and not line.startswith("#"):
                parts = line.rsplit(' ', 1)
                if len(parts) == 2:
                    merges.append((bytes(parts[0], "latin1"), bytes(parts[1], "latin1")))

    return Tokenizer(vocab, merges, special_tokens)


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 50,
    device: str = "cuda",
):
    """
    Generate text using the model.

    Args:
        model: Trained Transformer model
        tokenizer: BPE tokenizer
        prompt: Starting text
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_p: Nucleus sampling threshold
        top_k: Top-k sampling threshold
        device: Device to run on

    Returns:
        Generated text string
    """
    model.eval()

    # Encode prompt
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    # Get special token id for stopping
    eos_token = "<|endoftext|>"
    eos_id = None
    if eos_token.encode("utf-8") in tokenizer.byte_to_id:
        eos_id = tokenizer.byte_to_id[eos_token.encode("utf-8")]

    generated = input_ids

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Truncate to context length if needed
            context_length = model.context_length if hasattr(model, 'context_length') else 256
            input_context = generated[:, -context_length:]

            # Forward pass
            logits = model(input_context)

            # Get logits for next token (last position)
            next_token_logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                top_k_values, _ = torch.topk(next_token_logits, top_k)
                threshold = top_k_values[:, -1].unsqueeze(-1)
                next_token_logits = torch.where(
                    next_token_logits < threshold,
                    torch.full_like(next_token_logits, float('-inf')),
                    next_token_logits
                )

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits = next_token_logits.masked_fill(indices_to_remove, float('-inf'))

            # Sample from distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to generated
            generated = torch.cat([generated, next_token], dim=1)

            # Stop if EOS token
            if eos_id is not None and next_token.item() == eos_id:
                break

    # Decode generated tokens
    output_ids = generated[0].tolist()
    output_text = tokenizer.decode(output_ids)

    return output_text


def main():
    parser = argparse.ArgumentParser(description='Generate text with trained model')

    # Model parameters (should match training)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--theta", type=float, default=10000.0)

    # Generation parameters
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--vocab_path", type=str, default="../data/tinystories_vocab.json")
    parser.add_argument("--merges_path", type=str, default="../data/tinystories_merges.txt")
    parser.add_argument("--prompt", type=str, default="Once upon a time,")
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    print("=" * 50)
    print("Text Generation")
    print("=" * 50)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = load_tokenizer(args.vocab_path, args.merges_path, ["<|endoftext|>"])
    print(f"Vocab size: {len(tokenizer.byte_vocab)}")

    # Create model
    print("Creating model...")
    device = args.device if torch.cuda.is_available() else "cpu"
    model = Mo.Transformer(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        d_ff=args.d_ff,
        context_length=args.context_length,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        rope_theta=args.theta,
        device=device
    )

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    print(f"Loaded model from step {checkpoint.get('iteration', 'unknown')}")
    print()

    # Generate
    print(f"Prompt: {args.prompt}")
    print(f"Temperature: {args.temperature}, Top-p: {args.top_p}, Top-k: {args.top_k}")
    print("-" * 50)

    output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        device=device,
    )

    print("Generated text:")
    print("-" * 50)
    print(output)
    print("-" * 50)

    # Count tokens
    output_tokens = tokenizer.encode(output)
    print(f"Total tokens: {len(output_tokens)}")


if __name__ == "__main__":
    main()
