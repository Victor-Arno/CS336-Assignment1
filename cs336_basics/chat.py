#!/usr/bin/env python3
"""Interactive text generation with trained Transformer model"""

import json
import re
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
    stream: bool = True,
):
    """Generate text with optional streaming output"""
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
    generated_text = prompt

    if stream:
        print(prompt, end="", flush=True)

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

            # Decode new token
            new_token_text = tokenizer.decode([next_token.item()])

            if stream:
                print(new_token_text, end="", flush=True)

            generated_text += new_token_text

            # Stop if EOS token
            if eos_id is not None and next_token.item() == eos_id:
                break

    if stream:
        print()  # New line at end

    return generated_text


def parse_command(user_input, current_settings):
    """Parse user input for commands and settings"""
    settings = current_settings.copy()
    prompt = user_input

    # Check for /set commands
    # Format: /temp 0.8 or /top_p 0.9 or /top_k 50 or /max 256
    patterns = {
        r'/temp\s+([\d.]+)': 'temperature',
        r'/temperature\s+([\d.]+)': 'temperature',
        r'/top_p\s+([\d.]+)': 'top_p',
        r'/topp\s+([\d.]+)': 'top_p',
        r'/top_k\s+(\d+)': 'top_k',
        r'/topk\s+(\d+)': 'top_k',
        r'/max\s+(\d+)': 'max_tokens',
    }

    for pattern, key in patterns.items():
        match = re.search(pattern, user_input, re.IGNORECASE)
        if match:
            value = float(match.group(1)) if '.' in match.group(1) or key in ['temperature', 'top_p'] else int(match.group(1))
            settings[key] = value
            prompt = re.sub(pattern, '', prompt, flags=re.IGNORECASE).strip()

    return prompt, settings


def print_help():
    print("""
╔══════════════════════════════════════════════════════════════╗
║                    Commands                                   ║
╠══════════════════════════════════════════════════════════════╣
║  /temp <value>    - Set temperature (e.g., /temp 0.8)        ║
║  /top_p <value>   - Set top-p (e.g., /top_p 0.9)             ║
║  /top_k <value>   - Set top-k (e.g., /top_k 50)              ║
║  /max <value>     - Set max tokens (e.g., /max 256)          ║
║  /settings        - Show current settings                     ║
║  /help            - Show this help                            ║
║  /quit or /exit   - Exit the program                          ║
╠══════════════════════════════════════════════════════════════╣
║  Example: /temp 0.7 /top_p 0.85 Once upon a time,            ║
╚══════════════════════════════════════════════════════════════╝
""")


def main():
    # Default settings
    settings = {
        'temperature': 0.8,
        'top_p': 0.9,
        'top_k': 50,
        'max_tokens': 256,
    }

    # Paths (adjust as needed)
    checkpoint_path = "../checkpoints/tinystories_best.pt"
    vocab_path = "../data/tinystories_vocab.json"
    merges_path = "../data/tinystories_merges.txt"

    # Model config
    config = {
        'vocab_size': 10000,
        'd_model': 512,
        'num_layers': 4,
        'num_heads': 16,
        'd_ff': 1344,
        'context_length': 256,
        'theta': 10000.0,
    }

    print("=" * 60)
    print("     TinyStories Interactive Text Generation")
    print("=" * 60)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = load_tokenizer(vocab_path, merges_path, ["<|endoftext|>"])

    # Create model
    print("Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Mo.Transformer(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        d_ff=config['d_ff'],
        context_length=config['context_length'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        rope_theta=config['theta'],
        device=device
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    print(f"Model loaded from step {checkpoint.get('iteration', 'unknown')}")
    print(f"Device: {device}")
    print()
    print_help()

    # Interactive loop
    while True:
        try:
            print(f"\n[temp={settings['temperature']}, top_p={settings['top_p']}, top_k={settings['top_k']}, max={settings['max_tokens']}]")
            user_input = input(">>> ").strip()

            if not user_input:
                continue

            # Handle special commands
            if user_input.lower() in ['/quit', '/exit', '/q']:
                print("Goodbye!")
                break

            if user_input.lower() == '/help':
                print_help()
                continue

            if user_input.lower() == '/settings':
                print(f"Current settings: {settings}")
                continue

            # Parse commands and get prompt
            prompt, settings = parse_command(user_input, settings)

            if not prompt:
                print(f"Settings updated: {settings}")
                continue

            # Generate
            print("-" * 60)
            generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=settings['max_tokens'],
                temperature=settings['temperature'],
                top_p=settings['top_p'],
                top_k=settings['top_k'],
                device=device,
                stream=True,
            )
            print("-" * 60)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
