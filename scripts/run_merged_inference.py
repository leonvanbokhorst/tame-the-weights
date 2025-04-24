"""
Runs inference using a fully merged model.

Usage:
  python scripts/run_merged_inference.py --model_path <path_to_merged_model>
"""

import argparse
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# Optional, for better input experience
try:
    import readline
except ImportError:
    pass  # readline not available, input will be more basic


def main(args):
    print("Starting inference setup...")
    print(f"Model Path: {args.model_path}")
    print(f"Using Quantization: {args.use_quantization}")

    # --- 1. Configure Quantization (Optional) ---
    bnb_config = None
    if args.use_quantization:
        try:
            from bitsandbytes.cextension import COMPILED_WITH_CUDA

            if not COMPILED_WITH_CUDA:
                print(
                    "Warning: bitsandbytes installed without CUDA support. Disabling quantization."
                )
                args.use_quantization = False
            else:
                print("Configuring BitsAndBytes for inference...")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
        except ImportError:
            print("bitsandbytes not available. Disabling quantization.")
            args.use_quantization = False

    # --- 2. Load Merged Model & Tokenizer ---
    print(f"Loading merged model from ({args.model_path})...")
    # Set device_map="auto" for automatic multi-GPU handling (if available)
    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config

    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    print("Model loaded.")

    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, use_fast=False, legacy=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set pad_token to eos_token")
    print("Tokenizer loaded.")

    model.eval()  # Set the model to evaluation mode

    print("\nModel ready for inference!")
    print("Type 'quit' or 'exit' to stop.")

    # --- 3. Inference Loop ---
    while True:
        try:
            prompt = input("\nEnter prompt: ")
            if prompt.lower() in ["quit", "exit"]:
                break

            # Basic prompt formatting (adjust as needed based on model expectations)
            # For instruction-tuned models like Phi-3 instruct, follow their template
            # Example for Phi-3 Instruct:
            messages = [
                {"role": "user", "content": prompt},
            ]
            inputs = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(model.device)

            print("Generating response...")
            with torch.no_grad():  # Disable gradient calculation for inference
                # Adjust generation parameters as needed (max_new_tokens, temperature, etc.)
                outputs = model.generate(
                    input_ids=inputs,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,  # Important for open-ended generation
                    # use_flash_attention_2=True,
                    # flash_attention_kwargs={"return_softmax": False, "softmax_scale": None},
                )

            # Decode the generated tokens
            # Ensure we decode only the newly generated part, excluding the input prompt
            response_ids = outputs[0][inputs.shape[-1] :]
            response = tokenizer.decode(response_ids, skip_special_tokens=True)

            print("\n--- Model Response ---")
            print(response)
            print("----------------------")

        except EOFError:
            break  # Exit loop if input stream is closed
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            # Optionally, break or continue based on error type
            continue

    print("\nExiting inference script.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference with a fully merged model."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the merged model directory.",
    )
    parser.add_argument(
        "--use_quantization",
        action="store_true",
        help="Use 4-bit quantization for inference (requires bitsandbytes with CUDA).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=200,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for text generation (higher = more creative, lower = more deterministic).",
    )

    args = parser.parse_args()

    # Basic validation (check if model path exists)
    import os

    if not os.path.isdir(args.model_path):
        print(
            f"Error: Model path not found or is not a directory: {args.model_path}"
        )
        exit(1)

    main(args) 