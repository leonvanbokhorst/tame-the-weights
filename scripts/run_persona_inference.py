"""
Runs inference using a base model merged with a trained persona adapter.

Usage:
  python scripts/run_persona_inference.py --model_id <base_model_id> \
                                          --adapter_path <path_to_adapter_dir>
"""

import argparse
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel

# Optional, for better input experience
try:
    import readline
except ImportError:
    pass  # readline not available, input will be more basic


def main(args):
    print("Starting inference setup...")
    print(f"Base Model ID: {args.model_id}")
    print(f"Adapter Path: {args.adapter_path}")
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

    # --- 2. Load Base Model & Tokenizer ---
    print(f"Loading base model ({args.model_id})...")
    # Set device_map="auto" for automatic multi-GPU handling (if available)
    # Set trust_remote_code=True if necessary
    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config

    base_model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    print("Base model loaded.")

    print(f"Loading tokenizer for {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, trust_remote_code=True, use_fast=False, legacy=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set pad_token to eos_token")
    print("Tokenizer loaded.")

    # --- 3. Load PEFT Adapter and Merge ---
    print(f"Loading adapter from {args.adapter_path}...")
    # Load the LoRA adapter weights from the specified path
    # Ensure the path points to the directory containing 'adapter_config.json', etc.
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    print("Adapter loaded.")

    # Optional: Merge the adapter into the base model. This creates a new model
    # with the adapter weights integrated. Good for deployment, potentially uses more memory.
    # If not merging, the PeftModel wrapper handles applying the adapter during inference.
    if args.merge_adapter:
        print("Merging adapter into base model...")
        model = (
            model.merge_and_unload()
        )  # Use this if you want a standalone fine-tuned model
        print("Adapter merged.")

    model.eval()  # Set the model to evaluation mode

    print("\nModel ready for inference with persona!")
    print("Type 'quit' or 'exit' to stop.")

    # --- 4. Inference Loop ---
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

            # Alternative for simpler base models (no specific chat template):
            # inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to(model.device)

            print("Generating response...")
            with torch.no_grad():  # Disable gradient calculation for inference
                # Adjust generation parameters as needed (max_new_tokens, temperature, etc.)
                outputs = model.generate(
                    input_ids=inputs,
                    attention_mask=(
                        inputs.attention_mask
                        if hasattr(inputs, "attention_mask")
                        else None
                    ),  # Pass attention_mask if available
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,  # Important for open-ended generation
                )

            # Decode the generated tokens
            # Ensure we decode only the newly generated part, excluding the input prompt
            response_ids = outputs[0][inputs.shape[-1] :]
            response = tokenizer.decode(response_ids, skip_special_tokens=True)

            print("\n--- Persona Response ---")
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
        description="Run inference with a fine-tuned persona adapter."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="microsoft/phi-3-mini-4k-instruct",
        help="Base model ID from Hugging Face Hub.",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to the saved PEFT adapter directory.",
    )
    parser.add_argument(
        "--use_quantization",
        action="store_true",
        help="Use 4-bit quantization for inference (requires bitsandbytes with CUDA).",
    )
    parser.add_argument(
        "--merge_adapter",
        action="store_true",
        help="Merge the adapter weights into the base model after loading.",
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

    # Basic validation (check if adapter path exists)
    # More robust check would verify adapter_config.json etc.
    import os

    if not os.path.isdir(args.adapter_path):
        print(
            f"Error: Adapter path not found or is not a directory: {args.adapter_path}"
        )
        exit(1)

    main(args)
