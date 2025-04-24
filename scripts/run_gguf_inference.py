#!/usr/bin/env python
"""
Tests GGUF quantized model files using llama.cpp.

Usage:
  python scripts/test_gguf_model.py --model_path <path_to_gguf_file> [--n_ctx 2048] [--n_gpu_layers 0]
"""

import argparse
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


def load_llama_cpp():
    """Load the llama_cpp module with helpful error messages if it fails."""
    try:
        import llama_cpp
        return llama_cpp
    except ImportError:
        print("Error: llama_cpp module not found.")
        print("You can install it with: pip install llama-cpp-python")
        print("For GPU support: pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir --extra-index-url=https://abetlen.github.io/llama-cpp-python/cpu+cuda/")
        exit(1)


def format_message(messages: List[Dict[str, str]]) -> str:
    """Format messages according to Llama2/Mistral/Phi format."""
    formatted = ""
    for message in messages:
        role = message["role"]
        content = message["content"]
        
        if role == "system":
            formatted += f"{content}\n"
        elif role == "user":
            formatted += f"USER: {content}\n"
        elif role == "assistant":
            formatted += f"ASSISTANT: {content}\n"
        else:
            formatted += f"{role.upper()}: {content}\n"
    
    formatted += "ASSISTANT: "
    return formatted


def main(args):
    # Initialize
    llama_cpp = load_llama_cpp()
    model_path = os.path.abspath(args.model_path)
    print(f"Testing GGUF model: {model_path}")
    
    # Check if file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        exit(1)
    
    # Load model
    print(f"Loading model... (n_ctx={args.n_ctx}, n_gpu_layers={args.n_gpu_layers})")
    start_time = time.time()
    
    # Configure model parameters
    model_params = {
        "model_path": model_path,
        "n_ctx": args.n_ctx,
        "n_gpu_layers": args.n_gpu_layers,
        "verbose": args.verbose
    }
    
    # Add optional parameters if needed
    if args.n_batch:
        model_params["n_batch"] = args.n_batch
    if args.seed is not None:
        model_params["seed"] = args.seed
        
    try:
        model = llama_cpp.Llama(**model_params)
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)
        
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds")
    
    # Inference loop
    print("\nModel ready for testing!")
    print("Type 'quit' or 'exit' to stop.")
    
    while True:
        try:
            prompt = input("\nEnter prompt: ")
            if prompt.lower() in ["quit", "exit"]:
                break
                
            # Format prompt
            messages = [
                {"role": "user", "content": prompt}
            ]
            formatted_prompt = format_message(messages)
            
            # Perform inference
            print("Generating response...")
            start_time = time.time()
            
            # Set generation parameters
            kwargs = {
                "prompt": formatted_prompt,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "echo": False,
                "stream": True
            }
            
            # Stream response
            print("\n--- Model Response ---")
            response_text = ""
            for chunk in model(**kwargs):
                chunk_text = chunk["choices"][0]["text"]
                print(chunk_text, end="", flush=True)
                response_text += chunk_text
                
            inference_time = time.time() - start_time
            tokens_per_second = len(response_text.split()) / inference_time
            
            print("\n----------------------")
            print(f"Generated in {inference_time:.2f} seconds (~{tokens_per_second:.2f} tokens/sec)")
            
        except KeyboardInterrupt:
            print("\nInference interrupted.")
            continue
        except EOFError:
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            continue
            
    print("\nExiting test script.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test GGUF quantized models using llama.cpp")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the GGUF model file",
    )
    parser.add_argument(
        "--n_ctx",
        type=int,
        default=2048,
        help="Context size (token window)",
    )
    parser.add_argument(
        "--n_gpu_layers",
        type=int,
        default=0,
        help="Number of layers to offload to GPU (0 = CPU only)",
    )
    parser.add_argument(
        "--n_batch",
        type=int,
        default=None,
        help="Batch size for prompt processing",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=200,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for text generation",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible generation",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    main(args) 