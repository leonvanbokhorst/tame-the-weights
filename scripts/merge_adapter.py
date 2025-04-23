import torch
import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main(args):
    # --- Configuration from args ---
    BASE_MODEL_ID = args.base_model_id
    PERSONA_TO_MERGE = args.persona
    LOCAL_ADAPTER_DIR = args.adapter_dir
    OUTPUT_DIR_BASE = args.output_dir_base

    # Construct paths
    adapter_path = os.path.join(LOCAL_ADAPTER_DIR, f"{PERSONA_TO_MERGE}_final_adapter")
    output_path = os.path.join(OUTPUT_DIR_BASE, PERSONA_TO_MERGE)

    print(f"--- Starting Merge Process for Persona: {PERSONA_TO_MERGE} ---")
    print(f"Base Model: {BASE_MODEL_ID}")
    print(f"Adapter Path: {adapter_path}")
    print(f"Output Path: {output_path}")

    # Check if adapter exists
    if not os.path.isdir(adapter_path):
        print(f"Error: Adapter directory not found at {adapter_path}")
        exit(1)

    # Ensure output directory exists
    if not os.path.exists(output_path):
        print(f"Creating output directory: {output_path}")
        os.makedirs(output_path)

    # --- Load Base Model and Tokenizer ---
    print("Loading base model...")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            return_dict=True,
            torch_dtype=torch.float16, # Using float16 to potentially reduce memory
            device_map="auto", # Let transformers handle device placement
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"Error loading base model {BASE_MODEL_ID}: {e}")
        print("Check model ID, network connection, and available memory/disk space.")
        exit(1)

    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer for {BASE_MODEL_ID}: {e}")
        exit(1)

    # --- Load LoRA Adapter ---
    print(f"Loading adapter {PERSONA_TO_MERGE} from {adapter_path}...")
    try:
        # Ensure the base model is loaded before applying the adapter
        model = PeftModel.from_pretrained(base_model, adapter_path)
    except Exception as e:
        print(f"Error loading adapter from {adapter_path}: {e}")
        print("Ensure the adapter path is correct and compatible with the base model.")
        exit(1)
    print("Adapter loaded.")

    # --- Merge Adapter and Base Model ---
    print("Merging adapter weights into the base model...")
    try:
        model = model.merge_and_unload()
    except Exception as e:
        print(f"Error merging adapter: {e}")
        exit(1)
    print("Merge complete. Model is now a standalone fine-tuned model.")

    # --- Save Merged Model and Tokenizer ---
    print(f"Saving merged model to {output_path}...")
    try:
        model.save_pretrained(output_path)
    except Exception as e:
        print(f"Error saving merged model to {output_path}: {e}")
        exit(1)
    print("Merged model saved.")

    print(f"Saving tokenizer to {output_path}...")
    try:
        tokenizer.save_pretrained(output_path)
    except Exception as e:
        print(f"Error saving tokenizer to {output_path}: {e}")
        exit(1)
    print("Tokenizer saved.")

    print(f"\n--- Merge Process for {PERSONA_TO_MERGE} Finished Successfully --- ")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter with a base model.")
    parser.add_argument(
        "--persona",
        type=str,
        required=True,
        help="Name of the persona adapter to merge (e.g., 'captain_codebeard')"
    )
    parser.add_argument(
        "--base_model_id",
        type=str,
        default="microsoft/Phi-4-mini-instruct",
        help="Base model ID from Hugging Face Hub."
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default="trained_adapters",
        help="Directory containing the final adapter folders."
    )
    parser.add_argument(
        "--output_dir_base",
        type=str,
        default="merged_models",
        help="Base directory where the merged model folder will be saved."
    )

    args = parser.parse_args()
    main(args) 