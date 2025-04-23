"""
Fine-tunes a base model with a specified persona dataset using PEFT (QLoRA).

Usage:
  python scripts/fine_tune_persona.py --model_id <base_model_id> \
                                      --dataset_path <path_to_jsonl> \
                                      --persona_name <name_for_adapter> \
                                      --output_dir trained_adapters
"""

import argparse
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,  # Or DataCollatorForSeq2Seq if needed
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# --- Configuration ---
# TODO: Add more configuration options as needed (learning rate, epochs, etc.)


def print_trainable_parameters(model):
    """Prints the number of trainable parameters in the model."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || "
        f"trainable%: {100 * trainable_params / all_param:.2f}"
    )


def main(args):
    print("Starting fine-tuning process...")
    print(f"Base Model ID: {args.model_id}")
    print(f"Dataset Path: {args.dataset_path}")
    print(f"Persona Name: {args.persona_name}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Using Quantization: {args.use_quantization}")

    # --- 1. Load Tokenizer ---
    # Use legacy=False for models like Llama 3, Phi-3 etc.
    # Set trust_remote_code=True if necessary for the model.
    print(f"Loading tokenizer for {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, trust_remote_code=True, use_fast=False, legacy=False
    )
    # Set padding token if it's not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Common practice
        print("Set pad_token to eos_token")

    # --- 2. Load Dataset ---
    print(f"Loading dataset from {args.dataset_path}...")
    # Assumes jsonl format with a 'text' field or 'instruction'/'output' fields
    # TODO: Adapt this based on the actual JSONL structure in persona_data/README.md
    try:
        dataset = load_dataset("json", data_files=args.dataset_path, split="train")
        print(f"Dataset loaded successfully. Number of examples: {len(dataset)}")
        print(f"Dataset features: {dataset.features}")

        # Example: Preprocess/tokenize the dataset
        # This is a basic example, needs refinement based on chosen format
        def preprocess_function(examples):
            # Prioritize instruction/output format for models like Phi-3 Instruct
            if "instruction" in examples and "output" in examples:
                # Construct messages in the format the model expects
                # Use None for input if it's not present or empty
                messages = []
                # Process examples row by row as apply_chat_template usually works on conversations
                # Note: This loop inside map might be slow for large datasets.
                # Consider optimizing if performance becomes an issue.
                # However, HF datasets map often passes batches as dicts of lists,
                # so we need to iterate.
                inputs = []
                outputs = []
                if isinstance(
                    examples["instruction"], list
                ):  # Check if input is a batch
                    for i in range(len(examples["instruction"])):
                        instruction = examples["instruction"][i]
                        output = examples["output"][i]
                        # Handle optional 'input' field if present
                        input_field = examples.get(
                            "input", [None] * len(examples["instruction"])
                        )[i]

                        # Format based on whether 'input' exists for this example
                        if input_field:
                            user_content = f"{instruction}\\n\\nInput: {input_field}"
                        else:
                            user_content = instruction

                        # Create the chat structure
                        # The entire conversation (user prompt + assistant response) is the training target
                        msgs = [
                            {"role": "user", "content": user_content},
                            {"role": "assistant", "content": output},
                        ]
                        inputs.append(msgs)  # Keep track of formatted messages

                    # Tokenize the formatted chat strings
                    # Important: We train the model to predict the assistant's response,
                    # including the template tokens around it.
                    # Set add_generation_prompt=False because we provide the full conversation.
                    model_inputs = tokenizer.apply_chat_template(
                        inputs,  # Pass the list of message lists
                        add_generation_prompt=False,
                        truncation=True,
                        padding="max_length",
                        max_length=args.max_seq_length,  # Use an arg for max_length
                        return_tensors="pt",  # Return PyTorch tensors
                    )
                    # We need to return a dict compatible with the Trainer
                    # apply_chat_template might return just input_ids, attention_mask etc.
                    # For Causal LM, the 'labels' are typically the same as 'input_ids'
                    # The loss function handles shifting internally.
                    model_inputs["labels"] = model_inputs["input_ids"].clone()
                    return model_inputs

                else:  # Handle single example (less common with batched=True)
                    instruction = examples["instruction"]
                    output = examples["output"]
                    input_field = examples.get("input")
                    if input_field:
                        user_content = f"{instruction}\\n\\nInput: {input_field}"
                    else:
                        user_content = instruction
                    msgs = [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": output},
                    ]
                    # Tokenize single instance (less efficient)
                    model_inputs = tokenizer.apply_chat_template(
                        msgs,
                        add_generation_prompt=False,
                        truncation=True,
                        padding="max_length",
                        max_length=args.max_seq_length,  # Use an arg for max_length
                        return_tensors="pt",
                    )
                    model_inputs["labels"] = model_inputs["input_ids"].clone()
                    # Need to un-batch if input was single example for map compatibility?
                    # Let's assume map handles this.
                    return model_inputs

            # Fallback for simple 'text' format (less ideal for instruct models)
            elif "text" in examples:
                print(
                    "Warning: Using basic 'text' field format. Instruction format is preferred for instruct models."
                )
                # Simple tokenization for Causal LM
                model_inputs = tokenizer(
                    examples["text"],
                    truncation=True,
                    padding="max_length",
                    max_length=args.max_seq_length,  # Use an arg for max_length
                )
                # For Causal LM, labels are typically input_ids shifted
                model_inputs["labels"] = model_inputs["input_ids"].clone()
                return model_inputs
            else:
                raise ValueError(
                    "Dataset must contain 'instruction'/'output' fields or a 'text' field."
                )

        print("Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            preprocess_function, batched=True, remove_columns=dataset.column_names
        )
        print("Dataset tokenized.")

    except Exception as e:
        print(f"Error loading or processing dataset: {e}")
        print(
            "Please ensure the dataset exists, is a valid JSONL file, and matches the expected format."
        )
        return  # Exit if dataset loading fails

    # --- 3. Configure Quantization (Optional) ---
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
                print("Configuring BitsAndBytes for QLoRA...")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,  # Use bfloat16 for faster training
                )
        except ImportError:
            print("bitsandbytes not available. Disabling quantization.")
            args.use_quantization = False

    # --- 4. Load Base Model ---
    print(f"Loading base model ({args.model_id})...")
    # Set device_map="auto" for automatic multi-GPU handling (if available)
    # Set trust_remote_code=True if necessary
    model_kwargs = {
        "device_map": "auto",  # Automatically uses available GPU(s) or CPU
        "trust_remote_code": True,
    }
    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    print("Base model loaded.")

    # --- 5. Prepare Model for Training ---
    print("Preparing model for training...")
    model.gradient_checkpointing_enable()  # Reduce memory usage
    if args.use_quantization:
        model = prepare_model_for_kbit_training(model)
    print("Model prepared.")

    # --- 6. Configure LoRA ---
    print("Configuring LoRA...")
    # TODO: Target modules might need adjustment based on the specific model architecture
    # Common targets: 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'
    # Use tools or model documentation to find appropriate target modules
    lora_config = LoraConfig(
        r=16,  # Rank of the update matrices. Higher rank = more parameters, potentially more expressive.
        lora_alpha=32,  # LoRA scaling factor
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],  # Example for Llama-like models, ADJUST AS NEEDED!
        lora_dropout=0.05,  # Dropout probability for LoRA layers
        bias="none",  # Usually set to 'none' for LoRA
        task_type="CAUSAL_LM",  # Important for Causal Language Models
    )
    print("LoRA Configured.")

    # --- 7. Apply PEFT to the Model ---
    print("Applying PEFT (LoRA) to the model...")
    model = get_peft_model(model, lora_config)
    print("PEFT applied.")
    print_trainable_parameters(model)  # Show how few parameters are trainable!

    # --- 8. Training Arguments ---
    # Define output directory for checkpoints specific to this persona
    persona_output_dir = os.path.join(args.output_dir, args.persona_name)
    print(f"Training checkpoints will be saved to: {persona_output_dir}")

    # TODO: Adjust training hyperparameters as needed (learning rate, epochs, batch size, etc.)
    # These are example values and might need significant tuning.
    training_args = TrainingArguments(
        output_dir=persona_output_dir,
        per_device_train_batch_size=4,  # Adjust based on GPU memory
        gradient_accumulation_steps=4,  # Effective batch size = batch_size * grad_accum
        learning_rate=2e-4,
        num_train_epochs=1,  # Start with 1 epoch for fast iteration
        logging_steps=10,
        save_steps=50,  # Save checkpoints periodically
        fp16=False,  # QLoRA uses bfloat16 compute dtype, fp16 not directly used here but often needed
        bf16=args.use_bf16,  # Use bfloat16 precision if available (Ampere GPUs or newer)
        gradient_checkpointing=True,  # Already enabled, but good practice
        report_to="none",  # Disable default reporting (like wandb) unless configured
        # Add other relevant arguments: evaluation_strategy, save_total_limit, etc.
    )
    print("Training arguments set.")

    # --- 9. Data Collator ---
    # Use DataCollatorForLanguageModeling for Causal LM task
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    print("Data collator set.")

    # --- 10. Initialize Trainer ---
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        # eval_dataset=... # Add evaluation dataset if available
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    print("Trainer initialized.")

    # --- 11. Train ---
    print("Starting training...")
    try:
        trainer.train()
        print("Training finished.")
    except Exception as e:
        print(f"Error during training: {e}")
        # Potentially add cleanup or error handling here
        return

    # --- 12. Save Adapter ---
    final_adapter_path = os.path.join(
        args.output_dir, f"{args.persona_name}_final_adapter"
    )
    print(f"Saving final adapter model to {final_adapter_path}...")
    model.save_pretrained(final_adapter_path)  # Saves only the adapter weights
    tokenizer.save_pretrained(final_adapter_path)  # Save tokenizer alongside adapter
    print("Adapter saved successfully.")
    print("Fine-tuning process complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune a model with a specific persona using QLoRA."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="microsoft/phi-3-mini-4k-instruct",
        help="Base model ID from Hugging Face Hub.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the persona training data (JSONL format).",
    )
    parser.add_argument(
        "--persona_name",
        type=str,
        required=True,
        help="Name of the persona (used for saving the adapter).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="trained_adapters",
        help="Directory to save the trained adapters.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,  # Default max sequence length
        help="Maximum sequence length for tokenization.",
    )
    parser.add_argument(
        "--use_quantization",
        action="store_true",
        help="Use 4-bit quantization (requires bitsandbytes with CUDA).",
    )
    parser.add_argument(
        "--use_bf16",
        action="store_true",
        help="Use bfloat16 precision (faster on modern GPUs).",
    )
    # TODO: Add arguments for hyperparameters like learning_rate, epochs, batch_size, lora_r, lora_alpha etc.

    args = parser.parse_args()

    # Basic validation
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset path not found: {args.dataset_path}")
        exit(1)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    main(args)
