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
    # Assumes dataset is on Hugging Face Hub or a local path loadable by datasets
    # Filters by 'persona' column using args.persona_name
    # Assumes jsonl format with a 'text' field or 'instruction'/'output' fields
    # TODO: Adapt preprocess_function if dataset structure differs significantly
    try:
        # Load the dataset directly from Hugging Face Hub or local path
        # Assuming 'train' split exists, adjust if needed
        dataset = load_dataset(args.dataset_path, split="train")
        print(f"Full dataset loaded. Number of examples: {len(dataset)}")

        # Filter the dataset based on the persona name (assuming 'persona' column)
        print(f"Filtering for persona: '{args.persona_name}'...")
        filtered_dataset = dataset.filter(lambda example: example.get("persona") == args.persona_name)

        if len(filtered_dataset) == 0:
            raise ValueError(
                f"No data found for persona '{args.persona_name}' in dataset {args.dataset_path}. "
                f"Check dataset identifier, persona name, and column name ('persona')."
            )
        print(f"Filtered dataset ready. Number of examples: {len(filtered_dataset)}")
        print(f"Dataset features: {filtered_dataset.features}")


        # Example: Preprocess/tokenize the dataset
        # This is a basic example, needs refinement based on chosen format
        def preprocess_function(examples):
            # Prioritize instruction/output format for models like Phi-3 Instruct
            if "instruction" in examples and "output" in examples:
                inputs_batch = [] # List to hold all message lists for the batch

                # Determine if the input is a batch or a single example
                is_batch = isinstance(examples["instruction"], list)
                num_examples = len(examples["instruction"]) if is_batch else 1

                # Always iterate to construct the list of message lists
                for i in range(num_examples):
                    instruction = examples["instruction"][i] if is_batch else examples["instruction"]
                    output = examples["output"][i] if is_batch else examples["output"]
                    # Handle optional 'input' field
                    if is_batch:
                        input_field = examples.get("input", [None] * num_examples)[i]
                    else:
                        input_field = examples.get("input")

                    # Format user content
                    if input_field:
                        user_content = f"{instruction}\\n\\nInput: {input_field}"
                    else:
                        user_content = instruction

                    # Create the chat structure for this example
                    msgs = [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": output},
                    ]
                    inputs_batch.append(msgs) # Add this conversation to the batch list

                # Tokenize the entire batch of conversations at once
                # Set add_generation_prompt=False because we provide the full conversation.
                tokenized_list = tokenizer.apply_chat_template(
                    inputs_batch,  # Pass the list of message lists
                    add_generation_prompt=False,
                    truncation=True,
                    padding="max_length",
                    max_length=args.max_seq_length,
                )

                # Return dictionary with only input_ids; collator handles labels & tensors
                model_inputs = {"input_ids": tokenized_list}
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
        # --- Split dataset ---
        print("Splitting dataset into train and evaluation sets (90/10 split)...")
        split_dataset = filtered_dataset.train_test_split(test_size=0.1, seed=42) # Use a fixed seed for reproducibility
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
        print(f"Train examples: {len(train_dataset)}, Evaluation examples: {len(eval_dataset)}")


        # --- Tokenize datasets ---
        # Use the filtered_dataset for mapping
        tokenized_train_dataset = train_dataset.map(
            preprocess_function, batched=True, remove_columns=train_dataset.column_names
        )
        tokenized_eval_dataset = eval_dataset.map(
            preprocess_function, batched=True, remove_columns=eval_dataset.column_names
        )
        print("Datasets tokenized.")

    except Exception as e:
        print(f"Error loading or processing dataset: {e}")
        print(
            "Please ensure the dataset identifier/path is correct, the dataset is accessible (e.g., public or logged in), "
            "contains a 'train' split, a 'persona' column (if filtering), and matches the expected format."
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
        per_device_train_batch_size=2,  # Adjust based on GPU memory
        gradient_accumulation_steps=2,  # Effective batch size = batch_size * grad_accum
        learning_rate=2e-4,
        num_train_epochs=3, 
        logging_steps=10,
        save_steps=100,  # Save checkpoints periodically
        fp16=False,  # QLoRA uses bfloat16 compute dtype, fp16 not directly used here but often needed
        bf16=args.use_bf16,  # Use bfloat16 precision if available (Ampere GPUs or newer)
        gradient_checkpointing=True,  # Already enabled, but good practice
        report_to="wandb",  # Disable default reporting (like wandb) unless configured
        eval_strategy="steps",
        eval_steps=100,
        
        # remove_unused_columns=False, # Ensure this is commented out or removed (default is True)
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
        train_dataset=tokenized_train_dataset, # Use the tokenized training split
        eval_dataset=tokenized_eval_dataset, # Add evaluation dataset
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
        default="microsoft/phi-4-mini-instruct",
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

    # Basic validation - REMOVED check for local dataset_path existence
    # Allow dataset_path to be a HF identifier
    # if not os.path.exists(args.dataset_path):
    #     print(f"Error: Dataset path not found: {args.dataset_path}")
    #     exit(1)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    main(args)
