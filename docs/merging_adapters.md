# Merging LoRA Adapters

This document explains the process of merging a trained LoRA adapter with its base model to create a standalone, fine-tuned model.

## Why Merge?

While LoRA adapters are efficient for training and storage (as they only contain the *changes* to the base model), sometimes you need a complete, self-contained model:

*   **Compatibility:** Some tools and formats, like GGUF used by `llama.cpp` and Ollama, require a full model rather than separate base+adapter components.
*   **Simplified Deployment:** Having a single model directory can simplify deployment pipelines where loading separate adapters might be cumbersome.
*   **Distribution:** Allows sharing a ready-to-use persona model without requiring users to load both the base and the adapter separately.

## Using the `scripts/merge_adapter.py` Script

This project provides a script to handle the merging process using the PEFT library.

**Purpose:**

The script loads the specified base model (e.g., `microsoft/Phi-4-mini-instruct`) and a trained LoRA adapter (e.g., `captain_codebeard_final_adapter`), merges the adapter weights into the base model's weights, and saves the resulting full model to a new directory.

**Usage:**

The script is run from the command line and accepts several arguments:

```bash
python scripts/merge_adapter.py --persona <persona_name> [options]
```

**Required Argument:**

*   `--persona <persona_name>`: The name of the persona adapter to merge (e.g., `captain_codebeard`, `professor_snugglesworth`, `zen_coder`). The script expects the adapter to be located at `<adapter_dir>/<persona_name>_final_adapter/`.

**Optional Arguments:**

*   `--base_model_id <model_id>`: The Hugging Face ID of the base model (default: `microsoft/Phi-4-mini-instruct`).
*   `--adapter_dir <path>`: The directory containing the trained adapter folders (default: `trained_adapters`).
*   `--output_dir_base <path>`: The base directory where the merged model will be saved. A subdirectory named after the persona will be created here (default: `merged_models`).

**Example:**

```bash
# Merge Captain Codebeard using default settings
python scripts/merge_adapter.py --persona captain_codebeard

# Merge Professor Snugglesworth, specifying a different output location
python scripts/merge_adapter.py --persona professor_snugglesworth --output_dir_base ./standalone_models
```

**Output:**

The script will create a new directory (e.g., `merged_models/captain_codebeard/`) containing the complete merged model files (weights, configuration) and the associated tokenizer files, ready for use as a standard Hugging Face Transformers model.

## Next Steps After Merging

Once merged, the model in the output directory can be:

1.  Uploaded directly to the Hugging Face Hub as a standard model (see [README instructions](../README.md#uploading-merged-models-to-hugging-face-hub)).
2.  Converted to other formats like GGUF for use with tools like Ollama (see [llama.cpp and GGUF for Ollama](gguf_and_ollama.md)). 