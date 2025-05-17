# Fine-Tuning Persona Adapters

This document explains how to use the `scripts/fine_tune_persona.py` script to train a LoRA adapter for a specific persona. The script wraps the Hugging Face `transformers` and `peft` libraries to perform parameter-efficient fine-tuning (QLoRA) on a base model.

## Overview

1. **Load a tokenizer and dataset** – The script downloads the dataset, filters it by the chosen persona, and tokenizes each example using the model's chat template.
2. **Configure LoRA** – A LoRA configuration is applied to the base model so that only a small number of parameters are trained.
3. **Run training** – Training arguments control batch size, epochs and other hyper‑parameters. Checkpoints are written to the output directory.
4. **Save the adapter** – After training, the adapter weights and tokenizer are saved to `<output_dir>/<persona_name>_final_adapter`.

## Basic Usage

```bash
python scripts/fine_tune_persona.py \
    --model_id "microsoft/phi-4-mini-instruct" \
    --dataset_path "persona_data/captain_codebeard.jsonl" \
    --persona_name "captain_codebeard" \
    --output_dir "trained_adapters"
```

This trains a new adapter for the `captain_codebeard` persona using the provided dataset. The final adapter is saved inside `trained_adapters/`.

## Important Arguments

- `--model_id` – Base model to fine‑tune (defaults to `microsoft/phi-4-mini-instruct`).
- `--dataset_path` – Dataset identifier or local path in JSONL format.
- `--persona_name` – Name of the persona, used to filter the dataset and name the adapter directory.
- `--output_dir` – Where checkpoints and the final adapter are stored.
- `--max_seq_length` – Maximum length used during tokenization (default `512`).
- `--use_quantization` – Enable QLoRA 4‑bit quantization with `bitsandbytes`.
- `--use_bf16` – Use bfloat16 precision on supported GPUs.

Additional LoRA and training hyper‑parameters are defined in the script and can be modified as needed.

## Example

Assuming you generated additional training data and want to fine‑tune the Zen Coder persona with quantization enabled:

```bash
python scripts/fine_tune_persona.py \
    --model_id "microsoft/phi-4-mini-instruct" \
    --dataset_path "persona_data/zen_coder_full.jsonl" \
    --persona_name "zen_coder" \
    --output_dir "trained_adapters" \
    --use_quantization \
    --use_bf16
```

After training finishes, the adapter will be saved to `trained_adapters/zen_coder_final_adapter/`.




