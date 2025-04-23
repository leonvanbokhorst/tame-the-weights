# Tame the Weights: Plug-and-Play Personas

This project implements lightweight persona adapters for language models using Parameter-Efficient Fine-Tuning (PEFT) techniques like LoRA. Each adapter modifies the behavior of a base model to adopt a specific persona, without changing the underlying model weights.

## 🧙 Personas

Current personas include:

- **Professor Snugglesworth:** A brilliant, slightly aloof cat academic who explains concepts with feline analogies.
- **Captain Codebeard:** A swashbuckling pirate obsessed with clean code and best practices.
- **Zen Coder:** A calm, minimalist programmer who speaks in short, profound statements about software.

## 🧰 Project Structure

```
tame-the-weights/
├── persona_data/               # Training data for each persona (JSONL format)
│   ├── captain_codebeard.jsonl # Pirate coder persona training data
│   ├── professor_snugglesworth.jsonl # Cat academic persona training data
│   ├── zen_coder.jsonl         # Zen programmer persona training data
│   └── README.md               # Data format documentation
├── scripts/                    # Python scripts
│   ├── fine_tune_persona.py    # Script for fine-tuning a persona adapter
│   └── run_persona_inference.py # Script for inference with a trained adapter
├── docs/                       # Documentation
│   └── technical_approach.md   # Detailed explanation of the technical approach
├── trained_adapters/           # Saved adapter models (created during training)
└── requirements.txt            # Python dependencies
```

## 🚀 Setup

1. Create and activate a Python virtual environment:

```bash
# Using UV (for faster dependency resolution)
uv venv -p python3.12 .venv
source .venv/bin/activate

# Or using standard venv
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
# Using UV
uv pip sync requirements.txt

# Or using standard pip
pip install -r requirements.txt
```

## 🏃‍♂️ Usage

### Training a Persona Adapter

To fine-tune a base model with a specific persona:

```bash
python scripts/fine_tune_persona.py \
    --model_id "microsoft/phi-3-mini-4k-instruct" \
    --dataset_path "persona_data/captain_codebeard.jsonl" \
    --persona_name "captain_codebeard" \
    --output_dir "trained_adapters"
```

Additional options:

- `--max_seq_length`: Maximum sequence length for tokenization (default: 512)
- `--use_quantization`: Use 4-bit quantization if available (requires bitsandbytes with CUDA)
- `--use_bf16`: Use bfloat16 precision on supported hardware

### Running Inference with a Persona Adapter

To chat with a model using a trained persona adapter:

```bash
python scripts/run_persona_inference.py \
    --model_id "microsoft/phi-3-mini-4k-instruct" \
    --adapter_path "trained_adapters/captain_codebeard_final_adapter"
```

Additional options:

- `--use_quantization`: Use 4-bit quantization if available
- `--merge_adapter`: Merge adapter weights into the base model
- `--max_new_tokens`: Maximum tokens to generate (default: 200)
- `--temperature`: Generation temperature (default: 0.7)

## 📖 Documentation

For a detailed explanation of the technical approach, see [Technical Approach](docs/technical_approach.md), which covers:

- Core concepts of Parameter-Efficient Fine-Tuning
- Explanation of Low-Rank Adaptation (LoRA) with mathematical details
- Implementation of tokenization and model training
- Platform compatibility considerations
- Technical challenges and solutions

## 📝 Notes

- This project demonstrates Parameter-Efficient Fine-Tuning (PEFT) using LoRA.
- When using macOS with ARM64 (Apple Silicon), quantization is disabled as bitsandbytes is not compatible.
- The default base model is Microsoft's Phi-3-mini-4k-instruct, but other models can be specified.
- Training typically requires a CUDA-compatible GPU for reasonable performance.

## 📦 Dependencies

- PyTorch
- Transformers
- PEFT
- Datasets
- Accelerate
- Sentencepiece & Protobuf (for tokenizers)
