# llama.cpp and GGUF for Ollama

This document provides a brief overview of `llama.cpp`, the GGUF format, and their relevance to running the custom persona models locally using Ollama.

## What is `llama.cpp`?

[llama.cpp](https://github.com/ggerganov/llama.cpp) is a high-performance library written in C++ for running Large Language Models (LLMs) efficiently, even on consumer hardware (including CPUs). It was initially focused on Meta's LLaMA models but has expanded to support a wide range of architectures (like the Phi models we are using).

**Why is it relevant to this project?**

`llama.cpp` provides the essential command-line tools needed to:

1.  **Convert** models from the standard Hugging Face Transformers format (like our merged models) into the GGUF format.
2.  **Quantize** GGUF models, significantly reducing their size and computational requirements, making them suitable for local execution via tools like Ollama.

To use these tools, you need to clone the `llama.cpp` repository from GitHub and compile it on your system.

## What is GGUF?

GGUF (Georgi Gerganov Universal Format) is a file format developed by the `llama.cpp` team specifically for storing LLMs.

**Key Features:**

*   **Single File:** Typically stores the entire model (weights, tokenizer info, metadata) in one file.
*   **Quantization Support:** Designed to handle various quantization methods (like 4-bit, 5-bit, etc.) efficiently.
*   **Extensibility:** Allows adding new metadata without breaking compatibility.
*   **Framework Agnostic (in theory):** While primarily used by the `llama.cpp` ecosystem, it aims to be a universal format.

**Why is it relevant to this project?**

Ollama, the tool we aim to use for running our persona models easily, primarily uses the **GGUF format**. Therefore, we need to convert our merged persona models into quantized GGUF files before Ollama can use them.

## The Conversion Workflow

The typical process to get from our trained adapter to a model usable by Ollama looks like this:

1.  **Merge:** Combine the LoRA adapter with the base Hugging Face model (`scripts/merge_adapter.py`). This creates a full standalone model in HF format.
2.  **Convert to GGUF (f16):** Use `llama.cpp/convert.py` to change the merged HF model into an initial, unquantized (or float16) GGUF file.
3.  **Quantize GGUF:** Use `llama.cpp/quantize` (a compiled executable from the `llama.cpp` build) to reduce the size and precision of the GGUF file (e.g., to 4-bit quantization).
4.  **Create Ollama Modelfile:** Write a simple text file (`Modelfile`) defining the base GGUF model, parameters, and the system prompt for the persona.
5.  **Use with Ollama:** Use `ollama create` to bundle the quantized GGUF and the Modelfile, making the persona model available locally via `ollama run`. 