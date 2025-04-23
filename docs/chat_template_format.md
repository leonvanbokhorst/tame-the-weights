# Phi-4-mini-instruct Chat Template Format

This document explains the specific chat template format required by the `microsoft/Phi-4-mini-instruct` model, which is used as the base model in this project. Understanding this format is crucial for preparing training data and ensuring correct model behavior during fine-tuning and inference.

## Purpose of Chat Templates

Chat templates define how a conversation, typically consisting of turns from different roles (like system, user, assistant), is formatted into a single string that the language model can process. They include special tokens to delineate between messages and roles.

Using the correct template is essential because the model was trained (or fine-tuned) expecting this specific structure. Mismatched templates can lead to poor performance or nonsensical outputs.

## Phi-4-mini-instruct Template Structure

The required format for `microsoft/Phi-4-mini-instruct` is as follows:

```yaml
<|system|>System Message<|end|><|user|>User Message<|end|><|assistant|>Assistant Response
```

### Components:

*   **`<|system|>`**: Special token indicating the start of a system message. System messages provide high-level instructions or context for the conversation (e.g., defining the AI's persona or task).
*   **`<|user|>`**: Special token indicating the start of a user message. This is typically the user's query or instruction.
*   **`<|assistant|>`**: Special token indicating the start of the assistant's response. During training, this is followed by the desired output. During inference, the model generates text after this token.
*   **`<|end|>`**: Special token marking the end of a message from a specific role (`system` or `user`).

### Example:

Let's say we have:

*   **System Message:** "You are Captain Codebeard, a pirate coder."
*   **User Message:** "Explain Python list comprehensions, ye scallywag!"
*   **Assistant Response:** "Arr, matey! List comprehensions be a way to forge new lists from old ones in a single, swift line..."

The formatted input string for the model would look like this:

```
<|system|>You are Captain Codebeard, a pirate coder.<|end|><|user|>Explain Python list comprehensions, ye scallywag!<|end|><|assistant|>Arr, matey! List comprehensions be a way to forge new lists from old ones in a single, swift line...
```

## Application in Fine-Tuning

In our `scripts/fine_tune_persona.py` script, the `preprocess_function` takes the `instruction`, `input` (optional), and `output` fields from the dataset and formats them according to this template. It uses the `tokenizer.apply_chat_template` method provided by the Hugging Face `transformers` library, which automatically constructs the correct string based on the tokenizer's predefined chat template.

During training, the entire formatted string (including the system prompt, user query, *and* the target assistant response) is fed into the model. The model learns to predict the `<|assistant|>` response part, given the preceding context. 