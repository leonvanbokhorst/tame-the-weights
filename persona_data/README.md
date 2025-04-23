# Persona Training Data

This directory contains the training data for different personas. Each file corresponds to one persona and should be in JSON Lines (`.jsonl`) format.

Each line in a `.jsonl` file should be a JSON object representing a single training example. The exact format will depend on the fine-tuning task, but a common format for instruction-following or dialogue is:

```json
{
  "instruction": "Some instruction or context",
  "input": "Optional input for the instruction",
  "output": "The desired output in the persona's style"
}
```

Or, for simpler text generation:

```json
{ "text": "A piece of text written in the persona's voice." }
```

Files:

- `professor_snugglesworth.jsonl`: Data for the academic cat persona.
- `captain_codebeard.jsonl`: Data for the pirate coder persona.
- `zen_coder.jsonl`: Data for the minimalist coder persona.
