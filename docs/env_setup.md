# Setting Up API Credentials

## Using API Keys with the Data Generation Script

The `generate_persona_data.py` script can use external API providers (OpenAI and Anthropic) to create high-quality training data for personas. For this to work, you need to set up your API credentials.

## Option 1: Using a .env File (Recommended)

Create a file named `.env` in the project root with your API credentials:

```
# API Keys for persona data generation
# NEVER commit this file to version control!

# OpenAI API key (for generating training data with GPT-4)
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic API key (alternative for generating training data with Claude models)
# ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Specify custom model names (optional)
# OPENAI_MODEL=gpt-o3-nano
# ANTHROPIC_MODEL=claude-3-opus-20240229
```

The script will automatically load this file if the `python-dotenv` package is installed. If not, it will try to parse it using a basic implementation.

**Important**: Add `.env` to your `.gitignore` to prevent accidentally committing your API keys to version control.

## Option 2: Using Environment Variables

Set the environment variables directly in your shell:

```bash
# For OpenAI
export OPENAI_API_KEY="your_openai_api_key_here"
export OPENAI_MODEL="gpt-o3-nano"  # Optional: specify a custom model

# For Anthropic
export ANTHROPIC_API_KEY="your_anthropic_api_key_here"
```

## Option 3: Providing API Keys and Model Directly

You can also provide the API key and model directly as command-line arguments:

```bash
python scripts/generate_persona_data.py \
    --persona captain_codebeard \
    --count 200 \
    --api openai \
    --model gpt-o3-nano \
    --api-key "your_openai_api_key_here"
```

This is less secure as the key may be stored in your shell history.

## Getting API Keys

- **OpenAI API Keys**: Available from [OpenAI's platform](https://platform.openai.com/api-keys)
- **Anthropic API Keys**: Available from [Anthropic's console](https://console.anthropic.com/)

## Available Models

Depending on your API provider, you can specify different models:

### OpenAI Models

- `gpt-4` (default)
- `gpt-o3-nano`
- `gpt-3.5-turbo`
- Other models available in your OpenAI account

### Anthropic Models

- `claude-3-opus-20240229` (default)
- `claude-3-sonnet-20240229`
- `claude-3-haiku-20240307`
- Other Claude models available in your Anthropic account
