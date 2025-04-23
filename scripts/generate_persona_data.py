#!/usr/bin/env python3
"""
Generate training data for personas using external LLM APIs.

This script creates synthetic training examples for fine-tuning persona adapters,
using external APIs to generate diverse, high-quality data in the correct format.

Usage:
    python scripts/generate_persona_data.py \
        --persona "captain_codebeard" \
        --count 200 \
        --api "openai" \
        --output "persona_data/captain_codebeard_full.jsonl"
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Union, Any
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv

    # Load environment variables from .env file
    load_dotenv()
    logger.info("Loaded environment variables from .env file")
except ImportError:
    logger.warning("python-dotenv not installed. Will not load from .env file.")
    # If python-dotenv is not installed, try to load using a simple implementation
    env_path = Path(".") / ".env"
    if env_path.exists():
        logger.info("Found .env file, loading with basic parser...")
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip().strip("\"'")
        logger.info("Basic .env parsing complete")

# Optional imports for specific APIs
try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI package not installed. OpenAI API will not be available.")

try:
    import anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning(
        "Anthropic package not installed. Anthropic API will not be available."
    )

# Define persona descriptions
PERSONA_DESCRIPTIONS = {
    "captain_codebeard": """
Captain Codebeard is a swashbuckling pirate who is obsessed with clean code and best practices. 
Key traits:
- Uses pirate slang and nautical metaphors ("Arr!", "ye landlubber", "ship", "seas", "crew", etc.)
- Passionate about clean, maintainable code
- Criticizes vague variable names, missing type hints, and poor documentation
- Speaks authoritatively about programming best practices
- Has a gruff but helpful personality
- Compares good code to a "well-organized pirate ship"
- Uses pirate-themed metaphors for programming concepts (e.g., "debugging is like seeking buried treasure")
- Refers to functions as "commands" and variables as "treasures"
""",
    "professor_snugglesworth": """
Professor Snugglesworth is a brilliant but slightly condescending cat academic who explains concepts with feline analogies.
Key traits:
- Uses sophisticated vocabulary and academic tone
- Makes regular references to being a cat ("as any feline would know", "purrfectly logical")
- Often mentions superiority over dogs in intelligence and sophistication
- Includes cat behaviors in explanations (napping, knocking things off shelves, chasing laser pointers)
- Uses asterisks for action descriptions like "*licks paw thoughtfully*" or "*adjusts spectacles*"
- Makes analogies between complex topics and everyday cat experiences
- Speaks with authority and occasionally dismisses simpler explanations
- Occasionally uses cat puns, but not excessively (mostly "purr" related)
""",
    "zen_coder": """
Zen Coder is a calm, minimalist programmer who speaks in short, profound statements about software.
Key traits:
- Uses very concise, sparse language with short sentences
- Often omits articles or pronouns for brevity
- Speaks in a meditative, philosophical manner
- Makes analogies between code and nature (rivers, mountains, air, etc.)
- Emphasizes simplicity, clarity, and balance in code
- Rarely uses technical jargon, preferring simple metaphors
- Each statement feels complete and meaningful
- Never rambles or over-explains
- Occasionally uses sentence fragments for emphasis
- Values silence and space in code (proper whitespace, minimal comments)
- Treats programming as a spiritual practice rather than just a technical skill
""",
}

# Sample topics to ensure diverse training examples
PROGRAMMING_TOPICS = [
    "variable naming",
    "code organization",
    "functions",
    "debugging",
    "testing",
    "documentation",
    "error handling",
    "performance",
    "readability",
    "refactoring",
    "design patterns",
    "algorithms",
    "data structures",
    "code review",
    "version control",
    "comments",
    "type hints",
    "clean code",
    "technical debt",
    "logging",
    "programming languages",
    "frameworks",
    "libraries",
    "APIs",
    "databases",
    "caching",
    "concurrency",
    "memory management",
    "security",
    "deployment",
    "monitoring",
    "continuous integration",
]

GENERAL_TOPICS = [
    "learning to code",
    "career advice",
    "work-life balance",
    "collaboration",
    "mentoring",
    "technology trends",
    "AI",
    "ethics in technology",
    "open source",
    "startups",
    "big tech companies",
    "hardware",
    "mobile development",
    "web development",
    "game development",
    "cloud computing",
    "DevOps",
    "agile methodology",
    "project management",
]


class APIProvider:
    """Base class for API providers."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def generate(
        self, prompt: str, max_tokens: int = 4000, temperature: float = 0.7
    ) -> str:
        """Generate text from the API."""
        raise NotImplementedError("Subclasses must implement this method")


class OpenAIProvider(APIProvider):
    """Provider for OpenAI API."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        super().__init__(api_key or os.environ.get("OPENAI_API_KEY"))
        self.model = model
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided and not found in environment variables"
            )
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI package not installed. Install with 'pip install openai'"
            )

        openai.api_key = self.api_key
        logger.info("OpenAI API key loaded successfully")

    def generate(
        self, prompt: str, max_tokens: int = 4000, temperature: float = 0.7
    ) -> str:
        """Generate text using OpenAI API."""
        try:
            logger.info(f"Calling OpenAI API with model={self.model}")
            response = openai.chat.completions.create(
                model=self.model,  # Use the specified model
                messages=[
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": "Generate the examples in valid JSONL format.",
                    },
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise


class AnthropicProvider(APIProvider):
    """Provider for Anthropic API."""

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key or os.environ.get("ANTHROPIC_API_KEY"))
        if not self.api_key:
            raise ValueError(
                "Anthropic API key not provided and not found in environment variables"
            )
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "Anthropic package not installed. Install with 'pip install anthropic'"
            )

        self.client = anthropic.Anthropic(api_key=self.api_key)
        logger.info("Anthropic API key loaded successfully")

    def generate(
        self, prompt: str, max_tokens: int = 4000, temperature: float = 0.7
    ) -> str:
        """Generate text using Anthropic API."""
        try:
            logger.info(f"Calling Anthropic API with model=claude-3-opus-20240229")
            response = self.client.messages.create(
                model="claude-3-opus-20240229",  # Or specify a different model
                system=prompt,
                messages=[
                    {
                        "role": "user",
                        "content": "Generate the examples in valid JSONL format.",
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise


def get_provider(
    api: str, api_key: Optional[str] = None, model: Optional[str] = None
) -> APIProvider:
    """Get the appropriate API provider."""
    if api.lower() == "openai":
        return OpenAIProvider(
            api_key, model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        )
    elif api.lower() == "anthropic":
        return AnthropicProvider(api_key)
    else:
        raise ValueError(f"Unsupported API provider: {api}")


def load_seed_examples(persona_name: str) -> List[Dict[str, str]]:
    """Load existing examples for the persona to use as seeds."""
    seed_path = Path("persona_data") / f"{persona_name}.jsonl"

    if not seed_path.exists():
        logger.warning(f"No seed examples found at {seed_path}")
        return []

    examples = []
    try:
        with open(seed_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    examples.append(json.loads(line))
        logger.info(f"Loaded {len(examples)} seed examples for {persona_name}")
        return examples
    except Exception as e:
        logger.error(f"Error loading seed examples: {e}")
        return []


def create_system_prompt(
    persona_name: str,
    example_count: int,
    seed_examples: List[Dict[str, str]],
    batch_size: int,
) -> str:
    """Create the system prompt for the API."""

    if persona_name not in PERSONA_DESCRIPTIONS:
        raise ValueError(f"Unknown persona: {persona_name}")

    persona_desc = PERSONA_DESCRIPTIONS[persona_name]

    # Combine programming and general topics, then choose a diverse subset
    all_topics = PROGRAMMING_TOPICS + GENERAL_TOPICS

    system_prompt = f"""
You will generate {batch_size} diverse training examples for a language model persona.

THE PERSONA DESCRIPTION:
{persona_desc}

Each example should be in this JSON format on a single line (JSONL format):
{{"instruction": "User question or request", "input": "Optional additional context (can be empty string)", "output": "Persona's response in character"}}

IMPORTANT GUIDELINES:
- Generate diverse questions covering different topics and scenarios
- Ensure responses are STRONGLY in character with the persona traits described above
- Include both technical and non-technical questions
- Vary the length and complexity of responses
- Don't repeat similar phrasings or catchphrases too often
- Make the persona's voice distinctive and consistent
- The "input" field can be empty string for most examples, but include it with content for ~20% of examples
- ALWAYS return valid JSON that can be parsed - this is critical
- Each example must be on its own line as complete and valid JSON (JSONL format)
- For technical questions, the answers should be technically ACCURATE despite being in character

Please try to cover a diverse range of these topics: {', '.join(all_topics[:20])}

The total set will have {example_count} examples, but just generate {batch_size} examples in this batch.
"""

    # Add seed examples to show the style
    if seed_examples:
        system_prompt += (
            "\n\nHere are some example responses in the correct persona voice:\n"
        )
        for i, ex in enumerate(seed_examples[:5]):
            input_str = f"\nInput: {ex.get('input')}" if ex.get("input") else ""
            system_prompt += f"\nExample {i+1}:\nInstruction: {ex['instruction']}{input_str}\nOutput: {ex['output']}\n"

    return system_prompt


def parse_generated_examples(text: str) -> List[Dict[str, str]]:
    """Parse generated examples from text."""
    examples = []

    # Try parsing each line as JSON
    for line in text.strip().split("\n"):
        line = line.strip()
        if (
            not line
            or line.startswith("#")
            or line.startswith("```")
            or line.startswith("Example")
        ):
            continue

        try:
            # Handle cases where the API might wrap in code blocks
            if line.startswith("```json") or line.startswith("```jsonl"):
                continue
            if line.startswith("```") and line.endswith("```"):
                line = line[3:-3].strip()

            example = json.loads(line)
            # Validate that it has the required fields
            if "instruction" in example and "output" in example:
                # Ensure input field exists
                example.setdefault("input", "")
                examples.append(example)
            else:
                logger.warning(f"Example missing required fields: {line}")
        except json.JSONDecodeError:
            logger.warning(f"Could not parse line as JSON: {line}")

    return examples


def generate_examples(
    persona_name: str,
    example_count: int,
    api: str,
    batch_size: int = 20,
    temperature: float = 0.7,
    api_key: Optional[str] = None,
    output_file: Optional[str] = None,
    model: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Generate training examples for a persona."""

    # Load seed examples
    seed_examples = load_seed_examples(persona_name)

    # Get the appropriate provider
    provider = get_provider(api, api_key, model)

    # Calculate number of batches
    num_batches = (example_count + batch_size - 1) // batch_size  # Ceiling division

    all_examples = []

    for batch in range(num_batches):
        logger.info(
            f"Generating batch {batch+1}/{num_batches} ({len(all_examples)}/{example_count} examples so far)"
        )

        # Adjust batch size for the last batch
        current_batch_size = min(batch_size, example_count - len(all_examples))

        # Create the system prompt
        system_prompt = create_system_prompt(
            persona_name=persona_name,
            example_count=example_count,
            seed_examples=seed_examples,
            batch_size=current_batch_size,
        )

        # Generate examples
        try:
            generated_text = provider.generate(
                prompt=system_prompt,
                temperature=temperature,
                max_tokens=4000,  # Adjust as needed
            )

            # Parse examples
            batch_examples = parse_generated_examples(generated_text)
            logger.info(f"Generated {len(batch_examples)} examples in this batch")

            all_examples.extend(batch_examples)

            # Save progress if output file is specified
            if output_file and batch_examples:
                save_examples(all_examples, output_file)

            # Sleep to avoid rate limits
            if batch < num_batches - 1:
                time.sleep(2)  # Adjust based on API rate limits

        except Exception as e:
            logger.error(f"Error in batch {batch+1}: {e}")
            # Continue with next batch instead of failing completely

    return all_examples


def save_examples(examples: List[Dict[str, str]], output_file: str) -> None:
    """Save examples to a JSONL file."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    try:
        with open(output_file, "w") as f:
            for example in examples:
                f.write(json.dumps(example) + "\n")
        logger.info(f"Saved {len(examples)} examples to {output_file}")
    except Exception as e:
        logger.error(f"Error saving examples: {e}")
        raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate training data for personas")
    parser.add_argument(
        "--persona",
        required=True,
        choices=PERSONA_DESCRIPTIONS.keys(),
        help="Name of the persona to generate data for",
    )
    parser.add_argument(
        "--count", type=int, default=100, help="Number of examples to generate"
    )
    parser.add_argument(
        "--api",
        choices=["openai", "anthropic"],
        default="openai",
        help="API provider to use",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="API key (will use environment variable if not provided)",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model to use (defaults to gpt-4 for OpenAI, or can be set in .env file)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation (higher = more creative)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Number of examples to generate per API call",
    )
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set verbose logging if requested
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")

    # Set default output file if not specified
    if not args.output:
        args.output = f"persona_data/{args.persona}_generated.jsonl"

    try:
        examples = generate_examples(
            persona_name=args.persona,
            example_count=args.count,
            api=args.api,
            batch_size=args.batch_size,
            temperature=args.temperature,
            api_key=args.api_key,
            output_file=args.output,
            model=args.model,
        )

        logger.info(f"Generated {len(examples)} examples for {args.persona}")
        logger.info(f"Saved to {args.output}")

    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
