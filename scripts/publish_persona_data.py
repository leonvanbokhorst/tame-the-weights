#!/usr/bin/env python3
"""
Little Padawan script to publish persona_data directory as a dataset on Hugging Face Hub.
"""

import os
import sys
import argparse
from typing import List, Tuple
from dotenv import load_dotenv
from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the publish script."""
    parser = argparse.ArgumentParser(
        description="📡 Padawan script to publish persona_data to Hugging Face Hub."
    )
    parser.add_argument(
        "--repo-name",
        "-r",
        type=str,
        required=True,
        help="Name of the dataset repo on Hugging Face (e.g. persona_data).",
    )
    parser.add_argument(
        "--data-dir",
        "-d",
        type=str,
        default="persona_data",
        help="Path to the persona_data directory.",
    )
    return parser.parse_args()


def get_env_variables() -> Tuple[str, str]:
    """Load environment variables for Hugging Face username and token."""
    load_dotenv()
    token = (
        os.getenv("HUGGINGFACE_API_KEY")
        or os.getenv("HF_HUB_TOKEN")
        or os.getenv("HF_TOKEN")
    )
    username = (
        os.getenv("HF_USERNAME")
        or os.getenv("HUGGINGFACE_USERNAME")
        or os.getenv("HF_USER")
    )
    if token is None:
        raise EnvironmentError(
            "Hugging Face API token not found. Please set HUGGINGFACE_API_KEY in .env."
        )
    if username is None:
        raise EnvironmentError(
            "Hugging Face username not found. Please set HF_USERNAME in .env."
        )
    return username, token


def publish_data_to_hub(data_files: List[str], repo_id: str, token: str) -> None:
    """
    Load JSONL data files and push the resulting dataset to Hugging Face Hub.
    :param data_files: List of JSONL file paths.
    :param repo_id: Repository ID in the format <username>/<repo_name>.
    :param token: Hugging Face API token.
    """
    padawan_dataset = load_dataset("json", data_files=data_files)
    padawan_dataset.push_to_hub(repo_id=repo_id, token=token)
    print(
        f"🚀 Yay! Padawan successfully published dataset: https://huggingface.co/datasets/{repo_id}"
    )


def main() -> None:
    """Main entrypoint for the publish script."""
    args = parse_args()
    username, token = get_env_variables()
    repo_id = f"{username}/{args.repo_name}"
    # Gather JSONL files
    if not os.path.isdir(args.data_dir):
        print(f"Error: Data directory {args.data_dir} does not exist.")
        sys.exit(1)
    data_files = [
        os.path.join(args.data_dir, fname)
        for fname in os.listdir(args.data_dir)
        if fname.endswith(".jsonl")
    ]
    if not data_files:
        print(f"Error: No JSONL files found in {args.data_dir}.")
        sys.exit(1)
    print("🔍 Padawan found the following files:")
    for f in data_files:
        print(f"    - {f}")
    # Publish
    publish_data_to_hub(data_files, repo_id, token)


if __name__ == "__main__":
    main()
