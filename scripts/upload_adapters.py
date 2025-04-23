import os
import dotenv
from huggingface_hub import HfApi, upload_folder, hf_hub_download, upload_file
from huggingface_hub.utils import RepositoryNotFoundError, EntryNotFoundError
import tempfile

# --- Configuration ---
BASE_MODEL_ID = "microsoft/Phi-4-mini-instruct"
PERSONAS = ["captain_codebeard", "professor_snugglesworth", "zen_coder"]
LOCAL_ADAPTERS_DIR = "trained_adapters"
# Assumes your Hugging Face username is leonvanbokhorst based on previous context
HF_USERNAME = "leonvanbokhorst"
DATASET_ID = "leonvanbokhorst/tame-the-weights-personas"
DATASET_URL = f"https://huggingface.co/datasets/{DATASET_ID}"

# --- Load Environment Variables ---
print("Loading environment variables from .env file...")
dotenv.load_dotenv()
hf_token = os.getenv("HUGGINGFACE_API_KEY")

if not hf_token:
    print("Error: HUGGINGFACE_API_KEY not found in .env file.")
    print("Please ensure your Hugging Face API key is set as HUGGINGFACE_API_KEY in the .env file.")
    exit(1)
else:
    print("Hugging Face API key loaded successfully.")

# --- Prepare for Upload ---
# Clean the base model ID for use in the repo name (replace '/' with '-')
cleaned_base_model_id = BASE_MODEL_ID.replace("/", "-")
api = HfApi(token=hf_token)

# --- Upload Each Adapter ---
for persona in PERSONAS:
    print(f"\n--- Processing Persona: {persona} ---")

    # Construct local path and repo ID
    local_adapter_path = os.path.join(LOCAL_ADAPTERS_DIR, f"{persona}_final_adapter")
    repo_id = f"{HF_USERNAME}/{cleaned_base_model_id}-{persona}-adapter"
    repo_url = f"https://huggingface.co/{repo_id}"

    print(f"Local adapter path: {local_adapter_path}")
    print(f"Target Hugging Face repo ID: {repo_id}")

    # Check if local adapter directory exists
    if not os.path.isdir(local_adapter_path):
        print(f"Error: Local adapter directory not found: {local_adapter_path}")
        print("Skipping this persona.")
        continue

    # Create the repository on the Hub if it doesn't exist
    try:
        print(f"Ensuring repository {repo_id} exists on the Hub...")
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            exist_ok=True  # Don't raise error if repo already exists
        )
        print(f"Repository {repo_id} exists or was created.")
    except Exception as e:
        print(f"Error creating repository {repo_id}: {e}")
        print("Please check your token and permissions.")
        print("Skipping upload for this persona.")
        continue

    # Upload the adapter folder contents
    try:
        print(f"Uploading {persona} adapter contents to {repo_id}...")
        # Note: We previously used create_pr=True, changed to False for direct commit
        # based on user feedback/experience with needing to merge PRs.
        upload_folder(
            folder_path=local_adapter_path,
            repo_id=repo_id,
            repo_type="model",
            token=hf_token,
            commit_message=f"Upload {persona} adapter files for {BASE_MODEL_ID}",
            create_pr=False, # Commit directly to main
        )
        print(f"Successfully uploaded adapter files to {repo_id}")

    except Exception as e:
        print(f"Error uploading {persona} adapter files: {e}")
        print("Skipping README update for this persona.")
        continue

    # --- Create/Update README.md (Model Card) ---
    try:
        print(f"Creating/Updating README.md for {repo_id}...")
        readme_content = f"""
# LoRA Adapter: {persona}

This repository contains a LoRA (Low-Rank Adaptation) adapter for the base model `{BASE_MODEL_ID}`.

This adapter fine-tunes the base model to adopt the **{persona}** persona.

Find the adapter files in this repository.

## Training Data

This adapter was fine-tuned on the `{persona}` subset of the [{DATASET_ID}]({DATASET_URL}) dataset.

## Usage (Example with PEFT)

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_id = "{BASE_MODEL_ID}"
adapter_repo_id = "{repo_id}"

# Load the base model and tokenizer
model = AutoModelForCausalLM.from_pretrained(base_model_id)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# Load the PEFT model
model = PeftModel.from_pretrained(model, adapter_repo_id)

# Now you can use the model for inference with the persona applied
# Example:
input_text = "Explain the concept of technical debt."
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
        """

        # Create a temporary file to write the README content
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".md") as temp_readme:
            temp_readme.write(readme_content)
            temp_readme_path = temp_readme.name

        # Upload the README.md file
        upload_file(
            path_or_fileobj=temp_readme_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            token=hf_token,
            commit_message=f"Add/Update model card for {persona} adapter"
        )
        print(f"Successfully uploaded README.md to {repo_id}")

        # Clean up the temporary file
        os.remove(temp_readme_path)

    except Exception as e:
        print(f"Error creating/uploading README.md for {repo_id}: {e}")
        # Attempt to clean up temp file even if upload fails
        if 'temp_readme_path' in locals() and os.path.exists(temp_readme_path):
            try:
                os.remove(temp_readme_path)
            except OSError:
                pass # Ignore cleanup error if file is already gone or locked

print("\n--- All personas processed ---") 