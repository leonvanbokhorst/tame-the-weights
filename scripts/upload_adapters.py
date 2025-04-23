import os
import dotenv
from huggingface_hub import HfApi, upload_folder

# --- Configuration ---
BASE_MODEL_ID = "microsoft/Phi-4-mini-instruct"
PERSONAS = ["captain_codebeard", "professor_snugglesworth", "zen_coder"]
LOCAL_ADAPTERS_DIR = "trained_adapters"
# Assumes your Hugging Face username is leonvanbokhorst based on previous context
HF_USERNAME = "leonvanbokhorst"

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

    print(f"Local adapter path: {local_adapter_path}")
    print(f"Target Hugging Face repo ID: {repo_id}")

    # Check if local adapter directory exists
    if not os.path.isdir(local_adapter_path):
        print(f"Error: Local adapter directory not found: {local_adapter_path}")
        print("Skipping this persona.")
        continue

    # Upload the folder
    try:
        print(f"Uploading {persona} adapter to {repo_id}...")
        upload_folder(
            folder_path=local_adapter_path,
            repo_id=repo_id,
            repo_type="model", # Specify repository type as 'model'
            token=hf_token,    # Pass the token explicitly
            commit_message=f"Upload {persona} adapter for {BASE_MODEL_ID}",
            create_pr=True,    # Suggest creating a Pull Request if repo exists
        )
        print(f"Successfully uploaded {persona} adapter to {repo_id}")

    except Exception as e:
        print(f"Error uploading {persona} adapter: {e}")
        print("Please check your token, permissions, and repository status on Hugging Face Hub.")

print("\n--- All personas processed ---") 