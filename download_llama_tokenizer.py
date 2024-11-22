import os
from pathlib import Path
import requests
import shutil
from huggingface_hub import hf_hub_download
import sys

def download_from_hf():
    """Download tokenizer from Hugging Face Hub"""
    try:
        # Attempt to download from Hugging Face
        tokenizer_path = hf_hub_download(
            repo_id="nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
            filename="tokenizer.model",
            token=os.getenv("HF_TOKEN")
        )
        return tokenizer_path
    except Exception as e:
        print(f"Failed to download from Hugging Face: {e}")
        return None

def copy_from_local(llama_path: str):
    """Copy tokenizer from local LLaMA installation"""
    local_tokenizer = Path(llama_path) / "tokenizer.model"
    if local_tokenizer.exists():
        return str(local_tokenizer)
    return None

def main():
    save_path = Path("entropix/tokenizer.model")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        print("Downloading Llama-3.1-Nemotron-70B-Instruct-HF tokenizer...")
        tokenizer_path = hf_hub_download(
            repo_id="nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
            filename="tokenizer.json",
            token=os.getenv("HF_TOKEN")
        )
        
        # Copy to destination
        shutil.copy2(tokenizer_path, save_path)
        print(f"Successfully downloaded tokenizer to {save_path}")
        
    except Exception as e:
        print(f"Error downloading tokenizer: {e}")
        print("Please ensure:")
        print("1. You have huggingface_hub installed: pip install huggingface_hub")
        print("2. You have access to the model repo")
        print("3. HF_TOKEN is set if the repo is private: export HF_TOKEN=your_token")
        sys.exit(1)

if __name__ == "__main__":
    main()