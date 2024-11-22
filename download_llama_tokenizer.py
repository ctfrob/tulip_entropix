import os
from pathlib import Path
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import shutil
import sys

def main():
    save_dir = Path("entropix")
    model_id = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
    
    # Clear existing directory if it exists
    if (save_dir / "tokenizer.model").exists():
        shutil.rmtree(save_dir / "tokenizer.model")
    
    try:
        # Download and save complete tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=os.getenv("HF_TOKEN"),
            trust_remote_code=True
        )
        
        # Save all tokenizer files
        tokenizer.save_pretrained(save_dir / "tokenizer.model")
        print(f"Successfully saved tokenizer to {save_dir}/tokenizer.model")
        
    except Exception as e:
        print(f"Error downloading tokenizer: {e}")
        print("\nPlease ensure:")
        print("1. HF_TOKEN environment variable is set")
        print("2. You have access to the model repository")
        sys.exit(1)

if __name__ == "__main__":
    main()