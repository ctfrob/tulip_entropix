import tiktoken
import os
from pathlib import Path
import requests
import shutil

def main():
    # Get the base tokenizer
    encoding = tiktoken.get_encoding("cl100k_base")
    
    # URL for the cl100k_base tokenizer
    tokenizer_url = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
    
    # Create target directory
    save_path = Path("entropix/tokenizer.model")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Download the tokenizer file
    print("Downloading tokenizer...")
    response = requests.get(tokenizer_url)
    response.raise_for_status()  # Raise an error for bad responses
    
    # Save the file
    with open(save_path, 'wb') as f:
        f.write(response.content)
    
    print(f"Successfully downloaded tokenizer to {save_path}")

if __name__ == "__main__":
    main()