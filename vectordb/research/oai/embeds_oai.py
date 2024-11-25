from pathlib import Path
import pickle
import os
from typing import List, Dict, Any, Tuple
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIError
from tqdm import tqdm
import time
import json
import backoff
import httpx
import tiktoken

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()

class TextbookEmbedder:
    def __init__(self, 
                data_dir: str = "vectordb/data/textbooks", 
                chunk_size: int = 8000,
                max_batch_tokens: int = 7800):  # Close to limit but safe
        self.data_dir = Path(data_dir)
        self.chunk_size = chunk_size
        self.max_batch_tokens = max_batch_tokens
        self.vector_dim = 3072
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.checkpoint_dir = Path("vectordb/checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, texts: List[str]) -> int:
        if not texts:
            return 0
        return len(self.tokenizer.encode("".join(texts)))

    def create_dynamic_batch(self, documents: List[Dict], start_idx: int) -> Tuple[List[Dict], int]:
        batch = []
        current_texts = []
        idx = start_idx
        
        while idx < len(documents):
            doc = documents[idx]
            current_texts.append(doc['text'])
            
            # Check token count less frequently
            if len(batch) % 5 == 0:
                total_tokens = self.count_tokens(current_texts)
                if total_tokens > self.max_batch_tokens:
                    current_texts.pop()
                    break
                
            batch.append(doc)
            idx += 1
            
            # Cap batch size generously
            if len(batch) >= 30:
                break
        
        return batch, idx

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        try:
            response = self.client.embeddings.create(
                input=texts,
                model="text-embedding-3-large"
            )
            return [item.embedding for item in response.data]
        except (RateLimitError, APIError) as e:
            print(f"API Error: {e}")
            time.sleep(1)  # Brief pause on rate limit
            raise

    def generate_embeddings(self, documents: List[Dict]) -> np.ndarray:
        processed_docs, processed_embeddings, start_index = self.load_checkpoint()
        
        if start_index >= len(documents):
            return np.array(processed_embeddings)

        print(f"Starting from index {start_index}/{len(documents)}")
        start_time = time.time()
        
        try:
            with tqdm(total=len(documents), initial=start_index) as pbar:
                idx = start_index
                while idx < len(documents):
                    batch, next_idx = self.create_dynamic_batch(documents, idx)
                    
                    try:
                        batch_start = time.time()
                        batch_texts = [doc['text'] for doc in batch]
                        batch_embeddings = self.get_embeddings(batch_texts)
                        
                        processed_docs.extend(batch)
                        processed_embeddings.extend(batch_embeddings)
                        
                        # Stats
                        batch_time = time.time() - batch_start
                        total_time = time.time() - start_time
                        rate = len(processed_docs) / total_time if total_time > 0 else 0
                        
                        # Save checkpoint every 1000 documents
                        if len(processed_docs) % 1000 == 0:
                            self.save_checkpoint(processed_docs, processed_embeddings, next_idx)
                        
                        print(f"\nProcessed batch of {len(batch)} docs in {batch_time:.1f}s ({len(batch)/batch_time:.1f} docs/s)")
                        print(f"Overall rate: {rate:.1f} docs/s")
                        
                        pbar.update(next_idx - idx)
                        idx = next_idx
                        
                    except Exception as e:
                        print(f"\nError at batch {idx}: {e}")
                        self.save_checkpoint(processed_docs, processed_embeddings, idx)
                        idx += 1
                
        except KeyboardInterrupt:
            print("\nSaving progress before exit...")
            self.save_checkpoint(processed_docs, processed_embeddings, idx)
            raise
            
        # Final save
        self.save_checkpoint(processed_docs, processed_embeddings, len(documents))
        return np.array(processed_embeddings)

    def load_checkpoint(self) -> Tuple[List[Dict], List[List[float]], int]:
        checkpoint_file = self.checkpoint_dir / "embedding_checkpoint.json"
        if checkpoint_file.exists():
            with open(checkpoint_file) as f:
                checkpoint = json.load(f)
                return checkpoint['documents'], checkpoint['embeddings'], checkpoint['last_index']
        return [], [], 0

    def save_checkpoint(self, documents: List[Dict], embeddings: List[List[float]], last_index: int):
        checkpoint = {
            'documents': documents,
            'embeddings': embeddings,
            'last_index': last_index,
            'timestamp': time.time()
        }
        with open(self.checkpoint_dir / "embedding_checkpoint.json", 'w') as f:
            json.dump(checkpoint, f)

    def load_and_process_textbooks(self) -> List[Dict[str, Any]]:
        textbook_files = list(self.data_dir.glob("*.txt"))
        documents = []
        
        for file in textbook_files:
            with open(file) as f:
                text = f.read()
                current_chunk = []
                current_length = 0
                
                for paragraph in text.split("\n\n"):
                    paragraph = paragraph.strip()
                    if not paragraph:
                        continue
                        
                    if current_length + len(paragraph) > self.chunk_size:
                        if current_chunk:
                            documents.append({
                                'text': " ".join(current_chunk),
                                'metadata': {'source': str(file.name)}
                            })
                            current_chunk = [paragraph]
                            current_length = len(paragraph)
                    else:
                        current_chunk.append(paragraph)
                        current_length += len(paragraph)
                
                if current_chunk:
                    documents.append({
                        'text': " ".join(current_chunk),
                        'metadata': {'source': str(file.name)}
                    })
        
        print(f"Created {len(documents)} chunks from {len(textbook_files)} files")
        return documents

    def save_embeddings(self, 
                       documents: List[Dict], 
                       embeddings: np.ndarray, 
                       save_dir: str = "vectordb/embeddings_oaiembed"):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        with open(save_dir / "documents_oaiembed.pkl", "wb") as f:
            pickle.dump(documents, f)
        
        np.save(save_dir / "embeddings_oaiembed.npy", embeddings)
        print(f"Saved {len(documents)} documents and embeddings of shape {embeddings.shape}")

class QdrantLoader:
    def __init__(self, url: str, api_key: str, collection_name: str):
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name
        self.vector_dim = 3072  # text-embedding-3-large dimension

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        """Close the Qdrant client connection"""
        try:
            self.client.close()
            print("Qdrant client closed successfully")
        except Exception as e:
            print(f"Failed to close Qdrant client: {e}")

    def create_collection(self):
        """Create a new collection with the correct vector size"""
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE)
        )

    def upload_embeddings(self, documents: List[Dict], embeddings: np.ndarray, batch_size: int = 100):
        """Upload document embeddings to Qdrant in batches"""
        for i in range(0, len(documents), batch_size):
            batch_documents = documents[i:i+batch_size]
            batch_embeddings = embeddings[i:i+batch_size]
            
            points = [
                PointStruct(
                    id=idx,
                    vector=embedding.tolist(),
                    payload=doc
                )
                for idx, (doc, embedding) in enumerate(zip(batch_documents, batch_embeddings), start=i)
            ]
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-saved', action='store_true', 
                       help='Use saved embeddings instead of generating new ones')
    args = parser.parse_args()
    
    if args.use_saved:
        print("Loading saved embeddings...")
        save_dir = Path("vectordb/embeddings")
        with open(save_dir / "documents_oaiembed.pkl", "rb") as f:
            documents = pickle.load(f)
        embeddings = np.load(save_dir / "embeddings_oaiembed.npy")
        print(f"Loaded {len(documents)} documents and embeddings of shape {embeddings.shape}")
    else:
        print("Generating new embeddings...")
        embedder = TextbookEmbedder()
        documents = embedder.load_and_process_textbooks()
        embeddings = embedder.generate_embeddings(documents)
        embedder.save_embeddings(documents, embeddings)
    
    try:
        with QdrantLoader(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            collection_name="textbooks_oaiembed"
        ) as qdrant_loader:
            print(f"Recreating collection 'textbooks_oaiembed'...")
            try:
                qdrant_loader.client.delete_collection("textbooks_oaiembed")
                print("Deleted existing collection")
            except Exception as e:
                print(f"No existing collection to delete: {e}")
                
            qdrant_loader.create_collection()
            print("Created new collection")
            
            print("Starting upload to Qdrant...")
            qdrant_loader.upload_embeddings(documents, embeddings)
            print("Successfully uploaded embeddings to Qdrant")
            
    except Exception as e:
        print(f"Failed to upload to Qdrant: {e}")
        print("But don't worry - embeddings were saved locally and can be uploaded later")