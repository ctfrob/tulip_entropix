from pathlib import Path
import pickle
import os
from typing import List, Dict, Any
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
import time

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()

class TextbookEmbedder:
    def __init__(self, 
                data_dir: str = "vectordb/data/textbooks", 
                chunk_size: int = 2048,
                batch_size: int = 100):
        self.data_dir = Path(data_dir)
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.vector_dim = 3072  # text-embedding-3-large dimension
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def load_and_process_textbooks(self) -> List[Dict[str, Any]]:
        """Load and chunk textbook files into documents"""
        textbook_files = list(self.data_dir.glob("*.txt"))
        documents = []
        
        for file in textbook_files:
            with open(file) as f:
                text = f.read()
                chunks = text.split("\n\n")
                documents.extend([{
                    'text': chunk,
                    'metadata': {'source': str(file.name)}
                } for chunk in chunks if chunk.strip()])
        
        return documents

    def generate_embeddings(self, documents: List[Dict]) -> np.ndarray:
        """Generate embeddings using OpenAI API with batching and rate limiting"""
        embeddings = []
        
        for i in tqdm(range(0, len(documents), self.batch_size)):
            batch = documents[i:i + self.batch_size]
            batch_texts = [doc['text'] for doc in batch]
            
            try:
                response = self.client.embeddings.create(
                    input=batch_texts,
                    model="text-embedding-3-large"
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
                # Rate limiting - sleep between batches
                time.sleep(0.5)  # Adjust based on API limits
                
            except Exception as e:
                print(f"Error in batch {i}: {e}")
                # Implement retry logic if needed
                raise
        
        return np.array(embeddings)

    def save_embeddings(self, 
                       documents: List[Dict], 
                       embeddings: np.ndarray, 
                       save_dir: str = "vectordb/embeddings_oaiembed"):
        """Save documents and embeddings to disk"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        with open(save_dir / "documents_oaiembed.pkl", "wb") as f:
            pickle.dump(documents, f)
        
        np.save(save_dir / "embeddings_oaiembed.npy", embeddings)
        print(f"Saved {len(documents)} documents and embeddings of shape {embeddings.shape} to {save_dir}")

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