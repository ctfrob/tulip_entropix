from pathlib import Path
import torch
from transformers import AutoModel, AutoTokenizer
import pickle
import os
import random
from tqdm import tqdm
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


load_dotenv()

class TextbookVectorDB:
    def __init__(self):
        self.model = AutoModel.from_pretrained("jxm/cde-small-v1", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.qdrant = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )

class TextbookEmbedder:
    def __init__(self, data_dir: str = "vectordb/data/textbooks", chunk_size: int = 512):
        self.data_dir = Path(data_dir)
        self.chunk_size = chunk_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model and tokenizer
        self.model = AutoModel.from_pretrained("jxm/cde-small-v1", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model.to(self.device)

    def load_and_process_textbooks(self):
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

        document_prefix = "search_document: "
        return documents, [document_prefix + doc['text'] for doc in documents]

    def get_dataset_embeddings(self, documents, batch_size=32):
        corpus_size = self.model.config.transductive_corpus_size
        minicorpus_docs = random.choices(documents, k=corpus_size)
        
        tokenized = self.tokenizer(
            minicorpus_docs,
            truncation=True,
            padding=True,
            max_length=self.chunk_size,
            return_tensors="pt"
        ).to(self.device)

        dataset_embeddings = []
        for i in tqdm(range(0, len(tokenized["input_ids"]), batch_size)):
            batch = {k: v[i:i+batch_size] for k,v in tokenized.items()}
            with torch.no_grad():
                dataset_embeddings.append(
                    self.model.first_stage_model(**batch)
                )
        
        return torch.cat(dataset_embeddings)
    
    def save_dataset_embeddings(self, dataset_embeddings, save_path="vectordb/data/dataset_embeddings.pt"):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dataset_embeddings, save_path)
        print(f"Saved dataset embeddings to {save_path}")

    def load_dataset_embeddings(self, load_path="vectordb/data/dataset_embeddings.pt"):
        if not Path(load_path).exists():
            raise FileNotFoundError(f"Dataset embeddings file not found at {load_path}")
        dataset_embeddings = torch.load(load_path)
        print(f"Loaded dataset embeddings from {load_path}")
        return dataset_embeddings

    def generate_embeddings(self, documents, dataset_embeddings, batch_size=32):
        tokenized = self.tokenizer(
            documents,
            truncation=True,
            padding=True,
            max_length=self.chunk_size,
            return_tensors="pt"
        ).to(self.device)

        embeddings = []
        for i in tqdm(range(0, len(tokenized["input_ids"]), batch_size)):
            batch = {k: v[i:i+batch_size] for k,v in tokenized.items()}
            with torch.no_grad():
                batch_embeddings = self.model.second_stage_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    dataset_embeddings=dataset_embeddings
                )
                batch_embeddings /= batch_embeddings.norm(p=2, dim=1, keepdim=True)
                embeddings.append(batch_embeddings)

        final_embeddings = torch.cat(embeddings)
        print(f"Generated embeddings of shape {final_embeddings.shape}")
        return final_embeddings
    
    def save_embeddings(self, documents, embeddings, save_dir: str = "vectordb/embeddings"):
            """Save both documents and their embeddings to disk"""
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save documents
            with open(save_dir / "documents.pkl", "wb") as f:
                pickle.dump(documents, f)
            
            # Save embeddings
            torch.save(embeddings, save_dir / "embeddings.pt")
            
            print(f"Saved {len(documents)} documents and embeddings of shape {embeddings.shape} to {save_dir}")

class QdrantLoader:
    def __init__(self, url: str, api_key: str, collection_name: str):
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def close(self):
        """Close the Qdrant client connection"""
        try:
            self.client.close()
            print("Qdrant client closed successfully")
        except Exception as e:
            print(f"Failed to close Qdrant client: {e}")

    def create_collection(self, vector_size: int):
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )

    def upload_embeddings(self, documents, embeddings, batch_size=100):
        for i in range(0, len(documents), batch_size):
            batch_documents = documents[i:i+batch_size]
            batch_embeddings = embeddings[i:i+batch_size].cpu().numpy()
            
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
        with open(save_dir / "documents.pkl", "rb") as f:
            documents = pickle.load(f)
        final_embeddings = torch.load(save_dir / "embeddings.pt")
        print(f"Loaded {len(documents)} documents and embeddings of shape {final_embeddings.shape}")
    else:
        print("Generating new embeddings...")
        embedder = TextbookEmbedder()
        documents, processed_docs = embedder.load_and_process_textbooks()
        dataset_embeddings = embedder.get_dataset_embeddings(processed_docs)
        embedder.save_dataset_embeddings(dataset_embeddings)
        final_embeddings = embedder.generate_embeddings(processed_docs, dataset_embeddings)
        embedder.save_embeddings(documents, final_embeddings)
    
    try:
        with QdrantLoader(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            collection_name="textbooks"
        ) as qdrant_loader:
            print(f"Recreating collection 'textbooks'...")
            try:
                qdrant_loader.client.delete_collection("textbooks")
                print("Deleted existing collection")
            except Exception as e:
                print(f"No existing collection to delete: {e}")
                
            qdrant_loader.create_collection(final_embeddings.shape[1])
            print("Created new collection")
            
            print("Starting upload to Qdrant...")
            qdrant_loader.upload_embeddings(documents, final_embeddings)
            print("Successfully uploaded embeddings to Qdrant")
            
    except Exception as e:
        print(f"Failed to upload to Qdrant: {e}")
        print("But don't worry - embeddings were saved locally and can be uploaded later")