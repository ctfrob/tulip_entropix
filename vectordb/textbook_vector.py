from pathlib import Path
import torch
from transformers import AutoModel, AutoTokenizer
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
    def __init__(self, data_dir: str = "data/textbooks", chunk_size: int = 512):
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

        return torch.cat(embeddings)

class QdrantLoader:
    def __init__(self, url: str, api_key: str, collection_name: str):
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name

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
    embedder = TextbookEmbedder()
    documents, processed_docs = embedder.load_and_process_textbooks()
    dataset_embeddings = embedder.get_dataset_embeddings(processed_docs)
    final_embeddings = embedder.generate_embeddings(processed_docs, dataset_embeddings)

    qdrant_loader = QdrantLoader(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        collection_name="textbooks"
    )
    qdrant_loader.create_collection(final_embeddings.shape[1])
    qdrant_loader.upload_embeddings(documents, final_embeddings)