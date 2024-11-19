from pathlib import Path
import torch
from transformers import AutoModel, AutoTokenizer
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

class TextbookSearcher:
    def __init__(self):
        print("Initializing search model...")
        self.model = AutoModel.from_pretrained("jxm/cde-small-v1", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        print("Loading dataset embeddings...")
        self.dataset_embeddings = torch.load("vectordb/data/dataset_embeddings.pt")
        self.dataset_embeddings = self.dataset_embeddings.to(self.device)

        print("Connecting to Qdrant...")
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
    
    def encode_query(self, query: str):
        inputs = self.tokenizer(f"search_document: {query}",
                         return_tensors="pt",
                         truncation=True,
                         max_length=512).to(self.device)
        
        with torch.no_grad():
            embeddings = self.model.second_stage_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                dataset_embeddings=self.dataset_embeddings
            )
            embeddings /= embeddings.norm(p=2, dim=1, keepdim=True)
        
        return embeddings[0].cpu().numpy()
    
    def search(self, query: str, limit: int = 3):
        print(f"\nSearching for: '{query}'")
        query_vector = self.encode_query(query)
        
        results = self.client.search(
            collection_name="textbooks",
            query_vector=query_vector,
            limit=limit
        )
        
        print(f"\nTop {limit} results:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Score: {result.score:.4f}")
            print(f"Source: {result.payload['metadata']['source']}")
            print(f"Preview: {result.payload['text'][:150]}...")
        
        return results
    
    def close(self):
        self.client.close()

def interactive_search():
    searcher = TextbookSearcher()
    
    try:
        while True:
            query = input("\nEnter your search query (or 'quit' to exit): ")
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            searcher.search(query)
            
    except KeyboardInterrupt:
        print("\nSearch session ended by user")
    finally:
        searcher.close()
        print("Connection closed")

if __name__ == "__main__":
    interactive_search()