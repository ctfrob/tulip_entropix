from pathlib import Path
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
import numpy as np
from openai import OpenAI

load_dotenv()

class TextbookSearcher:
    def __init__(self):
        print("Initializing OpenAI client...")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        print("Connecting to Qdrant...")
        self.client_qdrant = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
    
    def encode_query(self, query: str):
        response = self.client.embeddings.create(
            input=query,
            model="text-embedding-3-large"
        )
        return np.array(response.data[0].embedding)
    
    def search(self, query: str, limit: int = 3):
        print(f"\nSearching for: '{query}'")
        query_vector = self.encode_query(query)
        
        results = self.client_qdrant.search(
            collection_name="textbooks_oaiembed",  # Updated collection name
            query_vector=query_vector.tolist(),
            limit=limit
        )
        
        print(f"\nTop {limit} results:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Score: {result.score:.4f}")
            print(f"Source: {result.payload['metadata']['source']}")
            print(f"Preview: {result.payload['text'][:150]}...")
        
        return results
    
    def close(self):
        self.client_qdrant.close()

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