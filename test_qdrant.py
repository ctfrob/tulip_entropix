from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv

load_dotenv()

def test_qdrant_connection():
    try:
        url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")
        
        print(f"Attempting to connect to Qdrant at: {url}")
        client = QdrantClient(url=url, api_key=api_key)
        
        # Try to get collection list - a simple operation
        collections = client.get_collections()
        print("Successfully connected to Qdrant!")
        print(f"Available collections: {collections}")
        
        return True
    except Exception as e:
        print(f"Connection failed with error: {e}")
        return False

if __name__ == "__main__":
    test_qdrant_connection()