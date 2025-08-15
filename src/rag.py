import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

class RAGSystem:
    def __init__(self, data_path='data/drug_side_effects.csv'):
        # 1. Load Data
        self.df = pd.read_csv(data_path)
        self.documents = (self.df['drug_name'] + ": " + self.df['description']).tolist()
        
        # 2. Initialize Sentence Transformer Model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 3. Create FAISS Vector Index
        self.embeddings = self.model.encode(self.documents, convert_to_tensor=True)
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings.numpy())
        print("RAG System Initialized.")

    def search(self, query: str, k: int = 3):
        """
        Searches the vector index for the most relevant documents.
        """
        query_embedding = self.model.encode([query], convert_to_tensor=True)
        distances, indices = self.index.search(query_embedding.numpy(), k)
        
        # Retrieve the documents based on indices
        results = [self.documents[i] for i in indices[0]]
        return results

# Example of how to use it
if __name__ == '__main__':
    rag = RAGSystem()
    query = "What are the side effects of Paracetamol?"
    retrieved_docs = rag.search(query)
    
    print(f"Query: {query}")
    print("\nRetrieved Documents:")
    for doc in retrieved_docs:
        print(f"- {doc}")
