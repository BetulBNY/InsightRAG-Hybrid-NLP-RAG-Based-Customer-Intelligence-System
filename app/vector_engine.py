import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
import os

class VectorEngine:
    def __init__(self, db_path="./chroma_db_v2"):

        # 1. DATABASE CREATION:
        # I specify where the database will be saved
        self.client = chromadb.PersistentClient(path=db_path)
        
        # 2. EMBEDDING MODEL:
        # I choose the embedding model (This model converts text into a 384-dimensional vector):
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # 3. COLLECTION CREATION:
        # I give embedding_function as a parameter to the collection, so that when I upsert data, it will automatically convert the documents to vectors using this embedding function and store them in the collection.
        self.collection = self.client.get_or_create_collection(
            name="customer_reviews",
            embedding_function=self.emb_fn
        )

    def upsert_reviews(self, df):
        """ Insert or update the analyzed data into the Vector DB. """
        print(f"{len(df)} number of reviews will be upserted to the vector database.")
        
        # ChromaDB expects us to provide the following:
        # ids (string), documents (text), metadatas (addiitonal info)
        self.collection.upsert(  # this will insert or update the data in the collection also converts the documents to vectors using the embedding function
            ids=[str(i) for i in df.index],
            documents=df['full_text'].tolist(),
            metadatas=[{  # With metadata I can filter the search results later, for example I can filter by department or by rating. So I can get more relevant results.
                "rating": int(row['Rating']),
                "sentiment": row['sentiment_label'],
                "consistency": float(row['consistency_score']),
                "is_shouting": bool(row['is_shouting']),
                "is_intense": bool(row['is_intense']), 
                "department": str(row['Department Name']) 
            } for _, row in df.iterrows()]
        )
        print("Vector database updated successfully!")

        # Note: If data with the same ID is added, the old data will be updated (upsert = update + insert)

    def search_reviews(self, query, n_results=5, min_rating=None):
        """
        Performs semantic search.
        With the min_rating parameter, I can get only results that have a certain rating and above. For example, if I set min_rating=4, it will only return reviews with 4 and 5 stars. This way I can get more relevant results.
        """
        where_clause = None
        if min_rating:
            where_clause = {"rating": {"$gte": min_rating}} # Just bring min_rating and above reviews

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_clause
        )
        return results

if __name__ == "__main__":
    v_engine = VectorEngine()
    
    if os.path.exists('data/analyzed_reviews.csv'):
        df = pd.read_csv('data/analyzed_reviews.csv')
        v_engine.upsert_reviews(df)
        
        # Sample semantic search: "sizing and fit issues"
        search_res = v_engine.search_reviews("sizing and fit issues")
        print("\n--- SEARCH RESULTS ---")
        for i, doc in enumerate(search_res['documents'][0]):
            meta = search_res['metadatas'][0][i]
            print(f"-> Rating: {meta['rating']} | Sentiment: {meta['sentiment']} | Review: {doc[:300]}...") 
        
            
    else:
        print("Error: 'data/analyzed_reviews.csv' not found! nlp_engine.py should be run first.") 
    