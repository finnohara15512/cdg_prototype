"""
RAG Demo Script - Vectorized Search with Discharge Summaries
Creates a vectorized database from discharge summary table and provides
an interactive terminal interface for querying specific encounter IDs.

Usage:
1. Install dependencies: pip install -r requirements.txt
2. Ensure Databricks connection is configured with profile "foha-auge"
3. Run: python rag_demo.py
4. Enter encounter IDs when prompted, type 'list' to see available IDs
5. Ask questions about the discharge summaries for specific encounters
"""

import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from litellm import completion
from databricks.connect import DatabricksSession
from dotenv import load_dotenv
import os
load_dotenv()

# Initialize Databricks connection
spark = DatabricksSession.builder.profile("foha-auge").getOrCreate()

class RAGDemo:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chroma_client = chromadb.Client()
        self.collection = None
        self.discharge_data = None
        
    def load_discharge_data(self):
        """Load discharge summary data from Databricks"""
        print("Loading discharge summary data...")
        discharge_summary = spark.sql("SELECT * FROM cdg_mvp_data.gold.discharge_summary")
        self.discharge_data = discharge_summary.toPandas()
        print(f"Loaded {len(self.discharge_data)} discharge summaries")
        
        # Display column information for debugging
        print(f"Columns: {list(self.discharge_data.columns)}")
        print(f"Sample encounter IDs: {self.discharge_data['filename_encounter_id'].unique()[:5] if 'filename_encounter_id' in self.discharge_data.columns else 'filename_encounter_id column not found'}")
        
        return self.discharge_data
    
    def create_vector_database(self):
        """Create vector database with embeddings grouped by filename_encounter_id"""
        print("Creating vector database...")
        
        if self.discharge_data is None or self.discharge_data.empty:
            raise ValueError("No discharge data loaded")
        
        # Verify required column exists
        if 'filename_encounter_id' not in self.discharge_data.columns:
            raise ValueError("Column 'filename_encounter_id' not found in discharge data")
        
        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="discharge_summaries",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Identify text columns (excluding ID and metadata columns)
        text_columns = []
        exclude_patterns = ['id', 'encounter', 'filename', 'date', 'time', 'created', 'updated']
        
        for col in self.discharge_data.columns:
            col_lower = col.lower()
            # Include columns that likely contain text content
            if (self.discharge_data[col].dtype == 'object' and 
                not any(pattern in col_lower for pattern in exclude_patterns) and
                col != 'filename_encounter_id'):
                text_columns.append(col)
        
        print(f"Using text columns for vectorization: {text_columns}")
        
        # Process documents by encounter_id
        documents = []
        metadatas = []
        ids = []
        
        # Group by filename_encounter_id and combine text content
        grouped_data = self.discharge_data.groupby('filename_encounter_id')
        
        for encounter_id, group in grouped_data:
            # Combine text from all text columns for this encounter
            text_parts = []
            
            for _, row in group.iterrows():
                for col in text_columns:
                    if pd.notna(row[col]) and str(row[col]).strip():
                        text_parts.append(str(row[col]).strip())
            
            if text_parts:
                # Join all text content for this encounter
                combined_text = ' '.join(text_parts)
                
                documents.append(combined_text)
                metadatas.append({"encounter_id": str(encounter_id)})
                ids.append(f"doc_{encounter_id}")
        
        # Add documents to collection with embeddings
        if documents:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Added {len(documents)} documents to vector database")
        else:
            print("No documents found to add to vector database")
    
    def search_by_encounter_id(self, encounter_id, query, top_k=3):
        """Search vector database filtered by encounter_id"""
        if not self.collection:
            raise ValueError("Vector database not initialized")
        
        # Search with encounter_id filter
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where={"encounter_id": str(encounter_id)}
        )
        
        return results
    
    def generate_response(self, encounter_id, user_query):
        """Generate LLM response using RAG"""
        try:
            # Search for relevant context
            search_results = self.search_by_encounter_id(encounter_id, user_query)
            
            if not search_results['documents'][0]:
                return f"No information found for encounter ID: {encounter_id}"
            
            # Prepare context from search results
            context = "\n".join(search_results['documents'][0])
            
            # Create prompt with context
            messages = [
                {
                    "role": "system",
                    "content": f"You are a medical assistant. Answer the user's question based on the following discharge summary context for encounter ID {encounter_id}:\n\n{context}"
                },
                {
                    "role": "user",
                    "content": user_query
                }
            ]
            
            # Generate response using litellm
            response = completion(
                model="openai/gpt-4o",
                messages=messages,
                stream=False,
                api_key=os.getenv("OPENAI_API_KEY"),
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def get_available_encounter_ids(self):
        """Get list of available encounter IDs"""
        if self.discharge_data is None:
            return []
        return self.discharge_data['filename_encounter_id'].unique().tolist()
    
    def validate_encounter_id(self, encounter_id):
        """Validate if encounter ID exists in the dataset"""
        available_ids = self.get_available_encounter_ids()
        return encounter_id in [str(id) for id in available_ids]
    
    def interactive_loop(self):
        """Main interactive terminal loop"""
        print("\n=== RAG Demo Interactive Interface ===")
        print("Enter 'quit' or 'exit' to stop the program")
        print("Type 'list' to see available encounter IDs")
        
        while True:
            try:
                # Get encounter ID
                encounter_id = input("\nEnter encounter ID: ").strip()
                if encounter_id.lower() in ['quit', 'exit']:
                    break
                
                if encounter_id.lower() == 'list':
                    available_ids = self.get_available_encounter_ids()
                    if available_ids:
                        print(f"\nAvailable encounter IDs ({len(available_ids)} total):")
                        # Show first 10 IDs
                        for id in available_ids[:10]:
                            print(f"  {id}")
                        if len(available_ids) > 10:
                            print(f"  ... and {len(available_ids) - 10} more")
                    else:
                        print("No encounter IDs found")
                    continue
                
                if not encounter_id:
                    print("Please enter a valid encounter ID")
                    continue
                
                # Validate encounter ID exists
                if not self.validate_encounter_id(encounter_id):
                    print(f"Encounter ID '{encounter_id}' not found in database")
                    print("Type 'list' to see available encounter IDs")
                    continue
                
                # Get user query
                user_query = input("Enter your question: ").strip()
                if user_query.lower() in ['quit', 'exit']:
                    break
                
                if not user_query:
                    print("Please enter a valid question")
                    continue
                
                print("\nGenerating response...")
                response = self.generate_response(encounter_id, user_query)
                print(f"\nResponse:\n{response}")
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {str(e)}")
                print("Please try again or type 'quit' to exit")

def main():
    """Main function to run the RAG demo"""
    rag_demo = RAGDemo()
    
    try:
        # Load data and create vector database
        rag_demo.load_discharge_data()
        rag_demo.create_vector_database()
        
        # Start interactive loop
        rag_demo.interactive_loop()
        
    except Exception as e:
        print(f"Error initializing RAG demo: {str(e)}")
        print("Please check your Databricks connection and data access")

if __name__ == "__main__":
    main()