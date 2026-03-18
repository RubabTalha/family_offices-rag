import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import pickle
import os

print("Loading your family office dataset...")
df = pd.read_csv('family_offices.csv')
print(f"✅ Loaded {len(df)} rows with {len(df.columns)} columns")

# Create rich text descriptions for each family office
print("\nCreating rich text descriptions...")
documents = []

for idx, row in df.iterrows():
    # Skip if firm name is missing
    if pd.isna(row.get('FO Firm Name')):
        continue
    
    # Build a rich text description combining all fields
    description = f"""
Family Office: {row.get('FO Firm Name', 'N/A')}
Website: {row.get('Website', 'N/A')}
Location: {row.get('Country', 'N/A')}, {row.get('City', 'N/A')}
Founded: {row.get('Founded Year', 'N/A')}
Type: {row.get('Type', 'N/A')}

Investment Focus: {row.get('Investment Focus', 'N/A')}
Check Size: ${row.get('Check Size Min', 'N/A')} - ${row.get('Check Size Max', 'N/A')}
AUM Range: {row.get('AUM Range', 'N/A')}

Recent Deals:
• {row.get('Recent Deal 1', 'N/A')}
• {row.get('Recent Deal 2', 'N/A')}
• {row.get('Recent Deal 3', 'N/A')}

Co-Investors: {row.get('Co-Investors', 'N/A')}
LP Relations: {row.get('LP Relations', 'N/A')}

Decision Maker: {row.get('Contact Name', 'N/A')} - {row.get('Contact Title', 'N/A')}
Contact: {row.get('Contact Email', 'N/A')}
LinkedIn: {row.get('Contact LinkedIn', 'N/A')}

Recent News: {row.get('Recent News', 'N/A')} ({row.get('News Date', 'N/A')})
News Source: {row.get('News Source', 'N/A')}
LinkedIn Activity: {row.get('LinkedIn Activity', 'N/A')}

Verification: {row.get('Verified Method', 'N/A')} on {row.get('Verified Date', 'N/A')}
Source: {row.get('Source URL', 'N/A')}
Notes: {row.get('Notes', 'N/A')}
"""
    
    documents.append({
        'id': f"fo_{idx}",
        'text': description,
        'metadata': {
            'firm_name': str(row.get('FO Firm Name', '')),
            'country': str(row.get('Country', '')),
            'investment_focus': str(row.get('Investment Focus', '')),
            'check_size_min': str(row.get('Check Size Min', '')),
            'check_size_max': str(row.get('Check Size Max', '')),
            'type': str(row.get('Type', '')),
            'city': str(row.get('City', ''))
        }
    })

print(f"✅ Created {len(documents)} rich documents")

# Load embedding model
print("\nLoading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings for all documents
print("Creating embeddings (this may take a few minutes)...")
texts = [doc['text'] for doc in documents]
embeddings = model.encode(texts, show_progress_bar=True)

# Initialize ChromaDB
print("\nSetting up vector database...")
client = chromadb.PersistentClient(path="./chroma_db")

# Create or get collection
collection_name = "family_offices"
try:
    client.delete_collection(collection_name)
    print(f"Deleted existing collection: {collection_name}")
except:
    pass

collection = client.create_collection(
    name=collection_name,
    metadata={"hnsw:space": "cosine"}
)

# Add documents to collection
print("Adding documents to vector database...")
batch_size = 100
for i in range(0, len(documents), batch_size):
    batch = documents[i:i+batch_size]
    batch_embeddings = embeddings[i:i+batch_size]
    
    collection.add(
        ids=[doc['id'] for doc in batch],
        documents=[doc['text'] for doc in batch],
        embeddings=[emb.tolist() for emb in batch_embeddings],
        metadatas=[doc['metadata'] for doc in batch]
    )
    print(f"Added batch {i//batch_size + 1}/{(len(documents)//batch_size)+1}")

print(f"\n✅ Successfully added {len(documents)} documents to ChromaDB")

# Save metadata for later use
with open('documents.pkl', 'wb') as f:
    pickle.dump(documents, f)

print("\n✅ Knowledge base created successfully!")
print(f"Files created:")
print(f"  - chroma_db/ (vector database)")
print(f"  - documents.pkl (document metadata)")