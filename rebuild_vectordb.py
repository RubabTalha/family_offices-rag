import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import pickle
import os

print("🚀 Rebuilding vector database from scratch...")

# Load your dataset
print("📊 Loading family office dataset...")
df = pd.read_csv('family_offices.csv')
print(f"✅ Loaded {len(df)} rows")

# Create rich text descriptions
print("📝 Creating rich text descriptions...")
documents = []

def safe_str(val):
    if pd.isna(val) or val is None:
        return ''
    return str(val)

for idx, row in df.iterrows():
    if pd.isna(row.get('FO Firm Name')):
        continue
    
    # Build rich description with ALL fields
    description = f"""
Family Office: {safe_str(row.get('FO Firm Name'))}
Website: {safe_str(row.get('Website'))}
Location: {safe_str(row.get('Country'))}, {safe_str(row.get('City'))}
Founded: {safe_str(row.get('Founded Year'))}
Type: {safe_str(row.get('Type'))}

Investment Focus: {safe_str(row.get('Investment Focus'))}
Check Size: ${safe_str(row.get('Check Size Min'))} - ${safe_str(row.get('Check Size Max'))}M
AUM Range: {safe_str(row.get('AUM Range'))}

Recent Deals:
• {safe_str(row.get('Recent Deal 1'))}
• {safe_str(row.get('Recent Deal 2'))}
• {safe_str(row.get('Recent Deal 3'))}

Co-Investors: {safe_str(row.get('Co-Investors'))}
LP Relations: {safe_str(row.get('LP Relations'))}

Decision Maker: {safe_str(row.get('Contact Name'))} - {safe_str(row.get('Contact Title'))}
Contact: {safe_str(row.get('Contact Email'))}
LinkedIn: {safe_str(row.get('Contact LinkedIn'))}

Recent News: {safe_str(row.get('Recent News'))} ({safe_str(row.get('News Date'))})
News Source: {safe_str(row.get('News Source'))}

Verification: {safe_str(row.get('Verified Method'))} on {safe_str(row.get('Verified Date'))}
Source: {safe_str(row.get('Source URL'))}
Notes: {safe_str(row.get('Notes'))}
"""
    
    documents.append({
        'id': f"fo_{idx}",
        'text': description,
        'metadata': {
            'firm_name': safe_str(row.get('FO Firm Name')),
            'country': safe_str(row.get('Country')),
            'investment_focus': safe_str(row.get('Investment Focus')),
            'type': safe_str(row.get('Type')),
            'city': safe_str(row.get('City'))
        }
    })

print(f"✅ Created {len(documents)} rich documents")

# Load embedding model
print("🧠 Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings
print("🔢 Creating embeddings (this takes 2-3 minutes)...")
texts = [doc['text'] for doc in documents]
embeddings = model.encode(texts, show_progress_bar=True)

# DELETE existing ChromaDB and recreate
print("🗑️ Removing old vector database...")
import shutil
if os.path.exists("./chroma_db"):
    shutil.rmtree("./chroma_db")

# Initialize fresh ChromaDB
print("🆕 Creating fresh vector database...")
client = chromadb.PersistentClient(path="./chroma_db")

# Create new collection
collection = client.create_collection(
    name="family_offices",
    metadata={"hnsw:space": "cosine"}
)

# Add documents in batches
print("📦 Adding documents to vector database...")
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
    print(f"  Added batch {i//batch_size + 1}/{(len(documents)//batch_size)+1}")

print(f"\n✅ Successfully added {len(documents)} documents to fresh ChromaDB!")
print("🎉 Vector database rebuilt successfully!")
