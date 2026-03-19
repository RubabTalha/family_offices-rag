import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
import shutil
import os

print("=" * 50)
print("🚀 REBUILDING VECTOR DATABASE - PURE RAG")
print("=" * 50)

# Load dataset
print("\n📊 Loading family office dataset...")
df = pd.read_csv('family_offices.csv')
print(f"✅ Loaded {len(df)} rows")

# Helper for safe strings
def safe_str(val):
    if pd.isna(val) or val is None:
        return ''
    return str(val)

# Create rich documents
print("\n📝 Creating rich text documents...")
documents = []

for idx, row in df.iterrows():
    if pd.isna(row.get('FO Firm Name')):
        continue
    
    # Build comprehensive document with ALL fields
    doc = f"""
FAMILY OFFICE: {safe_str(row.get('FO Firm Name'))}
WEBSITE: {safe_str(row.get('Website'))}
LOCATION: {safe_str(row.get('Country'))}, {safe_str(row.get('City'))}
FOUNDED: {safe_str(row.get('Founded Year'))}
TYPE: {safe_str(row.get('Type'))}

INVESTMENT FOCUS: {safe_str(row.get('Investment Focus'))}
CHECK SIZE: ${safe_str(row.get('Check Size Min'))} - ${safe_str(row.get('Check Size Max'))}M
AUM: {safe_str(row.get('AUM Range'))}

RECENT DEALS:
1. {safe_str(row.get('Recent Deal 1'))}
2. {safe_str(row.get('Recent Deal 2'))}
3. {safe_str(row.get('Recent Deal 3'))}

CO-INVESTORS: {safe_str(row.get('Co-Investors'))}
LP RELATIONS: {safe_str(row.get('LP Relations'))}

DECISION MAKER: {safe_str(row.get('Contact Name'))} - {safe_str(row.get('Contact Title'))}
EMAIL: {safe_str(row.get('Contact Email'))}
LINKEDIN: {safe_str(row.get('Contact LinkedIn'))}

RECENT NEWS: {safe_str(row.get('Recent News'))} ({safe_str(row.get('News Date'))})
SOURCE: {safe_str(row.get('News Source'))}

VERIFIED: {safe_str(row.get('Verified Method'))} on {safe_str(row.get('Verified Date'))}
SOURCE URL: {safe_str(row.get('Source URL'))}
NOTES: {safe_str(row.get('Notes'))}
"""
    
    documents.append({
        'id': f"fo_{idx}",
        'text': doc,
        'metadata': {
            'firm_name': safe_str(row.get('FO Firm Name')),
            'country': safe_str(row.get('Country')),
            'type': safe_str(row.get('Type')),
            'city': safe_str(row.get('City'))
        }
    })

print(f"✅ Created {len(documents)} documents")

# Load embedding model
print("\n🧠 Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings
print("\n🔢 Generating embeddings (this takes 2-3 minutes)...")
texts = [doc['text'] for doc in documents]
embeddings = model.encode(texts, show_progress_bar=True)

# Remove old database
print("\n🗑️ Removing old vector database...")
if os.path.exists("./chroma_db"):
    shutil.rmtree("./chroma_db")

# Create fresh database
print("\n🆕 Creating fresh vector database...")
client = chromadb.PersistentClient(path="./chroma_db")

# Create collection
collection = client.create_collection(
    name="family_offices",
    metadata={"hnsw:space": "cosine"}
)

# Add in batches
print("\n📦 Adding documents to vector database...")
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

print("\n" + "=" * 50)
print(f"✅ SUCCESS! Added {len(documents)} documents to fresh vector database")
print("=" * 50)
