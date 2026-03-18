# RAG Pipeline Documentation

## Stack Choices
- **Vector Database**: ChromaDB (local, free, easy to deploy)
- **Embeddings**: all-MiniLM-L6-v2 (good balance of speed/quality)
- **LLM**: Optional OpenAI GPT-3.5 (for enhanced answers)
- **Frontend**: Streamlit (fastest development, free hosting)

## Chunking Strategy
- Each family office is one document (record-level)
- Rich text combines all 25+ columns into natural language
- This preserves relationships between fields

## Embedding Model
- Sentence Transformers all-MiniLM-L6-v2
- 384-dimensional embeddings
- Fast inference, good semantic understanding 

## Retrieval Approach
- Cosine similarity search
- Top-k = 10 results
- Metadata preserved for filtering

## Challenges Faced
1. **Column alignment**: Ensured CSV columns matched code
2. **Deployment size**: Embedding model is ~80MB, works on free tier
3. **Missing data**: Handled NaN values gracefully

## What I'd Improve With More Time
1. Add metadata filtering (country, investment focus)
2. Implement hybrid search (keyword + semantic)
3. Add web fallback for out-of-scope queries
4. Create admin dashboard for dataset updates