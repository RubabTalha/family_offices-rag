import streamlit as st
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import openai
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Page config MUST be first command
st.set_page_config(
    page_title="Family Office Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS - Your Original Beautiful Light Purple Theme
# ============================================
st.markdown("""
<style>
    /* Main background - soft lavender gradient */
    .stApp {
        background: linear-gradient(135deg, #f5e9ff 0%, #e6d5ff 100%);
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-1wrcr25 {
        background: linear-gradient(180deg, #d9c9ff 0%, #c4b0ff 100%);
    }
    
    /* Headers with subtle purple */
    h1, h2, h3 {
        color: #4a2b7a !important;
        font-weight: 600 !important;
    }
    
    /* Cards for results */
    .result-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid rgba(106, 13, 173, 0.2);
        box-shadow: 0 4px 15px rgba(106, 13, 173, 0.1);
        transition: transform 0.3s ease;
    }
    
    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(106, 13, 173, 0.15);
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        border-left: 4px solid #8a2be2;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #8a2be2 0%, #9b4dff 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #9b4dff 0%, #b16eff 100%);
        box-shadow: 0 4px 15px rgba(138, 43, 226, 0.3);
        transform: translateY(-2px);
    }
    
    /* Input fields */
    .stTextInput > div > div > input {
        border: 2px solid #e0d0ff;
        border-radius: 10px;
        padding: 10px;
        background: rgba(255, 255, 255, 0.9);
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #8a2be2;
        box-shadow: 0 0 0 2px rgba(138, 43, 226, 0.2);
    }
    
    /* Chat messages */
    .chat-message {
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        background: white;
        border: 1px solid #e0d0ff;
    }
    
    .user-message {
        background: linear-gradient(135deg, #e6d5ff 0%, #d9c9ff 100%);
        margin-left: 20%;
    }
    
    .assistant-message {
        background: white;
        margin-right: 20%;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(138, 43, 226, 0.1);
        border-radius: 8px;
        color: #4a2b7a;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.5);
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        color: #4a2b7a;
    }
    
    .stTabs [aria-selected="true"] {
        background: white;
        color: #8a2be2;
        font-weight: 600;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 20px;
        color: #4a2b7a;
        font-size: 0.9em;
        border-top: 1px solid rgba(138, 43, 226, 0.2);
        margin-top: 50px;
    }
    
    /* Confidence badges */
    .confidence-high {
        background: #28a745;
        color: white;
        padding: 3px 10px;
        border-radius: 15px;
        font-size: 12px;
        display: inline-block;
        margin-right: 10px;
    }
    
    .confidence-medium {
        background: #ffc107;
        color: black;
        padding: 3px 10px;
        border-radius: 15px;
        font-size: 12px;
        display: inline-block;
        margin-right: 10px;
    }
    
    .confidence-low {
        background: #dc3545;
        color: white;
        padding: 3px 10px;
        border-radius: 15px;
        font-size: 12px;
        display: inline-block;
        margin-right: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# Initialize Session State
# ============================================
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'search_history' not in st.session_state:
    st.session_state.search_history = []
if 'favorites' not in st.session_state:
    st.session_state.favorites = []

# ============================================
# Header - Your Original Beautiful Header
# ============================================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h1 style="font-size: 3em; margin-bottom: 0;">🏦</h1>
        <h1 style="color: #4a2b7a; margin-top: -10px;">Family Office Intelligence</h1>
        <p style="color: #6a4e8c; font-size: 1.2em;">Natural Language Query System</p>
        <div style="height: 4px; width: 100px; background: linear-gradient(90deg, #8a2be2, #b16eff); margin: 20px auto;"></div>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# Load RAG System with Caching
# ============================================
@st.cache_resource
def load_rag_system():
    """Load embedding model and vector database ONCE"""
    with st.spinner("🔄 Loading intelligence system..."):
        try:
            # Load embedding model
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Load ChromaDB
            client = chromadb.PersistentClient(path="./chroma_db")
            collection = client.get_collection("family_offices")
            
            # Load CSV for metadata
            df = pd.read_csv('family_offices.csv')
            
            return model, collection, df
        except Exception as e:
            st.error(f"Error loading system: {e}")
            return None, None, None

# Load everything
model, collection, df = load_rag_system()

# ============================================
# Confidence Score Function
# ============================================
# ============================================
# Confidence Score Function - FIXED VERSION
# ============================================
def calculate_confidence(results):
    """Calculate confidence based on similarity scores"""
    if not results or not results.get('distances'):
        return 0, "low"
    
    # Check if distances[0] exists and is a list
    if not results['distances'] or len(results['distances']) == 0:
        return 0, "low"
    
    distances_first = results['distances'][0]
    
    # Handle case where distances_first is a single float (not a list)
    if isinstance(distances_first, (int, float)):
        # Single value case
        similarity = 1 - (distances_first / 2)
        confidence = similarity * 0.7 + 0.3  # Single result with good similarity
        if confidence > 0.7:
            return confidence, "high"
        elif confidence > 0.4:
            return confidence, "medium"
        else:
            return confidence, "low"
    
    # Handle case where distances_first is a list
    if not isinstance(distances_first, list):
        return 0, "low"
    
    # Convert distances to similarities (0-1 range)
    similarities = []
    for d in distances_first:
        if d is not None:
            similarities.append(1 - (d/2))
    
    if not similarities:
        return 0, "low"
    
    avg_similarity = sum(similarities) / len(similarities)
    num_results = len(similarities)
    
    # More results with high similarity = higher confidence
    confidence = (avg_similarity * 0.7) + (min(num_results / 10, 1) * 0.3)
    
    if confidence > 0.7:
        return confidence, "high"
    elif confidence > 0.4:
        return confidence, "medium"
    else:
        return confidence, "low"

# ============================================
# RAG Search Function
# ============================================
def rag_search(query_text, top_k=10, country=None, fo_type=None):
    if model is None or collection is None:
        return None
    
    # Create query embedding
    query_embedding = model.encode(query_text).tolist()
    
    # Search in ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k * 2,  # Get more for filtering
        include=["documents", "metadatas", "distances"]
    )
    
    # Apply filters if needed
    if (country and country != 'All') or (fo_type and fo_type != 'All'):
        filtered_indices = []
        if results['metadatas'][0]:
            for i, metadata in enumerate(results['metadatas'][0]):
                include = True
                if country and country != 'All':
                    if metadata.get('country', '').lower() != country.lower():
                        include = False
                if fo_type and fo_type != 'All':
                    if metadata.get('type', '').lower() != fo_type.lower():
                        include = False
                if include:
                    filtered_indices.append(i)
        
        # Prepare filtered results
        filtered_results = {
            'ids': [results['ids'][0][i] for i in filtered_indices[:top_k]],
            'documents': [results['documents'][0][i] for i in filtered_indices[:top_k]],
            'metadatas': [results['metadatas'][0][i] for i in filtered_indices[:top_k]],
            'distances': [results['distances'][0][i] for i in filtered_indices[:top_k]]
        }
        return filtered_results
    else:
        # Return top_k results without filtering
        return {
            'ids': results['ids'][0][:top_k],
            'documents': results['documents'][0][:top_k],
            'metadatas': results['metadatas'][0][:top_k],
            'distances': results['distances'][0][:top_k]
        }

# ============================================
# Generate LLM Answer
# ============================================
def generate_llm_answer(query_text, context_docs, confidence_level):
    if not context_docs:
        return "I don't have any relevant documents to answer your question."
    
    # Prepare context
    context = "\n\n---\n\n".join([
        f"Document {i+1}:\n{doc}" 
        for i, doc in enumerate(context_docs[:5])
    ])
    
    prompt = f"""You are a family office intelligence expert. Answer the question based ONLY on the provided context.

CONTEXT:
{context}

QUESTION: {query_text}

CONFIDENCE LEVEL: {confidence_level}

RULES:
1. If the answer is in the context, provide it with specific examples
2. If only partial information exists, say what you know and what's missing
3. If the context doesn't contain the answer, say: "I don't have information about that in my database"
4. NEVER make up information
5. Be specific - mention family office names, locations, and investment focus

ANSWER:"""
    
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful family office expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return None  # Return None if LLM fails

# ============================================
# Sidebar - Your Original Beautiful Sidebar
# ============================================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h2 style="color: #4a2b7a;">✨ Dashboard</h2>
    </div>
    """, unsafe_allow_html=True)
    
    if df is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin:0;">📊</h3>
                <h2 style="margin:0; color:#4a2b7a;">{len(df)}</h2>
                <p style="margin:0;">Total Offices</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            countries = df['Country'].nunique() if 'Country' in df.columns else 0
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin:0;">🌍</h3>
                <h2 style="margin:0; color:#4a2b7a;">{countries}</h2>
                <p style="margin:0;">Countries</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Investment focus distribution
        st.markdown("### 🎯 Investment Focus")
        if 'Investment Focus' in df.columns:
            focus_counts = df['Investment Focus'].value_counts().head(5)
            fig = px.pie(
                values=focus_counts.values,
                names=focus_counts.index,
                color_discrete_sequence=['#8a2be2', '#9b4dff', '#b16eff', '#c68eff', '#d9b0ff'],
                hole=0.4
            )
            fig.update_layout(
                showlegend=True,
                margin=dict(t=0, b=0, l=0, r=0),
                height=200,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Filters
        st.markdown("### 🎯 Filters")
        if 'Country' in df.columns:
            countries_list = ['All'] + sorted(df['Country'].dropna().unique().tolist())
            selected_country = st.selectbox("Country", countries_list)
        
        if 'Type' in df.columns:
            types_list = ['All'] + sorted(df['Type'].dropna().unique().tolist())
            selected_type = st.selectbox("Family Office Type", types_list)
        
        # RAG settings
        st.markdown("### ⚙️ Search Settings")
        top_k = st.slider("Number of results", min_value=3, max_value=20, value=10)
        
        use_llm = st.checkbox("🤖 Use AI for better answers", value=False)
        if use_llm:
            api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
            if api_key:
                openai.api_key = api_key
                st.success("✅ API key set")
        
        # Search history
        if st.session_state.search_history:
            st.markdown("### 📜 Recent Searches")
            for q in st.session_state.search_history[-5:]:
                if st.button(f"🔍 {q}", key=f"hist_{q}"):
                    st.session_state["example_query"] = q
                    st.rerun()
        
        # Footer in sidebar
        st.markdown("""
        <div style="position: fixed; bottom: 0; padding: 20px; text-align: center; width: 100%;">
            <p style="color: #4a2b7a; font-size: 0.8em;">Built with ❤️ for PolarityIQ</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================
# Main Tabs
# ============================================
tab1, tab2, tab3 = st.tabs(["🔍 Query", "📊 Analytics", "⭐ Favorites"])

# ============================================
# Tab 1: Query Interface - Your Original Beautiful Design
# ============================================
with tab1:
    # Search bar with fancy styling
    st.markdown("""
    <div style="background: rgba(255,255,255,0.5); padding: 30px; border-radius: 20px; margin: 20px 0;">
        <h3 style="text-align: center; color: #4a2b7a;">Ask anything about family offices</h3>
    """, unsafe_allow_html=True)
    
    query = st.text_input(
        "",
        placeholder="e.g., Which family offices invest in AI?",
        key="query_input",
        label_visibility="collapsed"
    )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Example queries as beautiful chips
    st.markdown("### 💡 Try these examples:")
    examples = [
        "🏦 Which family offices invest in AI?",
        "🌍 Show family offices in Europe",
        "💰 Find offices with check sizes > $10M",
        "👥 Who are decision makers at Singapore offices?",
        "🏥 Recent healthcare investments",
        "🤝 Co-investors with Sequoia"
    ]
    
    cols = st.columns(3)
    for i, ex in enumerate(examples):
        with cols[i % 3]:
            if st.button(ex, key=f"ex_{i}", use_container_width=True):
                query = ex.split(" ", 1)[1]  # Remove emoji
                st.session_state["example_query"] = query
                st.rerun()
    
    # Process query
    if query or 'example_query' in st.session_state:
        if 'example_query' in st.session_state:
            query = st.session_state['example_query']
            del st.session_state['example_query']
        
        # Add to search history
        if query not in st.session_state.search_history:
            st.session_state.search_history.append(query)
        
        st.markdown("---")
        
        # User message
        with st.chat_message("user"):
            st.markdown(query)
        
        # Search
        with st.spinner("🔍 Searching family office database..."):
            results = rag_search(
                query,
                top_k=top_k,
                country=selected_country if 'selected_country' in locals() else None,
                fo_type=selected_type if 'selected_type' in locals() else None
            )
        
        # Calculate confidence
        confidence_score, confidence_level = calculate_confidence(results)
        
        # Assistant response
        with st.chat_message("assistant"):
            if results and results['documents']:
                # Show confidence badge
                if confidence_level == "high":
                    st.markdown(f'<span class="confidence-high">✨ High Confidence ({confidence_score:.1%})</span>', unsafe_allow_html=True)
                elif confidence_level == "medium":
                    st.markdown(f'<span class="confidence-medium">📊 Medium Confidence ({confidence_score:.1%})</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="confidence-low">⚠️ Low Confidence ({confidence_score:.1%})</span>', unsafe_allow_html=True)
                
                # Generate LLM answer if enabled
                if use_llm and 'openai' in dir() and openai.api_key:
                    llm_answer = generate_llm_answer(query, results['documents'], confidence_level)
                    if llm_answer:
                        st.markdown("### Answer:")
                        st.markdown(llm_answer)
                    else:
                        st.markdown("### 📊 Found these family offices:")
                else:
                    st.markdown("### 📊 Found these family offices:")
                
                # Show results in expanders
                for i, (doc, metadata, dist) in enumerate(zip(
                    results['documents'][:5], 
                    results['metadatas'][:5],
                    results['distances'][:5]
                )):
                    similarity = 1 - (dist/2) if dist else 0.5
                    firm_name = metadata.get('firm_name', 'Family Office') if metadata else 'Family Office'
                    
                    with st.expander(f"🏦 **{firm_name}** (Match: {similarity:.1%})", expanded=i==0):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(doc)
                        with col2:
                            st.markdown("**Quick Actions:**")
                            if st.button("⭐ Save", key=f"fav_{i}_{firm_name}"):
                                st.session_state.favorites.append({
                                    'name': firm_name,
                                    'text': doc[:200] + '...',
                                    'query': query
                                })
                                st.success("Saved!")
            else:
                st.markdown("😕 I don't have information about that in my database. Try a different query or remove filters.")

# ============================================
# Tab 2: Analytics Dashboard
# ============================================
with tab2:
    st.markdown("## 📊 Dataset Analytics")
    
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🏦</h3>
                <h2>{len(df)}</h2>
                <p>Total Offices</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            verified = df['Verified Method'].notna().sum() if 'Verified Method' in df.columns else 0
            st.markdown(f"""
            <div class="metric-card">
                <h3>✅</h3>
                <h2>{verified}</h2>
                <p>Verified</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            if 'Check Size Max' in df.columns:
                df['Check Size Max'] = pd.to_numeric(df['Check Size Max'], errors='coerce')
                avg_check = df['Check Size Max'].mean()
                st.markdown(f"""
                <div class="metric-card">
                    <h3>💰</h3>
                    <h2>${avg_check:.1f}M</h2>
                    <p>Avg Max Check</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col4:
            if 'AUM Range' in df.columns:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📈</h3>
                    <h2>{df['AUM Range'].notna().sum()}</h2>
                    <p>With AUM Data</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Country' in df.columns:
                st.markdown("### 🌍 Top Countries")
                country_counts = df['Country'].value_counts().head(10)
                fig = px.bar(
                    x=country_counts.values,
                    y=country_counts.index,
                    orientation='h',
                    color=country_counts.values,
                    color_continuous_scale=['#e6d5ff', '#8a2be2']
                )
                fig.update_layout(
                    height=400,
                    xaxis_title="Number of Family Offices",
                    yaxis_title="",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'Type' in df.columns:
                st.markdown("### 🏛️ Office Type")
                type_counts = df['Type'].value_counts()
                fig = px.pie(
                    values=type_counts.values,
                    names=type_counts.index,
                    color_discrete_sequence=['#8a2be2', '#b16eff', '#d9b0ff']
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # Investment focus
        if 'Investment Focus' in df.columns:
            st.markdown("### 🎯 Investment Focus Distribution")
            focus_text = ' '.join(df['Investment Focus'].dropna().astype(str))
            st.markdown(f"""
            <div style="background: white; padding: 20px; border-radius: 15px;">
                <p style="font-size: 1.2em; line-height: 1.8;">{focus_text[:500]}...</p>
            </div>
            """, unsafe_allow_html=True)

# ============================================
# Tab 3: Favorites
# ============================================
with tab3:
    st.markdown("## ⭐ Saved Family Offices")
    
    if st.session_state.favorites:
        for i, fav in enumerate(st.session_state.favorites):
            with st.container():
                st.markdown(f"""
                <div class="result-card">
                    <h3>🏦 {fav['name']}</h3>
                    <p>{fav['text']}</p>
                    <p style="color: #6a4e8c;">Query: {fav.get('query', 'Saved manually')}</p>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Remove", key=f"remove_{i}"):
                    st.session_state.favorites.pop(i)
                    st.rerun()
    else:
        st.markdown("""
        <div style="text-align: center; padding: 50px;">
            <h1 style="font-size: 4em;">⭐</h1>
            <p style="color: #4a2b7a;">No favorites yet. Click the star icon on search results to save them here.</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================
# Footer - Your Original Beautiful Footer
# ============================================
st.markdown("""
<div class="footer">
    <p>Powered by ChromaDB + Sentence Transformers | Data last updated: March 2026</p>
    <p style="font-size: 0.8em;">388 family offices • 27 data points • Real-time RAG queries</p>
</div>
""", unsafe_allow_html=True)