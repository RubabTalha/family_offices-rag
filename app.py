import streamlit as st
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import openai
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import os

# Page config MUST be first command
st.set_page_config(
    page_title="Family Office Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS - Beautiful Light Purple Theme
# ============================================
st.markdown("""
<style>
    /* Main background - soft lavender gradient */
    .stApp {
        background: linear-gradient(135deg, #f5e9ff 0%, #e6d5ff 100%);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
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
    
    /* Success box */
    .success-box {
        background: rgba(40, 167, 69, 0.1);
        border-left: 4px solid #28a745;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    /* Warning box */
    .warning-box {
        background: rgba(255, 193, 7, 0.1);
        border-left: 4px solid #ffc107;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
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
if 'df' not in st.session_state:
    try:
        st.session_state.df = pd.read_csv('family_offices.csv')
        st.session_state.data_loaded = True
    except:
        st.session_state.df = None
        st.session_state.data_loaded = False

# ============================================
# Header
# ============================================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h1 style="font-size: 3em; margin-bottom: 0;">🏦</h1>
        <h1 style="color: #4a2b7a; margin-top: -10px;">Family Office Intelligence</h1>
        <p style="color: #6a4e8c; font-size: 1.2em;">RAG-Powered Search System</p>
        <div style="height: 4px; width: 100px; background: linear-gradient(90deg, #8a2be2, #b16eff); margin: 20px auto;"></div>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# Load RAG System with Caching - FIXED VERSION
# ============================================
@st.cache_resource
def load_rag_system():
    """Load embedding model and vector database with error handling"""
    with st.spinner("🔄 Loading intelligence system..."):
        try:
            # Load embedding model
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Try to load ChromaDB with compatibility mode
            try:
                client = chromadb.PersistentClient(path="./chroma_db")
                # Try to get collection, if fails we'll recreate
                try:
                    collection = client.get_collection("family_offices")
                    # Test if collection works
                    collection.count()
                except:
                    st.warning("⚠️ Vector database needs to be recreated locally. Using pandas search fallback.")
                    collection = None
            except Exception as e:
                st.warning(f"⚠️ ChromaDB error: {e}. Using pandas search fallback.")
                collection = None
            
            # Load CSV for metadata
            df = pd.read_csv('family_offices.csv')
            
            return model, collection, df
        except Exception as e:
            st.error(f"Error loading system: {e}")
            return None, None, st.session_state.df if st.session_state.data_loaded else None

# Load everything
model, collection, df = load_rag_system()

# ============================================
# Fallback Search Function (Uses pandas if ChromaDB fails)
# ============================================
def pandas_search(query_text, df, country=None, fo_type=None):
    """Simple pandas-based search as fallback"""
    if df is None:
        return []
    
    query_text = query_text.lower()
    results = []
    
    for idx, row in df.iterrows():
        score = 0
        # Search in multiple columns
        searchable_text = f"""
        {row.get('FO Firm Name', '')} 
        {row.get('Country', '')} 
        {row.get('Investment Focus', '')} 
        {row.get('Recent Deal 1', '')} 
        {row.get('Recent Deal 2', '')} 
        {row.get('Recent Deal 3', '')}
        """.lower()
        
        if query_text in searchable_text:
            score += 1
        
        # Apply filters
        if country and country != 'All':
            if str(row.get('Country', '')).lower() != country.lower():
                score = 0
        
        if fo_type and fo_type != 'All':
            if str(row.get('Type', '')).lower() != fo_type.lower():
                score = 0
        
        if score > 0:
            results.append({
                'firm_name': row.get('FO Firm Name', 'N/A'),
                'country': row.get('Country', 'N/A'),
                'investment_focus': row.get('Investment Focus', 'N/A'),
                'check_min': row.get('Check Size Min', 'N/A'),
                'check_max': row.get('Check Size Max', 'N/A'),
                'recent_deal': row.get('Recent Deal 1', 'N/A'),
                'text': f"""
Family Office: {row.get('FO Firm Name', 'N/A')}
Location: {row.get('Country', 'N/A')}, {row.get('City', 'N/A')}
Type: {row.get('Type', 'N/A')}
Investment Focus: {row.get('Investment Focus', 'N/A')}
Check Size: ${row.get('Check Size Min', 'N/A')} - ${row.get('Check Size Max', 'N/A')}
Recent Deal: {row.get('Recent Deal 1', 'N/A')}
Contact: {row.get('Contact Name', 'N/A')} - {row.get('Contact Title', 'N/A')}
"""
            })
    
    return results[:10]

# ============================================
# Confidence Score Function - FIXED
# ============================================
def calculate_confidence(results, method="rag"):
    """Calculate confidence based on results"""
    if not results:
        return 0, "low"
    
    if method == "rag" and isinstance(results, dict) and results.get('distances'):
        # RAG-based confidence
        distances = results.get('distances', [[]])[0]
        if distances and len(distances) > 0:
            avg_similarity = 1 - (sum(distances) / len(distances) / 2)
            num_results = len(distances)
            confidence = (avg_similarity * 0.7) + (min(num_results / 10, 1) * 0.3)
        else:
            confidence = 0.5
    else:
        # Pandas-based confidence (default to medium)
        confidence = 0.6
    
    if confidence > 0.7:
        return confidence, "high"
    elif confidence > 0.4:
        return confidence, "medium"
    else:
        return confidence, "low"

# ============================================
# RAG Search Function - FIXED for compatibility
# ============================================
def rag_search(query_text, top_k=10, country=None, fo_type=None):
    if model is None or collection is None:
        return None
    
    try:
        # Create query embedding
        query_embedding = model.encode(query_text).tolist()
        
        # Search in ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k * 2,
            include=["documents", "metadatas", "distances"]
        )
        
        # Apply filters
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
            
            filtered_results = {
                'ids': [results['ids'][0][i] for i in filtered_indices[:top_k]],
                'documents': [results['documents'][0][i] for i in filtered_indices[:top_k]],
                'metadatas': [results['metadatas'][0][i] for i in filtered_indices[:top_k]],
                'distances': [results['distances'][0][i] for i in filtered_indices[:top_k]]
            }
            return filtered_results
        else:
            return {
                'ids': results['ids'][0][:top_k],
                'documents': results['documents'][0][:top_k],
                'metadatas': results['metadatas'][0][:top_k],
                'distances': results['distances'][0][:top_k]
            }
    except Exception as e:
        st.warning(f"RAG search error: {e}. Using fallback search.")
        return None

# ============================================
# Sidebar
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
        
        # Filters
        st.markdown("### 🎯 Filters")
        if 'Country' in df.columns:
            countries_list = ['All'] + sorted(df['Country'].dropna().unique().tolist())
            selected_country = st.selectbox("Country", countries_list, key="country_filter")
        else:
            selected_country = 'All'
        
        if 'Type' in df.columns:
            types_list = ['All'] + sorted(df['Type'].dropna().unique().tolist())
            selected_type = st.selectbox("Family Office Type", types_list, key="type_filter")
        else:
            selected_type = 'All'
        
        # Search settings
        st.markdown("### ⚙️ Search Settings")
        top_k = st.slider("Number of results", min_value=3, max_value=20, value=10, key="top_k_slider")
        
        # Search history
        if st.session_state.search_history:
            st.markdown("### 📜 Recent Searches")
            for q in st.session_state.search_history[-5:]:
                if st.button(f"🔍 {q}", key=f"hist_{q}"):
                    st.session_state.example_query = q
                    st.rerun()

# ============================================
# Main Tabs
# ============================================
tab1, tab2, tab3 = st.tabs(["🔍 Query", "📊 Analytics", "⭐ Favorites"])

# ============================================
# Tab 1: Query Interface
# ============================================
with tab1:
    # Search bar
    st.markdown("""
    <div style="background: rgba(255,255,255,0.5); padding: 30px; border-radius: 20px; margin: 20px 0;">
        <h3 style="text-align: center; color: #4a2b7a;">Ask anything about family offices</h3>
    </div>
    """, unsafe_allow_html=True)
    
    query = st.text_input(
        "",
        placeholder="e.g., Which family offices invest in AI?",
        key="query_input",
        label_visibility="collapsed"
    )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Example queries
    st.markdown("### 💡 Try these examples:")
    examples = [
        "Which family offices invest in AI?",
        "Show family offices in Europe",
        "Recent climate tech deals",
        "Healthcare investors with $10M+ checks",
        "Singapore family offices",
        "Co-investors with Sequoia"
    ]
    
    cols = st.columns(3)
    for i, ex in enumerate(examples):
        with cols[i % 3]:
            if st.button(f"🔍 {ex}", key=f"ex_{i}", use_container_width=True):
                query = ex
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
        
        # Show user message
        with st.chat_message("user"):
            st.markdown(query)
        
        # Search
        with st.spinner("🔍 Searching..."):
            # Try RAG first, fallback to pandas
            rag_results = rag_search(query, top_k=top_k, country=selected_country, fo_type=selected_type)
            
            if rag_results and rag_results.get('documents'):
                results = rag_results
                search_method = "rag"
            else:
                results = pandas_search(query, df, country=selected_country, fo_type=selected_type)
                search_method = "pandas"
        
        # Calculate confidence
        confidence_score, confidence_level = calculate_confidence(results, search_method)
        
        # Assistant response
        with st.chat_message("assistant"):
            if results:
                # Show confidence badge
                if confidence_level == "high":
                    st.markdown(f'<span class="confidence-high">✨ High Confidence ({confidence_score:.1%})</span>', unsafe_allow_html=True)
                elif confidence_level == "medium":
                    st.markdown(f'<span class="confidence-medium">📊 Medium Confidence ({confidence_score:.1%})</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="confidence-low">⚠️ Low Confidence ({confidence_score:.1%})</span>', unsafe_allow_html=True)
                
                if search_method == "rag":
                    st.markdown("### 📊 Found these family offices (RAG search):")
                    for i, (doc, metadata) in enumerate(zip(results['documents'][:5], results['metadatas'][:5])):
                        firm_name = metadata.get('firm_name', 'Family Office')
                        with st.expander(f"🏦 **{firm_name}**", expanded=i==0):
                            st.markdown(doc)
                            if st.button("⭐ Save", key=f"fav_{i}_{firm_name}"):
                                st.session_state.favorites.append({
                                    'name': firm_name,
                                    'text': doc[:200] + '...',
                                    'query': query
                                })
                                st.success("Saved!")
                else:
                    st.markdown("### 📊 Found these family offices (quick search):")
                    for i, result in enumerate(results[:5]):
                        with st.expander(f"🏦 **{result['firm_name']}**", expanded=i==0):
                            st.markdown(result['text'])
                            if st.button("⭐ Save", key=f"fav_{i}_{result['firm_name']}"):
                                st.session_state.favorites.append({
                                    'name': result['firm_name'],
                                    'text': result['text'][:200] + '...',
                                    'query': query
                                })
                                st.success("Saved!")
            else:
                st.markdown("""
                <div class="warning-box">
                    <strong>😕 No results found.</strong> Try different keywords or remove filters.
                </div>
                """, unsafe_allow_html=True)

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
                aum_count = df['AUM Range'].notna().sum()
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📈</h3>
                    <h2>{aum_count}</h2>
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
# Footer
# ============================================
st.markdown("""
<div class="footer">
    <p>Powered by ChromaDB + Sentence Transformers | Fallback: Pandas Search</p>
    <p style="font-size: 0.8em;">388 family offices • 27 data points</p>
</div>
""", unsafe_allow_html=True)
