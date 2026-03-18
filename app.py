import streamlit as st
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import os
import numpy as np

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
    except Exception as e:
        st.session_state.df = None
        st.session_state.data_loaded = False
        st.error(f"Error loading CSV: {e}")

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
# Load RAG System with Caching
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
                    st.success("✅ RAG system loaded successfully!")
                except Exception as e:
                    st.warning(f"⚠️ Vector database issue: {e}. Using enhanced keyword search fallback.")
                    collection = None
            except Exception as e:
                st.warning(f"⚠️ ChromaDB error: {e}. Using enhanced keyword search fallback.")
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
# Helper Functions for Safe String Handling
# ============================================
def safe_str(val):
    """Convert any value to lowercase string safely"""
    if pd.isna(val) or val is None:
        return ''
    return str(val).lower()

def safe_str_display(val, default='N/A'):
    """Convert any value to display string safely"""
    if pd.isna(val) or val is None:
        return default
    return str(val)

def safe_int(val, default=0):
    """Convert any value to int safely"""
    if pd.isna(val) or val is None:
        return default
    try:
        return int(float(val))
    except:
        return default

# ============================================
# Improved Fallback Search Function - FULLY FIXED
# ============================================
def pandas_search(query_text, df, country=None, fo_type=None):
    """Improved pandas-based search with better matching and safe string handling"""
    if df is None or len(df) == 0:
        return []
    
    query_text = query_text.lower().strip()
    results = []
    
    # Split query into keywords for better matching
    keywords = [k for k in query_text.split() if len(k) > 2]  # Remove short words
    
    for idx, row in df.iterrows():
        score = 0
        matches = []
        
        # Get values safely using helper functions
        firm_name = safe_str(row.get('FO Firm Name'))
        country_val = safe_str(row.get('Country'))
        city_val = safe_str(row.get('City'))
        investment_focus = safe_str(row.get('Investment Focus'))
        recent_deal_1 = safe_str(row.get('Recent Deal 1'))
        recent_deal_2 = safe_str(row.get('Recent Deal 2'))
        recent_deal_3 = safe_str(row.get('Recent Deal 3'))
        contact_name = safe_str(row.get('Contact Name'))
        contact_title = safe_str(row.get('Contact Title'))
        notes_val = safe_str(row.get('Notes'))
        
        # Check for exact phrase match first (higher score)
        if query_text in firm_name or query_text in notes_val:
            score += 5
        
        # Check for "decision maker" related queries
        if any(word in query_text for word in ['decision', 'maker', 'who', 'contact', 'run', 'runs']):
            if contact_name and contact_name != '':
                score += 3
                matches.append('has_contact')
        
        # Check for location queries
        if 'singapore' in query_text and country_val == 'singapore':
            score += 10
            matches.append('singapore')
        
        if 'europe' in query_text and country_val in ['uk', 'germany', 'france', 'switzerland', 'denmark', 'sweden', 'italy', 'spain', 'netherlands']:
            score += 5
            matches.append('europe')
        
        if 'asia' in query_text and country_val in ['singapore', 'hong kong', 'china', 'japan', 'korea', 'india']:
            score += 5
            matches.append('asia')
        
        if 'usa' in query_text or 'us' in query_text or 'america' in query_text:
            if country_val in ['usa', 'united states', 'us']:
                score += 5
                matches.append('usa')
        
        # Then check individual keywords
        for keyword in keywords:
            searchable_fields = [
                firm_name, country_val, city_val, investment_focus,
                recent_deal_1, recent_deal_2, recent_deal_3,
                contact_name, contact_title, notes_val
            ]
            for field_value in searchable_fields:
                if keyword in field_value:
                    score += 1
                    matches.append(keyword)
        
        # Apply filters
        if country and country != 'All':
            if country_val != country.lower():
                score = 0
        
        if fo_type and fo_type != 'All':
            type_val = safe_str(row.get('Type'))
            if type_val != fo_type.lower():
                score = 0
        
        if score > 0:
            # Get display values safely
            display_firm = safe_str_display(row.get('FO Firm Name'))
            display_country = safe_str_display(row.get('Country'))
            display_city = safe_str_display(row.get('City'))
            display_contact = safe_str_display(row.get('Contact Name'))
            display_title = safe_str_display(row.get('Contact Title'))
            display_email = safe_str_display(row.get('Contact Email'))
            display_linkedin = safe_str_display(row.get('Contact LinkedIn'))
            display_focus = safe_str_display(row.get('Investment Focus'))
            display_check_min = safe_str_display(row.get('Check Size Min'))
            display_check_max = safe_str_display(row.get('Check Size Max'))
            display_type = safe_str_display(row.get('Type'))
            display_deal = safe_str_display(row.get('Recent Deal 1'))
            display_notes = safe_str_display(row.get('Notes'))
            
            # Truncate notes if too long
            if len(display_notes) > 200:
                display_notes = display_notes[:200] + '...'
            
            # Create rich result text
            result_text = f"""
🏢 **{display_firm}**
📍 **Location:** {display_country}, {display_city}
👤 **Decision Maker:** {display_contact} - {display_title}
📧 **Contact:** {display_email}
🔗 **LinkedIn:** {display_linkedin}
💰 **Investment Focus:** {display_focus}
💵 **Check Size:** ${display_check_min} - ${display_check_max}M
📊 **Type:** {display_type}
🤝 **Recent Deal:** {display_deal}
📝 **Notes:** {display_notes}
"""
            
            results.append({
                'score': score,
                'firm_name': display_firm,
                'country': display_country,
                'city': display_city,
                'contact_name': display_contact,
                'contact_title': display_title,
                'contact_email': display_email,
                'investment_focus': display_focus,
                'text': result_text,
                'match_count': len(matches)
            })
    
    # Sort by score (highest first)
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:15]  # Return top 15

# ============================================
# Confidence Score Function
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
        # Pandas-based confidence based on score distribution
        if isinstance(results, list) and len(results) > 0:
            avg_score = sum([r.get('score', 0) for r in results]) / len(results)
            confidence = min(avg_score / 20, 0.9)  # Normalize to 0-0.9 range
        else:
            confidence = 0.5
    
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
    
    if df is not None and st.session_state.data_loaded:
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
        
        st.markdown("---")
        
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
        
        st.markdown("---")
        
        # Search history
        if st.session_state.search_history:
            st.markdown("### 📜 Recent Searches")
            for q in st.session_state.search_history[-5:]:
                if st.button(f"🔍 {q}", key=f"hist_{q}", use_container_width=True):
                    st.session_state.example_query = q
                    st.rerun()
    else:
        st.error("❌ Dataset not loaded. Please check family_offices.csv")

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
        "Who are decision makers in Singapore",
        "Co-investors with Sequoia",
        "Family offices in Germany",
        "Contact information for VMS Group"
    ]
    
    cols = st.columns(4)
    for i, ex in enumerate(examples):
        with cols[i % 4]:
            if st.button(f"🔍 {ex[:15]}...", key=f"ex_{i}", help=ex, use_container_width=True):
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
            st.markdown(f"**You:** {query}")
        
        # Search
        with st.spinner("🔍 Searching..."):
            # Try RAG first, fallback to pandas
            rag_results = rag_search(query, top_k=top_k, country=selected_country, fo_type=selected_type)
            
            if rag_results and rag_results.get('documents') and len(rag_results['documents']) > 0:
                results = rag_results
                search_method = "rag"
                st.success("✅ Using RAG semantic search")
            else:
                results = pandas_search(query, df, country=selected_country, fo_type=selected_type)
                search_method = "pandas"
                st.info("📊 Using enhanced keyword search")
        
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
                    st.markdown(f"### 📊 Found {len(results['documents'])} family offices")
                    for i, (doc, metadata) in enumerate(zip(results['documents'][:7], results['metadatas'][:7])):
                        firm_name = metadata.get('firm_name', 'Family Office')
                        with st.expander(f"🏦 **{firm_name}**", expanded=i==0):
                            st.markdown(doc)
                            col1, col2 = st.columns([4, 1])
                            with col2:
                                if st.button("⭐ Save", key=f"fav_rag_{i}_{firm_name}"):
                                    st.session_state.favorites.append({
                                        'name': firm_name,
                                        'text': doc[:200] + '...',
                                        'query': query
                                    })
                                    st.success("Saved!")
                else:
                    st.markdown(f"### 📊 Found {len(results)} family offices")
                    for i, result in enumerate(results[:7]):
                        with st.expander(f"🏦 **{result['firm_name']}** (Match score: {result['score']})", expanded=i==0):
                            col1, col2 = st.columns([4, 1])
                            with col1:
                                st.markdown(result['text'])
                            with col2:
                                st.markdown("**Quick Actions:**")
                                if st.button("⭐ Save", key=f"fav_pandas_{i}_{result['firm_name']}"):
                                    st.session_state.favorites.append({
                                        'name': result['firm_name'],
                                        'text': f"{result['firm_name']} - {result.get('contact_name', 'N/A')} ({result['country']})",
                                        'query': query
                                    })
                                    st.success("Saved!")
                                
                                if result.get('contact_name') and result['contact_name'] != 'N/A':
                                    st.info(f"👤 {result['contact_name']}")
                                
                                if result.get('contact_email') and result['contact_email'] != 'N/A':
                                    st.caption(f"📧 {result['contact_email']}")
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
    
    if df is not None and st.session_state.data_loaded:
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
                if pd.isna(avg_check):
                    avg_check = 0
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
        
        st.markdown("---")
        
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
                    title="Number of Family Offices by Country",
                    color=country_counts.values,
                    color_continuous_scale=['#e6d5ff', '#8a2be2']
                )
                fig.update_layout(
                    height=400,
                    xaxis_title="Count",
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
                    title="SFO vs MFO Distribution",
                    color_discrete_sequence=['#8a2be2', '#b16eff', '#d9b0ff']
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # Investment Focus Analysis
        if 'Investment Focus' in df.columns:
            st.markdown("### 🎯 Top Investment Focus Areas")
            all_focuses = []
            for focus in df['Investment Focus'].dropna():
                if pd.isna(focus):
                    continue
                for f in str(focus).split(','):
                    all_focuses.append(f.strip())
            
            if all_focuses:
                focus_counts = pd.Series(all_focuses).value_counts().head(15)
                fig = px.bar(
                    x=focus_counts.values,
                    y=focus_counts.index,
                    orientation='h',
                    title="Most Common Investment Focus Areas",
                    color=focus_counts.values,
                    color_continuous_scale=['#e6d5ff', '#8a2be2']
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("❌ Dataset not available for analytics")

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
    <p>Powered by ChromaDB + Sentence Transformers | Fallback: Enhanced Keyword Search</p>
    <p style="font-size: 0.8em;">388 family offices • 27 data points • Updated March 2026</p>
</div>
""", unsafe_allow_html=True)
