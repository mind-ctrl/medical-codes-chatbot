"""
Medical Coding Assistant - Streamlit Chatbot
Beautiful UI for interacting with the Medical Coding RAG system
"""

import streamlit as st
import requests
import time
from typing import Dict, List

# Configuration - Use Streamlit secrets for cloud deployment, fallback to localhost for local dev
API_BASE_URL = st.secrets.get("API_BASE_URL", "http://localhost:8080") if hasattr(st, 'secrets') else "http://localhost:8080"

# Page config
st.set_page_config(
    page_title="Medical Coding Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Light Theme
st.markdown("""
<style>
    /* Main container background */
    .main {
        background-color: #eeefef !important;
    }

    /* Override Streamlit defaults */
    [data-testid="stAppViewContainer"] {
        background-color: #eeefef !important;
    }
    [data-testid="stHeader"] {
        background-color: #eeefef !important;
    }
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
    }

    /* Headers */
    .main-header {
        font-size: 2.5rem;
        color: #0a1118;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #0a1118;
        opacity: 0.7;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Input fields - Light theme */
    .stTextInput>div>div>input,
    .stTextArea>div>div>textarea {
        background-color: #ffffff !important;
        color: #0a1118 !important;
        border: 1px solid #d0d0d0 !important;
        border-radius: 5px !important;
    }
    .stTextInput>div>div>input::placeholder {
        color: #888888 !important;
    }

    /* Selectbox - Light theme */
    .stSelectbox>div>div>div {
        background-color: #ffffff !important;
        color: #0a1118 !important;
        border: 1px solid #d0d0d0 !important;
    }

    /* Slider */
    .stSlider>div>div>div {
        background-color: #d0d0d0 !important;
    }

    /* Button styling - Light theme */
    .stButton>button {
        width: 100%;
        background-color: #ffffff !important;
        color: #0a1118 !important;
        border: 2px solid #0a1118 !important;
        border-radius: 5px !important;
        padding: 0.5rem 1rem !important;
        font-size: 1.1rem !important;
        font-weight: 500 !important;
    }
    .stButton>button:hover {
        background-color: #f5f5f5 !important;
        border: 2px solid #0a1118 !important;
    }
    .stButton>button:active {
        background-color: #e8e8e8 !important;
    }


    /* Code cards */
    .code-card {
        background-color: #ffffff;
        border-left: 4px solid #0a1118;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(10, 17, 24, 0.1);
    }
    .code-header {
        font-size: 1.3rem;
        font-weight: bold;
        color: #0a1118;
    }

    /* Confidence scores */
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ff8c00;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }

    /* Text colors */
    .stMarkdown, .stText, p, span, label, h1, h2, h3 {
        color: #0a1118 !important;
    }

    /* Info/success boxes */
    .stAlert {
        background-color: #ffffff !important;
        color: #0a1118 !important;
    }
</style>
""", unsafe_allow_html=True)


def check_api_health() -> bool:
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_api_stats() -> Dict:
    """Get database statistics"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


def search_codes(query: str, mode: str, max_results: int) -> Dict:
    """Search for medical codes"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/code-suggestions",
            json={
                "clinical_description": query,
                "search_mode": mode,
                "max_results": max_results
            },
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
    except requests.exceptions.Timeout:
        return {"error": "Request timeout - API took too long to respond"}
    except Exception as e:
        return {"error": f"Connection error: {str(e)}"}


def get_confidence_class(score: float) -> str:
    """Get CSS class for confidence score"""
    if score >= 0.8:
        return "confidence-high"
    elif score >= 0.6:
        return "confidence-medium"
    else:
        return "confidence-low"


def display_code_card(code: Dict, code_type: str):
    """Display a code suggestion - simple text format"""
    confidence_class = get_confidence_class(code['confidence_score'])

    st.markdown(f"**{code['code']}** ({code_type}) - {code.get('category', 'N/A')}")
    st.markdown(f"{code['description']}")
    st.markdown(f"Confidence: <span class='{confidence_class}'>{code['confidence_score']:.0%}</span>", unsafe_allow_html=True)
    if code.get('reasoning'):
        st.markdown(f"*{code['reasoning']}*")
    st.markdown("---")


def project_overview_page():
    """Display project overview and technical details"""
    st.markdown('<h1 class="main-header">Project Overview</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Medical Coding RAG System - Technical Documentation</p>', unsafe_allow_html=True)
    st.divider()

    # Project Summary
    st.markdown("## Project Summary")
    st.markdown("""
    This project implements an AI-powered Medical Coding Assistant using **Retrieval-Augmented Generation (RAG)**
    to suggest accurate CPT and ICD-10 codes based on clinical descriptions. The system combines semantic search
    with hybrid retrieval techniques to provide fast, relevant code suggestions for medical billing and documentation.
    """)
    st.divider()

    # Architecture
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### System Architecture")
        st.markdown("""
        **Backend (FastAPI)**
        - RESTful API endpoints
        - Asynchronous request handling
        - Connection pooling
        - Response caching

        **Database (Neon PostgreSQL)**
        - pgvector extension for embeddings
        - 1,163 CPT codes
        - 74,260 ICD-10 codes
        - Vector similarity indices
        """)

    with col2:
        st.markdown("### Tech Stack")
        st.markdown("""
        **Core Technologies**
        - Python 3.11+
        - FastAPI (Backend)
        - Streamlit (Frontend)
        - PostgreSQL + pgvector

        **AI/ML Components**
        - Sentence Transformers (all-MiniLM-L6-v2)
        - 384-dimensional embeddings
        - Perplexity API (LLM reranking)
        - Hybrid search (vector + keyword)
        """)

    st.divider()

    # Features
    st.markdown("### Key Features")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **Hybrid Search**
        - Vector similarity search
        - Full-text keyword search
        - Reciprocal Rank Fusion
        - Confidence scoring
        """)

    with col2:
        st.markdown("""
        **Performance**
        - < 500ms response time
        - Query caching
        - Batch embeddings
        - Optimized indices
        """)

    with col3:
        st.markdown("""
        **User Experience**
        - Real-time suggestions
        - Confidence indicators
        - Multiple search modes
        - Light/clean UI
        """)

    st.divider()

    # How It Works
    st.markdown("### How It Works")
    st.markdown("""
    1. **User Input**: Clinical description entered via Streamlit interface
    2. **Embedding Generation**: Text converted to 384-dim vector using Sentence Transformers
    3. **Hybrid Search**:
       - Vector cosine similarity search on embeddings
       - Full-text search on descriptions
       - Results merged using Reciprocal Rank Fusion
    4. **Ranking**: Top matches scored and ranked by relevance
    5. **Response**: CPT and ICD-10 codes returned with confidence scores
    """)

    st.divider()

    # Database Schema
    st.markdown("### Database Schema")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **CPT Codes Table**
        - `cpt_code` (VARCHAR)
        - `description` (TEXT)
        - `category` (VARCHAR)
        - `code_status` (VARCHAR)
        - `embedding` (VECTOR[384])
        - `description_tsv` (TSVECTOR)
        """)

    with col2:
        st.markdown("""
        **ICD-10 Codes Table**
        - `icd10_code` (VARCHAR)
        - `description` (TEXT)
        - `chapter` (VARCHAR)
        - `block` (VARCHAR)
        - `embedding` (VECTOR[384])
        - `description_tsv` (TSVECTOR)
        """)

    st.divider()

    # Performance Metrics
    st.markdown("### Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Codes", "75,423")
    with col2:
        st.metric("Avg Response Time", "~300ms")
    with col3:
        st.metric("Embedding Dimension", "384")
    with col4:
        st.metric("Search Modes", "3")

    st.divider()

    # Future Enhancements
    st.markdown("### Future Enhancements")
    st.markdown("""
    - Real-time learning from user feedback
    - Analytics dashboard for code usage patterns
    - Integration with EHR systems
    - Multi-language support
    - Advanced LLM reasoning for complex cases
    - Mobile application
    - HIPAA compliance features
    - Performance monitoring and optimization
    """)


# Main app
def main():
    # Page selector in sidebar (check first before rendering anything)
    with st.sidebar:
        st.header("Settings")
        st.divider()
        st.subheader("Navigation")
        page = st.radio(
            "Select Page",
            ["Medical Coding Assistant", "Project Overview"],
            key="page_selector"
        )

    # Render the selected page
    if page == "Project Overview":
        project_overview_page()
        return

    # Medical Coding Assistant page (default)
    # Header
    st.markdown('<h1 class="main-header">Medical Coding Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered CPT and ICD-10 code suggestions using RAG</p>', unsafe_allow_html=True)

    # Description
    st.markdown("""
    **About Medical Coding:**

    Medical coding is essential for healthcare billing and insurance claims. This system uses two main code types:
    - **ICD-10 Codes** (International Classification of Diseases): Identify patient diagnoses, symptoms, and conditions
    - **CPT Codes** (Current Procedural Terminology): Describe medical procedures, services, and tests - *note: CPT codes in this demo are fictional/generated for demonstration purposes*

    In real-world practice, medical coders review clinical documentation and assign appropriate codes for accurate billing,
    insurance reimbursement, and medical record-keeping.
    """)

    # Example queries in columns
    st.markdown("**Example queries:**")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        - Patient with type 2 diabetes
        - Chest pain with hypertension
        - Annual wellness visit
        - Knee replacement surgery
        - Pneumonia with respiratory distress
        """)

    with col2:
        st.markdown("""
        - Routine colonoscopy screening
        - Migraine headache treatment
        - Urinary tract infection
        - Laceration repair on forehead
        - Influenza vaccination administration
        """)
    st.divider()

    # Sidebar (continued)
    with st.sidebar:
        st.divider()

        # API Health Check
        api_healthy = check_api_health()
        if api_healthy:
            st.success("API Connected")

            # Get stats
            stats = get_api_stats()
            if stats:
                st.info(f"""
                **Database Statistics:**
                - CPT Codes: {stats['total_cpt_codes']:,}
                - ICD-10 Codes: {stats['total_icd10_codes']:,}
                """)
        else:
            st.error("API Disconnected")
            st.warning("Make sure the backend is running:\n```\ncd backend\nuvicorn app.main:app --reload\n```")
            st.stop()

        # Use expert mode for AI reasoning
        search_mode = "expert"
        max_results = 5

    # Main content
    # Query input
    query = st.text_area(
        "Enter Clinical Description",
        value=st.session_state.get('query_input', ''),
        height=100,
        placeholder="e.g., Patient presents with type 2 diabetes and hypertension...",
        help="Describe the patient's condition, symptoms, or procedure"
    )

    search_button = st.button("Search Codes", type="primary")

    # Search and display results
    if search_button or (query and len(query) >= 10):
        if len(query) < 10:
            st.warning("Please enter at least 10 characters")
        else:
            with st.spinner("Searching for medical codes..."):
                start_time = time.time()
                results = search_codes(query, search_mode, max_results)
                elapsed = time.time() - start_time

                if "error" in results:
                    st.error(f"{results['error']}")
                else:
                    # Success message
                    st.success(f"Found {len(results.get('cpt_codes', []))} CPT and {len(results.get('icd10_codes', []))} ICD-10 codes in {elapsed:.2f}s")

                    # Display explanation (if available)
                    if results.get('explanation'):
                        with st.expander("Overall Explanation", expanded=True):
                            st.info(results['explanation'])

                    # Display results in tabs - ICD-10 first, then CPT
                    tab1, tab2 = st.tabs(["ICD-10 Codes (Diagnoses)", "CPT Codes (Procedures)"])

                    with tab1:
                        icd10_codes = results.get('icd10_codes', [])[:5]  # Max 5 codes
                        if icd10_codes:
                            # Top 2 recommendations with explanation
                            st.subheader("Top Recommendations")
                            for code in icd10_codes[:2]:
                                display_code_card(code, "ICD-10")

                            # Table for all codes
                            if len(icd10_codes) > 0:
                                st.subheader("All ICD-10 Codes")
                                table_data = []
                                for code in icd10_codes:
                                    table_data.append({
                                        "Code": code['code'],
                                        "Description": code['description'][:80] + "..." if len(code['description']) > 80 else code['description'],
                                        "Chapter": code.get('category', 'N/A'),
                                        "Confidence": f"{code['confidence_score']:.0%}",
                                        "Reasoning": code.get('reasoning', 'N/A')[:100] + "..." if code.get('reasoning') and len(code.get('reasoning', '')) > 100 else code.get('reasoning', 'N/A')
                                    })
                                st.table(table_data)
                        else:
                            st.warning("No ICD-10 codes found")

                    with tab2:
                        cpt_codes = results.get('cpt_codes', [])[:5]  # Max 5 codes
                        if cpt_codes:
                            # Top 2 recommendations with explanation
                            st.subheader("Top Recommendations")
                            for code in cpt_codes[:2]:
                                display_code_card(code, "CPT")

                            # Table for all codes
                            if len(cpt_codes) > 0:
                                st.subheader("All CPT Codes")
                                table_data = []
                                for code in cpt_codes:
                                    table_data.append({
                                        "Code": code['code'],
                                        "Description": code['description'][:80] + "..." if len(code['description']) > 80 else code['description'],
                                        "Category": code.get('category', 'N/A'),
                                        "Confidence": f"{code['confidence_score']:.0%}",
                                        "Reasoning": code.get('reasoning', 'N/A')[:100] + "..." if code.get('reasoning') and len(code.get('reasoning', '')) > 100 else code.get('reasoning', 'N/A')
                                    })
                                st.table(table_data)
                        else:
                            st.warning("No CPT codes found")

                    # Processing info
                    st.divider()
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Processing Time", f"{results['processing_time_ms']:.0f}ms")
                    with col2:
                        st.metric("Total Results", len(cpt_codes) + len(icd10_codes))


if __name__ == "__main__":
    # Initialize session state
    if 'query_input' not in st.session_state:
        st.session_state.query_input = ""

    main()
