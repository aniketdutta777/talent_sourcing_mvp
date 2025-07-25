import streamlit as st
import os
import shutil 

# Import functions and variables from our core_logic.py
# Note: We now import perform_claude_search_with_tool instead of query_database
from core_logic import initialize_database, perform_claude_search_with_tool, chroma_client, COLLECTION_NAME, DATABASE_DIR

# --- Streamlit Page Configuration ---
st.set_page_config(layout="wide", page_title="LLM Talent Sourcing MVP (Tool Use)")

st.title("🌟 LLM-Powered Talent Sourcing (Tool Use MVP)")
st.subheader("Leveraging Claude's tool use to query the resume database.")

# --- Sidebar for Database Setup ---
st.sidebar.header("1. Database Setup")
    
num_mock_resumes = st.sidebar.slider("Number of mock resumes to create:", 50, 1000, 100)
    
db_initialized = False
try:
    collection_check = chroma_client.get_collection(name=COLLECTION_NAME)
    if collection_check.count() > 0:
        db_initialized = True
        st.sidebar.success(f"Database initialized with {collection_check.count()} resumes.")
    else:
        st.sidebar.info("Database is currently empty.")
except Exception:
    st.sidebar.warning("ChromaDB collection not found. Database might need initialization.")

if st.sidebar.button(f"Initialize Database with {num_mock_resumes} Resumes", disabled=db_initialized):
    with st.spinner(f"Initializing database with {num_mock_resumes} mock resumes... This might take a moment."):
        try:
            initialize_database(num_mock_resumes)
            st.sidebar.success("Database initialized successfully!")
            st.rerun() 
        except Exception as e:
            st.sidebar.error(f"Error initializing database: {e}. Check your API keys and internet.")

if st.sidebar.button("Clear Database", disabled=not db_initialized):
    with st.spinner("Clearing database..."):
        try:
            chroma_client.delete_collection(name=COLLECTION_NAME)
            raw_resumes_path = os.path.join(DATABASE_DIR, "raw_resumes")
            if os.path.exists(raw_resumes_path):
                shutil.rmtree(raw_resumes_path)
            st.sidebar.success("Database cleared!")
            st.rerun() 
        except Exception as e:
            st.sidebar.error(f"Error clearing database: {e}")
            
st.sidebar.markdown("---") 

# --- Main Area for Querying ---
st.header("2. Query the Resume Database")
    
query = st.text_area(
    "Enter your hiring manager query (e.g., 'Find a Senior Software Engineer with expertise in Python, AWS, and Machine Learning, with experience in FinTech. Must have led a team.')",
    "Looking for a Mid-level Marketing Manager with B2B SaaS experience, strong in lead generation and content strategy. Can you find some candidates for me?" # Added a more conversational query
)

# Note: The num_results_llm_review slider will now control the 'num_profiles_to_retrieve'
# parameter passed to perform_claude_search_with_tool, which in turn limits the tool's output.
num_profiles_to_retrieve = st.slider(
    "Number of top similar profiles for the AI to retrieve and analyze:",
    min_value=3, max_value=15, value=7,
    help="This controls how many candidates Claude's search tool will retrieve for its analysis."
)

if st.button("Find Matching Profiles", type="primary", disabled=not db_initialized):
    if not query:
        st.warning("Please enter your search query.")
    else:
        with st.spinner("Asking Claude to find and analyze profiles..."):
            # Call the new orchestration function
            result_text = perform_claude_search_with_tool(query, num_profiles_to_retrieve=num_profiles_to_retrieve)
            
            st.markdown("### AI's Analysis & Top Recommendations:")
            st.markdown(result_text)
        
        st.success("Search complete!")
elif not db_initialized:
    st.info("Please initialize the database from the sidebar before searching.")

st.markdown("---")
st.markdown("💡 *This MVP demonstrates Claude's ability to use a custom tool to query your database. This is a step towards more advanced AI agents!*")