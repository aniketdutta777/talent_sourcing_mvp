import streamlit as st
import os
import shutil # For clearing the database folder

# Import functions and variables from our core_logic.py
from core_logic import initialize_database, query_database, chroma_client, COLLECTION_NAME, DATABASE_DIR

# --- Streamlit Page Configuration ---
# This sets up how your web page will look in the browser
st.set_page_config(layout="wide", page_title="LLM Talent Sourcing MVP (Simple)")

st.title("🌟 LLM-Powered Talent Sourcing (Simplified MVP)")
st.subheader("Query a pre-generated resume database with AI.")

# --- Sidebar for Database Setup ---
st.sidebar.header("1. Database Setup")

# Slider to choose how many mock resumes to create
num_mock_resumes = st.sidebar.slider("Number of mock resumes to create:", 50, 1000, 100)

# Check if the database is already initialized
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
            initialize_database(num_mock_resumes) # Call the function from core_logic.py
            st.sidebar.success("Database initialized successfully!")
            st.rerun() # Rerun the Streamlit app to update the UI (disable button, show count)
        except Exception as e:
            st.sidebar.error(f"Error initializing database: {e}")

if st.sidebar.button("Clear Database", disabled=not db_initialized):
    with st.spinner("Clearing database..."):
        try:
            chroma_client.delete_collection(name=COLLECTION_NAME)
            import shutil
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
    "Enter your hiring manager query (e.g., 'Senior Software Engineer with expertise in Python, AWS, and Machine Learning, with experience in FinTech. Must have led a team.')",
    "Looking for a Mid-level Marketing Manager with B2B SaaS experience, strong in lead generation and content strategy."
)

num_results_llm_review = st.slider(
    "Number of top similar profiles for the AI to review in detail:",
    min_value=3, max_value=15, value=7,
    help="The system first finds this many semantically similar profiles, then the AI reviews them to give you refined recommendations."
)

if st.button("Find Matching Profiles", type="primary", disabled=not db_initialized):
    if not query:
        st.warning("Please enter your search query.")
    else:
        with st.spinner("Searching and analyzing profiles based on your query..."):
            result_text = query_database(query, num_results=num_results_llm_review)

            st.markdown("### AI's Analysis & Top Recommendations:")
            st.markdown(result_text)

        st.success("Search complete!")
elif not db_initialized:
    st.info("Please initialize the database from the sidebar before searching.")

st.markdown("---")
st.markdown("💡 *This simplified MVP demonstrates the core LLM-powered search. For a full product, you'd add client management, robust data parsing, and scalable cloud infrastructure.*")