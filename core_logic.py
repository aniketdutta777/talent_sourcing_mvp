import json
import os
import uuid
import chromadb
from openai import OpenAI
from anthropic import Anthropic
import random

COLLECTION_NAME = "all_resumes"
DATABASE_DIR = "./mock_resume_database"

# --- GLOBAL CLIENTS ---
global_client_openai = None
global_client_anthropic = None

def initialize_api_clients():
    """Initializes API clients with keys from environment variables."""
    global global_client_openai, global_client_anthropic
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    global_client_openai = OpenAI(api_key=OPENAI_API_KEY)
    global_client_anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)

# --- DATABASE SETUP ---
try:
    chroma_client = chromadb.Client()
    print("DEBUG: ChromaDB client initialized.")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to initialize ChromaDB client: {e}. Falling back to local.")
    chroma_client = chromadb.PersistentClient(path=os.path.join(DATABASE_DIR, "chroma_db"))

def get_embedding(text, model="text-embedding-3-small"):
    """Generates an embedding for the given text."""
    if global_client_openai is None:
        raise RuntimeError("OpenAI client not initialized.")
    text = text.replace("\n", " ")
    return global_client_openai.embeddings.create(input=[text], model=model).data[0].embedding

def generate_fake_resume_data(num_resumes=100):
    # This function remains the same, providing mock data for Lark's Database
    resumes = []
    # ... (rest of the function is unchanged)
    return resumes

def initialize_database(num_resumes=100):
    """Populates the ChromaDB database with mock resumes if it's empty."""
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    if collection.count() > 0:
        print(f"Database already initialized with {collection.count()} resumes.")
        return
    # ... (rest of the function is unchanged)

# --- SEARCH LOGIC FOR LARK'S DATABASE ---
def search_lark_database(user_query: str, num_profiles_to_retrieve: int) -> dict:
    """Encapsulates the original search logic for the central database."""
    # This is the original _perform_claude_search_with_tool_internal function, now renamed
    system_message = """You are an expert HR recruitment assistant... (rest of prompt)""" # Abbreviated
    messages = [{"role": "user", "content": user_query}]
    # ... (The entire Claude tool-use workflow from the previous version goes here)
    # This ensures existing functionality is preserved.
    # For brevity, the full implementation is assumed from the previous correct version.
    # This function will return the final dictionary with status, analysis_data, and usage.
    
    # Returning a placeholder for now to represent the full logic block.
    # IN YOUR REAL FILE, PASTE THE ENTIRE `_perform_claude_search_with_tool_internal` LOGIC HERE
    print("--- Firing search against Lark's Database ---")
    # This is where the full logic from your previous file would go.
    # For now, we return a mock response.
    return {"status": "success", "analysis_data": {"overall_summary": "Search of Lark's database was successful.", "candidates": [], "overall_recommendation": "N/A"}, "usage": {"input_tokens": 100, "output_tokens": 100}}


# --- NEW: PLACEHOLDER FOR GOOGLE DRIVE SEARCH LOGIC ---
def search_google_drive(user_query: str, num_profiles_to_retrieve: int, folder_ids: list, user_id: str) -> dict:
    """
    Placeholder for the logic that will search files in a user's Google Drive.
    """
    print(f"--- Firing search against Google Drive for user {user_id} ---")
    print(f"Targeting folders: {folder_ids}")
    # In the next step, we will build the logic here to:
    # 1. Use the user's token to connect to the Google Drive API.
    # 2. List and download files from the specified folder_ids.
    # 3. Process these files (extract text, create embeddings).
    # 4. Store them in ChromaDB with the user_id in the metadata.
    # 5. Perform a filtered search on ChromaDB for that user_id.
    # 6. Send the results to Claude for analysis.
    
    # For now, return a placeholder message.
    return {"status": "success", "analysis_data": {"overall_summary": "Google Drive search is not yet implemented.", "candidates": [], "overall_recommendation": "Please check back later."}, "usage": {"input_tokens": 0, "output_tokens": 0}}


# --- MAIN ROUTER FUNCTION ---
def perform_claude_search_with_tool(user_query: str, num_profiles_to_retrieve: int, source: str, folder_ids: list, user_id: str) -> dict:
    """
    Acts as a router, directing the search request to the appropriate data source.
    """
    if global_client_anthropic is None:
        raise RuntimeError("Anthropic client not initialized.")

    if source == "Lark's Database":
        return search_lark_database(user_query, num_profiles_to_retrieve)
    
    elif source == "Google Drive":
        return search_google_drive(user_query, num_profiles_to_retrieve, folder_ids, user_id)

    elif source == "Both":
        # For now, we will just search Lark's DB for "Both".
        # A real implementation would merge results from both sources.
        print("--- Source 'Both' selected, defaulting to Lark's Database for MVP ---")
        return search_lark_database(user_query, num_profiles_to_retrieve)
        
    else:
        return {"status": "error", "message": f"Invalid source specified: {source}"}