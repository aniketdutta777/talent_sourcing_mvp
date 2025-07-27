import json
import os
import uuid
import chromadb
import io
import fitz # PyMuPDF
from openai import OpenAI
from anthropic import Anthropic
import random
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

COLLECTION_NAME = "all_resumes"
GDRIVE_COLLECTION_NAME = "gdrive_resumes"
DATABASE_DIR = "./mock_resume_database"

# --- GLOBAL CLIENTS (Initialized to None) ---
global_client_openai = None
global_client_anthropic = None
chroma_client = None # <-- MOVED INITIALIZATION OUT OF GLOBAL SCOPE

# --- INITIALIZATION FUNCTIONS ---
def initialize_api_clients():
    """Initializes API clients with keys from environment variables."""
    global global_client_openai, global_client_anthropic
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    global_client_openai = OpenAI(api_key=OPENAI_API_KEY)
    global_client_anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)
    print("DEBUG: OpenAI and Anthropic clients initialized.")

def initialize_chroma_client():
    """Initializes the ChromaDB client."""
    global chroma_client
    try:
        chroma_client = chromadb.Client()
        print("DEBUG: ChromaDB client initialized.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to initialize ChromaDB client: {e}. Falling back to local.")
        chroma_client = chromadb.PersistentClient(path=os.path.join(DATABASE_DIR, "chroma_db"))

def get_embedding(text, model="text-embedding-3-small"):
    if global_client_openai is None: raise RuntimeError("OpenAI client not initialized.")
    text = text.replace("\n", " ")
    return global_client_openai.embeddings.create(input=[text], model=model).data[0].embedding

# --- (generate_fake_resume_data remains the same) ---

def initialize_database(num_resumes=100):
    if chroma_client is None: raise RuntimeError("Chroma client not initialized.")
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    if collection.count() > 0:
        print(f"Database already initialized.")
        return
    # ... (rest of database initialization logic) ...

# --- (The rest of your functions: resume_search_tool, search_lark_database, search_google_drive, etc. remain the same) ---
# --- For brevity, they are not repeated here, but should be in your file. ---

def perform_claude_search_with_tool(user_query: str, num_profiles_to_retrieve: int, source: str, folder_ids: list, user_id: str, token: dict) -> dict:
    if source == "Lark's Database":
        return search_lark_database(user_query, num_profiles_to_retrieve)
    elif source == "Google Drive":
        return search_google_drive(user_query, num_profiles_to_retrieve, folder_ids, user_id, token)
    elif source == "Both":
        print("--- Source 'Both' selected, defaulting to Google Drive search for MVP ---")
        return search_google_drive(user_query, num_profiles_to_retrieve, folder_ids, user_id, token)
    else:
        return {"status": "error", "message": f"Invalid source specified: {source}"}