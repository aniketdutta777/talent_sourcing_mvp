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

COLLECTION_NAME = "lark_static_resumes"
GDRIVE_COLLECTION_NAME = "gdrive_resumes"
DATABASE_DIR = "./lark_db" # Using a clear directory name

STATIC_RESUME_DATA = [
    {"id": "dev-001", "name": "Alice Anderson", "email": "alice.a@example.com", "phone": "(123) 555-0101", "pdf_url": "https://example.com/resumes/dev-001.pdf", "job_title": "Senior Software Engineer", "industry": "Tech", "level": "Senior", "skills": ["Python", "AWS", "SQL", "DevOps", "FastAPI"], "raw_text": "Alice Anderson | Senior Software Engineer with 8 years of experience in the Tech industry. Expert in Python, AWS cloud services, and building scalable backend systems with FastAPI. Proven track record in leading DevOps practices and database management with SQL."},
    {"id": "pm-001", "name": "Bob Bannon", "email": "bob.b@example.com", "phone": "(123) 555-0102", "pdf_url": "https://example.com/resumes/pm-001.pdf", "job_title": "Product Manager", "industry": "SaaS", "level": "Mid", "skills": ["Product Management", "Agile", "JIRA", "Roadmap Planning", "User Research"], "raw_text": "Bob Bannon | Mid-level Product Manager with 4 years of experience in the B2B SaaS industry. Skilled in Agile methodologies, roadmap planning, and conducting user research to drive product decisions. Proficient with JIRA and other product management tools."},
    {"id": "mkt-001", "name": "Charlie Clark", "email": "charlie.c@example.com", "phone": "(123) 555-0103", "pdf_url": "https://example.com/resumes/mkt-001.pdf", "job_title": "Marketing Manager", "industry": "E-commerce", "level": "Junior", "skills": ["Marketing Strategy", "SEO", "Content Creation", "Google Analytics"], "raw_text": "Charlie Clark | Junior Marketing Manager with 2 years of experience in the E-commerce sector. Specializes in content creation and SEO. Certified in Google Analytics and passionate about data-driven marketing strategies."}
]

global_client_openai = None
global_client_anthropic = None
chroma_client = None

def initialize_api_clients():
    global global_client_openai, global_client_anthropic
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    global_client_openai = OpenAI(api_key=OPENAI_API_KEY)
    global_client_anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)
    print("DEBUG: OpenAI and Anthropic clients initialized.")

# --- CORRECTED DATABASE INITIALIZATION ---
def initialize_chroma_client():
    """
    Initializes a persistent ChromaDB client to ensure data is shared across all server workers.
    """
    global chroma_client
    # Always use a persistent client in a server environment
    chroma_client = chromadb.PersistentClient(path=DATABASE_DIR)
    print(f"DEBUG: Persistent ChromaDB client initialized at path: {DATABASE_DIR}")

def get_embedding(text, model="text-embedding-3-small"):
    if global_client_openai is None: raise RuntimeError("OpenAI client not initialized.")
    text = text.replace("\n", " ")
    return global_client_openai.embeddings.create(input=[text], model=model).data[0].embedding

def initialize_database():
    if chroma_client is None: raise RuntimeError("Chroma client not initialized.")
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    if collection.count() == 0:
        print(f"Database collection '{COLLECTION_NAME}' is empty. Populating with static data...")
        resumes_data = STATIC_RESUME_DATA
        for resume in resumes_data:
            collection.add(
                embeddings=[get_embedding(resume["raw_text"])],
                documents=[resume["raw_text"]],
                metadatas=[{"resume_id": resume["id"], "name": resume["name"], "email": resume["email"], "phone": resume["phone"], "pdf_url": resume["pdf_url"], "job_title": resume["job_title"], "level": resume["level"], "industry": resume["industry"], "skills": ", ".join(resume["skills"])}],
                ids=[resume["id"]]
            )
        print(f"Successfully added {len(resumes_data)} static resumes to ChromaDB.")
    else:
        print(f"Static database already initialized with {collection.count()} resumes.")

def resume_search_tool(query: str, num_results: int = 5, level: str = None, industry: str = None) -> list[dict]:
    # This function is correct and remains unchanged
    pass

resume_search_tool_schema = {
    # This schema is correct and remains unchanged
}

def search_lark_database(user_query: str, num_profiles_to_retrieve: int) -> dict:
    # This function is correct and remains unchanged
    pass

def _get_google_drive_service(token_data: dict):
    # This function is correct and remains unchanged
    pass

def _extract_folder_id_from_url(url: str) -> str:
    # This function is correct and remains unchanged
    pass

def _extract_text_from_pdf(pdf_content: bytes) -> str:
    # This function is correct and remains unchanged
    pass

def search_google_drive(user_query: str, num_profiles_to_retrieve: int, folder_ids: list, user_id: str, token: dict) -> dict:
    # This function is correct and remains unchanged
    pass

def perform_claude_search_with_tool(user_query: str, num_profiles_to_retrieve: int, source: str, folder_ids: list, user_id: str, token: dict) -> dict:
    # This function is correct and remains unchanged
    pass