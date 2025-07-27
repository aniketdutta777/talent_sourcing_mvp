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

def initialize_chroma_client():
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

def generate_fake_resume_data(num_resumes=100):
    # This function is correct and remains unchanged
    pass

def initialize_database(num_resumes=100):
    # This function is correct and remains unchanged
    pass

def resume_search_tool(query: str, num_results: int = 5, level: str = None, industry: str = None) -> list[dict]:
    # This function is correct and remains unchanged
    pass

resume_search_tool_schema = {
    # This schema is correct and remains unchanged
}

# --- CORRECTED LARK DATABASE SEARCH ---
def search_lark_database(user_query: str, num_profiles_to_retrieve: int) -> dict:
    print("--- Firing search against Lark's Database ---")
    # --- FIX: The example JSON now includes a full candidate object for clarity ---
    system_message = """You are an expert HR recruitment assistant. Use the `resume_search_tool` to find candidates. After using the tool, analyze the results and provide a summary. If the tool returns no candidates, inform the user clearly. YOUR FINAL OUTPUT MUST BE VALID JSON. Structure your response as follows:
```json
{
  "overall_summary": "Overall summary of the search results and candidate quality.",
  "candidates": [
    {
      "name": "Candidate Name",
      "contact_information": {
        "email": "candidate@example.com",
        "phone": "(123) 555-1234"
      },
      "summary": "A concise summary of why this candidate is a good fit, referencing their skills and experience against the job query.",
      "resume_pdf_url": "[https://example.com/resumes/candidate_id.pdf](https://example.com/resumes/candidate_id.pdf)"
    }
  ],
  "overall_recommendation": "Final thoughts on the candidate pool and a recommendation for who to interview first."
}
```"""
    messages = [{"role": "user", "content": user_query}]
    try:
        response = global_client_anthropic.messages.create(model="claude-3-haiku-20240307", max_tokens=2000, temperature=0.0, tools=[resume_search_tool_schema], messages=messages, system=system_message)
        if response.stop_reason == "tool_use":
            tool_use = next((block for block in response.content if block.type == "tool_use"), None)
            if not tool_use: return {"status": "error", "message": "Claude indicated tool use, but no tool was specified."}
            tool_output = resume_search_tool(**tool_use.input)
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_use.id, "content": json.dumps(tool_output)}]})
            final_response = global_client_anthropic.messages.create(model="claude-3-haiku-20240307", max_tokens=2000, temperature=0.5, messages=messages, system=system_message)
            json_string = final_response.content[0].text.strip().lstrip("```json").rstrip("```")
            parsed_json = json.loads(json_string)
            usage_data = {"input_tokens": final_response.usage.input_tokens, "output_tokens": final_response.usage.output_tokens}
            return {"status": "success", "analysis_data": parsed_json, "usage": usage_data}
    except Exception as e:
        return {"status": "error", "message": f"LLM analysis failed: {e}"}
    return {"status": "error", "message": "Claude did not use the tool as expected."}

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

