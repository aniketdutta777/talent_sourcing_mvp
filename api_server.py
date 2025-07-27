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
    global global_client_openai, global_client_anthropic
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    global_client_openai = OpenAI(api_key=OPENAI_API_KEY)
    global_client_anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)

# --- DATABASE SETUP ---
try:
    chroma_client = chromadb.Client()
except Exception as e:
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

# --- FILLED-IN SEARCH LOGIC FOR LARK'S DATABASE ---
def search_lark_database(user_query: str, num_profiles_to_retrieve: int) -> dict:
    print("--- Firing search against Lark's Database ---")
    system_message = """You are an expert HR recruitment assistant. Use the `resume_search_tool` to find candidates. After using the tool, analyze the results and provide a summary. If the tool returns no candidates, inform the user clearly. YOUR FINAL OUTPUT MUST BE VALID JSON. Structure your response as follows:
```json
{
  "overall_summary": "Overall summary of the search results and candidate quality.",
  "candidates": [],
  "overall_recommendation": "Final thoughts on the candidate pool and a recommendation for who to interview first."
}
```"""
    messages = [{"role": "user", "content": user_query}]
    try:
        response = global_client_anthropic.messages.create(
            model="claude-3-haiku-20240307", max_tokens=2000, temperature=0.0,
            tools=[resume_search_tool_schema], messages=messages, system=system_message
        )
    except Exception as e:
        return {"status": "error", "message": f"Error initiating conversation with Claude: {e}"}

    if response.stop_reason == "tool_use":
        tool_use = next((block for block in response.content if block.type == "tool_use"), None)
        if not tool_use: return {"status": "error", "message": "Claude indicated tool use, but no tool was specified."}
        
        tool_output = resume_search_tool(**tool_use.input)
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_use.id, "content": json.dumps(tool_output)}]})

        try:
            final_response = global_client_anthropic.messages.create(
                model="claude-3-haiku-20240307", max_tokens=2000, temperature=0.5,
                messages=messages, system=system_message
            )
            json_string = final_response.content[0].text.strip().lstrip("```json").rstrip("```").strip()
            parsed_json = json.loads(json_string)
            usage_data = {"input_tokens": final_response.usage.input_tokens, "output_tokens": final_response.usage.output_tokens}
            return {"status": "success", "analysis_data": parsed_json, "usage": usage_data}
        except Exception as e:
            raw_output = final_response.content[0].text if final_response.content else "No content"
            return {"status": "error", "message": f"LLM analysis failed: {e}", "raw_llm_output": raw_output}

    return {"status": "error", "message": "Claude did not use the tool as expected."}

# --- PLACEHOLDER FOR GOOGLE DRIVE SEARCH LOGIC ---
def search_google_drive(user_query: str, num_profiles_to_retrieve: int, folder_ids: list, user_id: str, token: dict) -> dict:
    print(f"--- Firing search against Google Drive for user {user_id} ---")
    print(f"Targeting folders: {folder_ids}")
    return {"status": "success", "analysis_data": {"overall_summary": "Google Drive search is not yet implemented.", "candidates": [], "overall_recommendation": "Please check back later."}, "usage": {"input_tokens": 0, "output_tokens": 0}}

# --- MAIN ROUTER FUNCTION ---
def perform_claude_search_with_tool(user_query: str, num_profiles_to_retrieve: int, source: str, folder_ids: list, user_id: str, token: dict) -> dict:
    if global_client_anthropic is None: raise RuntimeError("Anthropic client not initialized.")
    if source == "Lark's Database":
        return search_lark_database(user_query, num_profiles_to_retrieve)
    elif source == "Google Drive":
        return search_google_drive(user_query, num_profiles_to_retrieve, folder_ids, user_id, token)
    elif source == "Both":
        print("--- Source 'Both' selected, defaulting to Lark's Database for MVP ---")
        return search_lark_database(user_query, num_profiles_to_retrieve)
    else:
        return {"status": "error", "message": f"Invalid source specified: {source}"}