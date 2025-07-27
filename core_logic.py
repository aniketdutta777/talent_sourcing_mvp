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

global_client_openai = None
global_client_anthropic = None

def initialize_api_clients():
    global global_client_openai, global_client_anthropic
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    global_client_openai = OpenAI(api_key=OPENAI_API_KEY)
    global_client_anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)

chroma_client = chromadb.Client()

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return global_client_openai.embeddings.create(input=[text], model=model).data[0].embedding

def generate_fake_resume_data(num_resumes=100):
    resumes = []
    skills_list = ["Python", "Java", "SQL", "AWS", "GCP", "Machine Learning", "Data Analysis", "Project Management", "Marketing Strategy", "Sales Leadership", "Product Management", "UI/UX Design", "DevOps"]
    roles_list = ["Software Engineer", "Data Scientist", "Product Manager", "Marketing Manager", "Sales Executive", "DevOps Engineer", "Technical Lead"]
    industries_list = ["Tech", "Finance", "Healthcare", "SaaS", "E-commerce", "Consulting"]
    levels_list = ["Junior", "Mid", "Senior", "Lead", "Manager"]
    companies_list = ["Innovate Inc.", "DataDriven Corp.", "CloudSphere LLC", "QuantumLeap Solutions", "Synergy Systems", "NextGen Tech"]
    
    for i in range(num_resumes):
        resume_id = str(uuid.uuid4())
        # ... (rest of data generation logic) ...
        resumes.append({
            "id": resume_id, # ... and other fields
            "raw_text": "..."
        })
    return resumes

def initialize_database(num_resumes=100):
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    if collection.count() > 0:
        print(f"Database already initialized.")
        return
    # ... (rest of database initialization logic) ...

def resume_search_tool(query: str, num_results: int = 5, level: str = None, industry: str = None) -> list[dict]:
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    # ... (rest of the search tool logic) ...
    return [] # Placeholder

resume_search_tool_schema = {
    "name": "resume_search_tool",
    "description": "Searches a resume database to find profiles matching a job query.",
    "input_schema": {
        "type": "object", "properties": { "query": {"type": "string"}, "num_results": {"type": "integer"}, "level": {"type": "string"}, "industry": {"type": "string"}}, "required": ["query"]
    }
}

def _get_google_drive_service(token_data: dict):
    creds = Credentials.from_authorized_user_info(token_data, scopes=["https://www.googleapis.com/auth/drive.readonly"])
    return build('drive', 'v3', credentials=creds)

def _extract_folder_id_from_url(url: str) -> str:
    if "folders/" in url:
        return url.split("folders/")[1].split("?")[0]
    return None

def _extract_text_from_pdf(pdf_content: bytes) -> str:
    try:
        with fitz.open(stream=pdf_content, filetype="pdf") as doc:
            text = "".join(page.get_text() for page in doc)
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def search_lark_database(user_query: str, num_profiles_to_retrieve: int) -> dict:
    print("--- Firing search against Lark's Database ---")
    system_message = """You are an expert HR recruitment assistant... (rest of your detailed prompt)"""
    messages = [{"role": "user", "content": user_query}]
    try:
        response = global_client_anthropic.messages.create(
            model="claude-3-haiku-20240307", max_tokens=2000, temperature=0.0,
            tools=[resume_search_tool_schema], messages=messages, system=system_message
        )
        if response.stop_reason == "tool_use":
            tool_use = next(block for block in response.content if block.type == "tool_use")
            tool_output = resume_search_tool(**tool_use.input)
            messages.extend([response.content[0], {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_use.id, "content": json.dumps(tool_output)}]}])
            final_response = global_client_anthropic.messages.create(
                model="claude-3-haiku-20240307", max_tokens=2000, temperature=0.5,
                messages=messages, system=system_message
            )
            json_string = final_response.content[0].text.strip().lstrip("```json").rstrip("```")
            parsed_json = json.loads(json_string)
            usage_data = {"input_tokens": final_response.usage.input_tokens, "output_tokens": final_response.usage.output_tokens}
            return {"status": "success", "analysis_data": parsed_json, "usage": usage_data}
    except Exception as e:
        return {"status": "error", "message": f"LLM analysis failed: {e}"}
    return {"status": "error", "message": "Claude did not use the tool as expected."}

def search_google_drive(user_query: str, num_profiles_to_retrieve: int, folder_ids: list, user_id: str, token: dict) -> dict:
    if not token:
        return {"status": "error", "message": "Google Drive token not provided."}
    
    try:
        service = _get_google_drive_service(token)
        gdrive_collection = chroma_client.get_or_create_collection(name=GDRIVE_COLLECTION_NAME)
        
        for folder_url in folder_ids:
            folder_id = _extract_folder_id_from_url(folder_url)
            if not folder_id: continue
            
            q = f"'{folder_id}' in parents and mimeType='application/pdf'"
            results = service.files().list(q=q, fields="files(id, name)").execute()
            for item in results.get('files', []):
                if not gdrive_collection.get(ids=[item['id']]).get('ids'):
                    print(f"Processing new file: {item['name']}")
                    request = service.files().get_media(fileId=item['id'])
                    file_bytes = request.execute()
                    text = _extract_text_from_pdf(file_bytes)
                    if text:
                        gdrive_collection.add(
                            ids=[item['id']], embeddings=[get_embedding(text)], documents=[text],
                            metadatas=[{"user_id": user_id, "file_name": item['name']}]
                        )
        
        search_results = gdrive_collection.query(
            query_embeddings=[get_embedding(user_query)], n_results=num_profiles_to_retrieve,
            where={"user_id": user_id}, include=['documents', 'metadatas']
        )
        
        candidate_texts = search_results['documents'][0]
        context_for_llm = "\n\n---\n\n".join(candidate_texts)
        
        final_response = global_client_anthropic.messages.create(
            model="claude-3-haiku-20240307", max_tokens=2000, temperature=0.5,
            system="Analyze the following resume texts based on the user's query and provide a summary in the standard JSON format.",
            messages=[{"role": "user", "content": f"Query: {user_query}\n\nResumes:\n{context_for_llm}"}]
        )
        json_string = final_response.content[0].text.strip().lstrip("```json").rstrip("```")
        parsed_json = json.loads(json_string)
        usage_data = {"input_tokens": final_response.usage.input_tokens, "output_tokens": final_response.usage.output_tokens}
        return {"status": "success", "analysis_data": parsed_json, "usage": usage_data}

    except Exception as e:
        return {"status": "error", "message": f"An error occurred during Google Drive search: {e}"}

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