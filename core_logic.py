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
DATABASE_DIR = "./lark_db"

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

def initialize_chroma_client():
    global chroma_client
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
    if chroma_client is None: raise RuntimeError("Chroma client not initialized.")
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    embedding = get_embedding(query)
    filter_conditions = []
    if level: filter_conditions.append({"level": {"$eq": level}})
    if industry: filter_conditions.append({"industry": {"$eq": industry}})
    where_filter = None
    if len(filter_conditions) > 1: where_filter = {"$and": filter_conditions}
    elif len(filter_conditions) == 1: where_filter = filter_conditions[0]
    results = collection.query(query_embeddings=[embedding], n_results=num_results, where=where_filter, include=['metadatas', 'documents'])
    candidates_data = []
    if results and results['ids'] and results['ids'][0]:
        print(f"Tool: Found {len(results['ids'][0])} potential candidates matching filters.")
        for i, metadata in enumerate(results['metadatas'][0]):
            candidates_data.append({"name": metadata.get("name"), "contact_information": {"email": metadata.get("email"), "phone": metadata.get("phone")}, "resume_pdf_url": metadata.get("pdf_url"), "raw_resume_text": results['documents'][0][i]})
    else:
        print("Tool: No candidates found for the query with the specified filters.")
        return [{"message": "No candidates found matching the search criteria and filters."}]
    return candidates_data

resume_search_tool_schema = {
    "name": "resume_search_tool", "description": "Searches a resume database to find profiles matching a job query.",
    "input_schema": {"type": "object", "properties": {"query": {"type": "string"}, "num_results": {"type": "integer"}, "level": {"type": "string"}, "industry": {"type": "string"}},"required": ["query"]}
}

def search_lark_database(user_query: str, num_profiles_to_retrieve: int) -> dict:
    print("--- Firing search against Lark's Database ---")
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
  "overall_recommendation": "Final thoughts on the candidate pool."
}
```"""
    messages = [{"role": "user", "content": user_query}]
    try:
        response = global_client_anthropic.messages.create(model="claude-3-haiku-20240307", max_tokens=2000, temperature=0.0, tools=[resume_search_tool_schema], messages=messages, system=system_message)
        if response.stop_reason == "tool_use":
            tool_use = next((block for block in response.content if block.type == "tool_use"), None)
            if not tool_use: return {"status": "error", "message": "Claude indicated tool use, but no tool was specified."}
            tool_output = resume_search_tool(**tool_use.input)
            if (isinstance(tool_output, list) and len(tool_output) > 0 and tool_output[0].get("message", "").startswith("No candidates found")):
                return {"status": "success", "analysis_data": {"overall_summary": "The initial search did not find any relevant candidates in the database for this query.", "candidates": [], "overall_recommendation": "Try broadening your search terms."}, "usage": {"input_tokens": 0, "output_tokens": 0}}
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_use.id, "content": json.dumps(tool_output)}]})
            final_response = global_client_anthropic.messages.create(model="claude-3-haiku-20240307", max_tokens=2000, temperature=0.5, messages=messages, system=system_message)
            json_string = final_response.content[0].text.strip().lstrip("```json").rstrip("```").strip()
            if not json_string: return {"status": "error", "message": "The AI failed to produce an analysis for the found candidates."}
            parsed_json = json.loads(json_string)
            usage_data = {"input_tokens": final_response.usage.input_tokens, "output_tokens": final_response.usage.output_tokens}
            return {"status": "success", "analysis_data": parsed_json, "usage": usage_data}
        elif response.stop_reason == "end_turn":
            return {"status": "success", "analysis_data": {"overall_summary": f"The AI provided a direct response: {response.content[0].text}", "candidates": [], "overall_recommendation": "No candidates were searched."}, "usage": {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}}
    except Exception as e:
        return {"status": "error", "message": f"LLM analysis failed: {e}"}
    return {"status": "error", "message": f"Unexpected response from Claude with stop reason: {response.stop_reason}"}

def _get_google_drive_service(token_data: dict):
    GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
    GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
    info = {**token_data, "client_id": GOOGLE_CLIENT_ID, "client_secret": GOOGLE_CLIENT_SECRET, "token_uri": "https://oauth2.googleapis.com/token"}
    creds = Credentials.from_authorized_user_info(info)
    return build('drive', 'v3', credentials=creds)

def _extract_folder_id_from_url(url: str) -> str:
    if "folders/" in url: return url.split("folders/")[1].split("?")[0]
    return None

def _extract_text_from_pdf(pdf_content: bytes) -> str:
    try:
        with fitz.open(stream=pdf_content, filetype="pdf") as doc: text = "".join(page.get_text() for page in doc)
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def search_google_drive(user_query: str, num_profiles_to_retrieve: int, folder_ids: list, user_id: str, token: dict) -> dict:
    if not token: return {"status": "error", "message": "Google Drive token not provided."}
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
                        gdrive_collection.add(ids=[item['id']], embeddings=[get_embedding(text)], documents=[text], metadatas=[{"user_id": user_id, "file_name": item['name']}])
        search_results = gdrive_collection.query(query_embeddings=[get_embedding(user_query)], n_results=num_profiles_to_retrieve, where={"user_id": user_id}, include=['documents'])
        if not search_results['documents'] or not search_results['documents'][0]:
            return {"status": "success", "analysis_data": {"overall_summary": "No relevant resumes were found in your Google Drive for this query.", "candidates": [], "overall_recommendation": "Try a different query or add more resumes to the selected folders."},"usage": {"input_tokens": 0, "output_tokens": 0}}
        context_for_llm = "\n\n---\n\n".join(search_results['documents'][0])
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
        print("--- Source 'Both' selected, defaulting to Lark's Database for MVP ---")
        return search_lark_database(user_query, num_profiles_to_retrieve)
    else:
        return {"status": "error", "message": f"Invalid source specified: {source}"}