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
chroma_client = None

# --- INITIALIZATION FUNCTIONS ---
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
    resumes = []
    skills_list = ["Python", "Java", "SQL", "AWS", "GCP", "Machine Learning", "Data Analysis", "Project Management", "Marketing Strategy", "Sales Leadership", "Product Management", "UI/UX Design", "DevOps"]
    roles_list = ["Software Engineer", "Data Scientist", "Product Manager", "Marketing Manager", "Sales Executive", "DevOps Engineer", "Technical Lead"]
    industries_list = ["Tech", "Finance", "Healthcare", "SaaS", "E-commerce", "Consulting"]
    levels_list = ["Junior", "Mid", "Senior", "Lead", "Manager"]
    companies_list = ["Innovate Inc.", "DataDriven Corp.", "CloudSphere LLC", "QuantumLeap Solutions", "Synergy Systems", "NextGen Tech"]
    for i in range(num_resumes):
        resume_id = str(uuid.uuid4())
        name = f"Candidate {i+1}"
        email = f"{name.lower().replace(' ', '.')}{random.randint(10,99)}@example.com"
        phone = f"(123) 555-{i:04d}"
        pdf_url = f"https://example.com/resumes/{resume_id}.pdf"
        job_title = random.choice(roles_list)
        industry = random.choice(industries_list)
        level = random.choice(levels_list)
        candidate_skills = random.sample(skills_list, random.randint(3, 7))
        experience_years = random.randint(2, 12)
        full_raw_text = f"Name: {name}\nEmail: {email} | Phone: {phone}\n\nSummary: A results-oriented {level} {job_title} with {experience_years} years in the {industry} industry. Skilled in {', '.join(candidate_skills)}."
        resumes.append({
            "id": resume_id, "name": name, "email": email, "phone": phone, "pdf_url": pdf_url,
            "job_title": job_title, "industry": industry, "level": level,
            "skills": candidate_skills, "raw_text": full_raw_text
        })
    return resumes

def initialize_database(num_resumes=100):
    if chroma_client is None: raise RuntimeError("Chroma client not initialized.")
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    if collection.count() > 0:
        print(f"Database already initialized with {collection.count()} resumes.")
        return
    resumes_data = generate_fake_resume_data(num_resumes)
    for resume in resumes_data:
        collection.add(
            embeddings=[get_embedding(resume["raw_text"])],
            documents=[resume["raw_text"]],
            metadatas=[{"resume_id": resume["id"], "name": resume["name"], "email": resume["email"], "phone": resume["phone"], "pdf_url": resume["pdf_url"], "job_title": resume["job_title"], "level": resume["level"], "industry": resume["industry"], "skills": ", ".join(resume["skills"])}],
            ids=[resume["id"]]
        )
    print(f"Successfully added {len(resumes_data)} resumes to ChromaDB.")

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
        for i, metadata in enumerate(results['metadatas'][0]):
            candidates_data.append({"name": metadata.get("name"), "contact_information": {"email": metadata.get("email"), "phone": metadata.get("phone")}, "resume_pdf_url": metadata.get("pdf_url"), "raw_resume_text": results['documents'][0][i]})
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