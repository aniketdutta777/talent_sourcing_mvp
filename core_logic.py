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
    """Generates a list of realistic mock resumes with diverse attributes."""
    resumes = []
    skills_list = ["Python", "Java", "SQL", "AWS", "GCP", "Machine Learning", "Data Analysis", "Project Management", "Marketing Strategy", "Sales Leadership", "Product Management", "UI/UX Design", "DevOps"]
    roles_list = ["Software Engineer", "Data Scientist", "Product Manager", "Marketing Manager", "Sales Executive", "DevOps Engineer", "Technical Lead"]
    industries_list = ["Tech", "Finance", "Healthcare", "SaaS", "E-commerce", "Consulting"]
    levels_list = ["Junior", "Mid", "Senior", "Lead", "Manager"]
    companies_list = ["Innovate Inc.", "DataDriven Corp.", "CloudSphere LLC", "QuantumLeap Solutions", "Synergy Systems", "NextGen Tech"]
    universities_list = ["State University", "Tech Institute", "City College", "Northern University"]

    for i in range(num_resumes):
        resume_id = str(uuid.uuid4())
        job_title = random.choice(roles_list)
        industry = random.choice(industries_list)
        level = random.choice(levels_list)
        candidate_skills = random.sample(skills_list, random.randint(3, 7))
        experience_years = random.randint(2, 12)
        name = f"Candidate {i+1}"
        email = f"{name.lower().replace(' ', '.')}{random.randint(10,99)}@example.com"
        phone = f"(123) 555-{i:04d}"
        pdf_url = f"https://example.com/resumes/{resume_id}.pdf"

        full_raw_text = f"Name: {name}\nEmail: {email} | Phone: {phone}\n\nSummary: A results-oriented {level} {job_title} with {experience_years} years in the {industry} industry. Skilled in {', '.join(candidate_skills)}."

        resumes.append({
            "id": resume_id, "name": name, "email": email, "phone": phone, "pdf_url": pdf_url,
            "job_title": job_title, "industry": industry, "level": level,
            "skills": candidate_skills, "raw_text": full_raw_text
        })
    return resumes

def initialize_database(num_resumes=100):
    """Populates the ChromaDB database with mock resumes if it's empty."""
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    if collection.count() > 0:
        print(f"Database already initialized with {collection.count()} resumes.")
        return

    resumes_data = generate_fake_resume_data(num_resumes)
    for resume in resumes_data:
        collection.add(
            embeddings=[get_embedding(resume["raw_text"])],
            documents=[resume["raw_text"]],
            metadatas=[{
                "resume_id": resume["id"], "name": resume["name"], "email": resume["email"],
                "phone": resume["phone"], "pdf_url": resume["pdf_url"], "job_title": resume["job_title"],
                "level": resume["level"], "industry": resume["industry"], "skills": ", ".join(resume["skills"])
            }],
            ids=[resume["id"]]
        )
    print(f"Successfully added {len(resumes_data)} resumes to ChromaDB.")

# --- CORRECTED SEARCH TOOL ---
def resume_search_tool(query: str, num_results: int = 5, level: str = None, industry: str = None) -> list[dict]:
    """
    Searches the resume database for candidates matching the given query,
    with optional filtering by level and industry using the correct ChromaDB syntax.
    """
    print(f"\n--- Tool Call: resume_search_tool(query='{query}', num_results={num_results}, level='{level}', industry='{industry}') ---")
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    embedding = get_embedding(query)

    # Build the metadata filter (where clause) for ChromaDB
    filter_conditions = []
    if level:
        filter_conditions.append({"level": {"$eq": level}})
    if industry:
        filter_conditions.append({"industry": {"$eq": industry}})

    where_filter = None
    if len(filter_conditions) > 1:
        where_filter = {"$and": filter_conditions}
    elif len(filter_conditions) == 1:
        where_filter = filter_conditions[0]

    print(f"ChromaDB Query Filter: {where_filter}")

    results = collection.query(
        query_embeddings=[embedding],
        n_results=num_results,
        where=where_filter,
        include=['metadatas', 'documents']
    )

    candidates_data = []
    if results and results['ids'] and results['ids'][0]:
        print(f"Tool: Found {len(results['ids'][0])} potential candidates matching filters.")
        for i, metadata in enumerate(results['metadatas'][0]):
            # This now returns all the rich data Claude needs
            candidates_data.append({
                "name": metadata.get("name"),
                "contact_information": {
                    "email": metadata.get("email"),
                    "phone": metadata.get("phone")
                },
                "resume_pdf_url": metadata.get("pdf_url"),
                "raw_resume_text": results['documents'][0][i]
            })
    else:
        print("Tool: No candidates found for the query with the specified filters.")
        return [{"message": "No candidates found matching the search criteria and filters."}]

    return candidates_data

# --- CORRECTED TOOL SCHEMA ---
resume_search_tool_schema = {
    "name": "resume_search_tool",
    "description": "Searches a resume database to find profiles matching a job query. Supports filtering by experience level and industry.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The job description or requirements."},
            "num_results": {"type": "integer", "description": "Number of profiles to retrieve.", "default": 5},
            "level": {"type": "string", "description": "Optional filter for experience level (e.g., 'Senior', 'Junior')."},
            "industry": {"type": "string", "description": "Optional filter for industry (e.g., 'Finance', 'SaaS')."}
        },
        "required": ["query"]
    }
}

def _perform_claude_search_with_tool_internal(user_query: str, num_profiles_to_retrieve: int = 7) -> dict:
    """Internal helper for performing the complete Claude search and analysis workflow."""
    # --- CORRECTED SYSTEM MESSAGE ---
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
        response = global_client_anthropic.messages.create(
            model="claude-3-haiku-20240307", max_tokens=2000, temperature=0.0,
            tools=[resume_search_tool_schema], messages=messages, system=system_message
        )
    except Exception as e:
        return {"status": "error", "message": f"Error initiating conversation with Claude: {e}"}

    if response.stop_reason == "tool_use":
        tool_use = next((block for block in response.content if block.type == "tool_use"), None)
        if not tool_use:
            return {"status": "error", "message": "Claude indicated tool use, but no tool was specified."}

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

def perform_claude_search_with_tool(user_query: str, num_profiles_to_retrieve: int = 7) -> dict:
    """Wrapper function that exposes the internal search orchestration."""
    if global_client_anthropic is None:
        raise RuntimeError("Anthropic client not initialized.")
    return _perform_claude_search_with_tool_internal(user_query, num_profiles_to_retrieve)