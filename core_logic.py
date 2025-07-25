import json
import os
import uuid
import chromadb
# Import both OpenAI for embeddings and Anthropic for chat completions
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv
import random 

# Load environment variables (API keys)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Initialize OpenAI client for embeddings
client_openai = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Anthropic client for chat completions (Claude)
client_anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)

# Base directory for our single, mock database
DATABASE_DIR = "./mock_resume_database"
os.makedirs(DATABASE_DIR, exist_ok=True) # Create this folder if it doesn't already exist

# Initialize ChromaDB client pointing to our single database directory
chroma_client = chromadb.PersistentClient(path=os.path.join(DATABASE_DIR, "chroma_db"))
COLLECTION_NAME = "all_resumes" 

# --- Helper Function: Get Embedding (Uses OpenAI) ---
def get_embedding(text, model="text-embedding-3-small"):
    """
    Generates a numerical 'embedding' (a list of numbers) for a given piece of text.
    Embeddings are crucial because they allow us to measure the 'semantic similarity'
    between pieces of text (like a job query and a resume).
    We use OpenAI's 'text-embedding-3-small' model for this.
    """
    text = text.replace("\n", " ") 
    try:
        response = client_openai.embeddings.create(input=[text], model=model)
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding for text (first 50 chars: '{text[:50]}...'). Error: {e}")
        return None

# --- Database Initialization Function ---
def initialize_database(num_resumes=100):
    """
    This function creates our mock resume database.
    It generates 'num_resumes' fake profiles, converts their text into embeddings (using OpenAI),
    and stores them in ChromaDB. It also saves the full resume text as JSON files.
    This only needs to be run ONCE when you start your app for the first time
    or if you want to regenerate the database.
    """
    print(f"Initializing mock database with {num_resumes} resumes...")
    
    try:
        collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    except Exception as e:
        print(f"Error getting/creating collection '{COLLECTION_NAME}': {e}.")
        print("Attempting to delete and recreate the collection. (Might happen if a previous run failed or schema changed.)")
        try:
            chroma_client.delete_collection(name=COLLECTION_NAME) 
            collection = chroma_client.create_collection(name=COLLECTION_NAME) 
        except Exception as inner_e:
            print(f"FATAL: Failed to delete and recreate collection: {inner_e}. Please check permissions or manually delete '{os.path.join(DATABASE_DIR, 'chroma_db')}' folder.")
            raise 

    resumes_data = generate_fake_resume_data(num_resumes)

    raw_resumes_dir = os.path.join(DATABASE_DIR, "raw_resumes")
    os.makedirs(raw_resumes_dir, exist_ok=True)

    docs_to_add = []
    metadatas_to_add = []
    ids_to_add = []
    embeddings_to_add = []

    print(f"Preparing to process {len(resumes_data)} resumes...")

    for resume in resumes_data:
        resume_id = resume.get("id", str(uuid.uuid4()))
        raw_text = resume.get("raw_text", "")
        
        if not raw_text:
            print(f"Warning: Skipping resume {resume_id} due to missing 'raw_text'.")
            continue

        embedding = get_embedding(raw_text)
        if embedding is None:
            print(f"Warning: Could not generate embedding for resume {resume_id}. Skipping.")
            continue

        resume_file_path = os.path.join(raw_resumes_dir, f"{resume_id}.json")
        with open(resume_file_path, "w") as f:
            json.dump(resume, f, indent=4)

        docs_to_add.append(raw_text)
        metadatas_to_add.append({
            "resume_id": resume_id,
            "name": resume.get("name", "N/A"),
            "job_title": resume.get("job_title", "N/A"),
            "level": resume.get("level", "N/A"),
            "industry": resume.get("industry", "N/A"),
            "skills": ", ".join(resume.get("skills", [])), 
        })
        ids_to_add.append(resume_id)
        embeddings_to_add.append(embedding)

    if ids_to_add: 
        collection.add(
            embeddings=embeddings_to_add,
            documents=docs_to_add,
            metadatas=metadatas_to_add,
            ids=ids_to_add
        )
        print(f"Successfully added {len(ids_to_add)} resumes to ChromaDB collection '{COLLECTION_NAME}'.")
    else:
        print(f"No valid resumes to add to the database.")

# --- Query Function ---
def query_database(query: str, num_results: int = 5):
    """
    This is the main search function. It takes a hiring manager's query,
    finds the most relevant resumes using embeddings (OpenAI), and then uses
    Claude to provide a detailed analysis and ranking.
    """
    if not query:
        return "Error: Query cannot be empty. Please enter what you are looking for."

    try:
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
        if collection.count() == 0:
            return "The resume database is empty. Please click 'Initialize Database' in the sidebar first."
    except Exception as e:
        return f"Error: Could not access the resume database. Please ensure it's initialized. (Details: {e})"

    query_embedding = get_embedding(query)
    if query_embedding is None:
        return "Error: Could not generate embedding for your query. Please check your internet connection or OpenAI API key."

    # Perform the semantic search in ChromaDB.
    # THIS IS THE CORRECTED LINE: Removed 'ids' from the include list.
    results = collection.query(
        query_embeddings=[query_embedding], 
        n_results=num_results,             
        include=['metadatas'] # <-- FIXED THIS LINE
    )

    candidates_for_llm = [] 
    raw_resumes_dir = os.path.join(DATABASE_DIR, "raw_resumes")

    if results and results['ids'] and results['ids'][0]: 
        print(f"Found {len(results['ids'][0])} potential candidates via semantic search. Preparing for Claude analysis...")
        for i, resume_id in enumerate(results['ids'][0]):
            metadata = results['metadatas'][0][i] 
            
            resume_file_path = os.path.join(raw_resumes_dir, f"{resume_id}.json")
            try:
                with open(resume_file_path, "r") as f:
                    full_resume_data = json.load(f)
                    raw_text = full_resume_data.get("raw_text", "No raw text available for detailed review.")
            except FileNotFoundError:
                raw_text = "Raw resume file not found on disk."
                print(f"Warning: Raw resume file not found for {resume_id}. Check if '{resume_file_path}' exists.")
            except json.JSONDecodeError:
                raw_text = "Could not parse raw resume JSON file. File might be corrupted."
                print(f"Warning: Could not parse raw resume JSON for {resume_id}.")
            
            candidates_for_llm.append({
                "id": resume_id,
                "name": metadata.get("name", "N/A"),
                "job_title": metadata.get("job_title", "N/A"),
                "level": metadata.get("level", "N/A"),
                "industry": metadata.get("industry", "N/A"),
                "skills": metadata.get("skills", "N/A"),
                "raw_resume_text": raw_text 
            })
    else:
        return "No candidates found matching the initial semantic search criteria. Try a different query or initialize the database."

    return get_llm_summary_and_ranking(query, candidates_for_llm)

# --- LLM Analysis Function (Uses Claude) ---
def get_llm_summary_and_ranking(query, candidates):
    """
    Uses Claude (Anthropic's model) to analyze candidates against the query
    and provide a summarized, ranked list with reasoning.
    """
    if not candidates:
        return "No candidates provided for LLM analysis."

    system_message = "You are an expert HR recruitment assistant. Your task is to evaluate candidate profiles against a hiring manager's query and provide insightful analysis and recommendations. Focus on core skills, experience, and leadership where relevant. Be concise but thorough, and always suggest top candidates with reasons."

    prompt_content = [
        f"The hiring manager is looking for candidates with the following requirements: '{query}'\n",
        "Here are some candidate profiles from our proprietary database. Please evaluate how well each candidate matches the query, explain your reasoning clearly, and provide a confidence score (0-100) for each based on the provided resume text. Finally, recommend the top 3 best-fit candidates and briefly explain why they stand out.\n",
        "--- Candidates for Evaluation ---\n"
    ]

    for i, candidate in enumerate(candidates):
        limited_raw_text = candidate['raw_resume_text'][:1500]
        if len(candidate['raw_resume_text']) > 1500:
            limited_raw_text += "...\n(Full resume text truncated for brevity)"
        
        prompt_content.append(f"Candidate {i+1}: Name: {candidate['name']}, Job Title: {candidate['job_title']}, Level: {candidate['level']}, Industry: {candidate['industry']}, Skills: {candidate['skills']}\n")
        prompt_content.append(f"Detailed Resume Snippet for Candidate {i+1}:\n{limited_raw_text}\n")
        prompt_content.append("---\n") 
    
    prompt_content.append("\nYour Evaluation and Top Recommendations (format clearly, use bullet points for justifications):")

    try:
        response = client_anthropic.messages.create(
            model="claude-3-haiku-20240307", 
            max_tokens=1500, 
            temperature=0.5, 
            system=system_message, 
            messages=[
                {"role": "user", "content": "".join(prompt_content)} 
            ]
        )
        return response.content[0].text 
    except Exception as e:
        return f"Error communicating with Claude: {e}. Please check your Anthropic API key and internet connection."

# --- Helper: Generate Fake Resume Data (for database initialization) ---
def generate_fake_resume_data(num_resumes=10):
    """
    Generates a list of fake resume dictionaries. This helps us quickly create
    a test database without needing real resume files.
    Each dictionary represents one candidate profile.
    """
    resumes = []
    skills_list = ["Python", "Java", "SQL", "AWS", "Azure", "GCP", "Machine Learning", "Data Analysis", "Project Management", "Marketing Strategy", "Sales Leadership", "Financial Modeling", "HR Management", "Product Management", "UI/UX Design", "Backend Development", "Frontend Development", "DevOps", "Cybersecurity", "Blockchain", "Salesforce CRM", "SAP ERP"]
    roles_list = ["Software Engineer", "Data Scientist", "Product Manager", "Marketing Manager", "Sales Executive", "HR Business Partner", "Financial Analyst", "UX Designer", "DevOps Engineer", "Business Analyst", "Technical Lead", "Director of Engineering", "VP of Sales"]
    industries_list = ["Tech", "Finance", "Healthcare", "Retail", "SaaS", "Biotech", "Manufacturing", "E-commerce", "Consulting", "Automotive"]
    levels_list = ["Junior", "Mid", "Senior", "Lead", "Manager", "Director", "VP"]
    
    for i in range(num_resumes):
        name = f"Candidate {i+1} from GlobalSolutions Inc."
        job_title = random.choice(roles_list)
        industry = random.choice(industries_list)
        level = random.choice(levels_list)
        num_skills = random.randint(3, 8)
        candidate_skills = random.sample(skills_list, num_skills)
        experience_years = random.randint(2, 15)
        
        experience_summary = (
            f"{experience_years} years of experience as a {job_title} at a {level} level in the {industry} sector. "
            f"Highly proficient in {', '.join(candidate_skills)}. "
            f"Proven track record in {'driving revenue growth' if 'Sales' in job_title else 'building scalable systems' if 'Engineer' in job_title else 'leading cross-functional teams' if 'Manager' in job_title else 'analyzing complex data'}. "
            f"Strong problem-solving abilities and a collaborative mindset."
        )
        
        full_raw_text = (
            f"Name: {name}\n"
            f"Email: {name.lower().replace(' ', '.').replace('from.globalsolutions.inc.', '').replace('..', '.')}@example.com | Phone: (123) 555-{i:04d}\n\n"
            f"**Summary:** A results-oriented and experienced {level} {job_title} with {experience_years} years in the {industry} industry. "
            f"Skilled in {', '.join(candidate_skills)}. Adept at {experience_summary.split('Proven track record in ')[-1].lower()}.\n\n"
            f"**Experience:**\n"
            f"**Acme Corp** - {job_title} ({random.randint(1, experience_years-1)} years)\n"
            f"  - Led {random.randint(1, 3)} major projects, improving efficiency by {random.randint(10, 40)}%.\n"
            f"  - Mentored junior team members and fostered a collaborative environment.\n"
            f"  - Developed and deployed X, Y, and Z features using {random.choice(candidate_skills)}.\n"
            f"**Global Innovations** - {random.choice(['Junior', 'Associate'])} {job_title.split(' ')[-1]} ({random.randint(1, 3)} years)\n"
            f"  - Contributed to {random.randint(2, 5)} product releases.\n"
            f"  - Performed data analysis using {random.choice(candidate_skills)}.\n\n"
            f"**Education:** Bachelor's Degree in {random.choice(['Computer Science', 'Business Administration', 'Marketing', 'Finance', 'Engineering'])} from a well-regarded university."
        )

        resumes.append({
            "id": str(uuid.uuid4()),
            "name": name,
            "job_title": job_title,
            "industry": industry,
            "level": level,
            "skills": candidate_skills,
            "raw_text": full_raw_text
        })
    return resumes