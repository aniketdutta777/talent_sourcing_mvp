import json
import os
import uuid
import chromadb
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv
import random 

# --- Load environment variables (API keys) and DEBUG prints ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
# This key is for your own API authentication. REMOVED DEFAULT VALUE for stricter checking.
YOUR_MOCK_API_KEY = os.getenv("YOUR_MOCK_API_KEY") 

# --- DEBUG PRINT: VERY IMPORTANT TO SEE THIS OUTPUT ---
if not YOUR_MOCK_API_KEY:
    print("CRITICAL DEBUG (core_logic.py): YOUR_MOCK_API_KEY is EMPTY or not loaded! Check .env file.")
else:
    print(f"DEBUG (core_logic.py): YOUR_MOCK_API_KEY loaded: '{YOUR_MOCK_API_KEY}'")
# -----------------------------------------------------


# Initialize OpenAI client for embeddings
client_openai = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Anthropic client for chat completions (Claude)
client_anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)

# Base directory for our single, mock database
DATABASE_DIR = "./mock_resume_database"
os.makedirs(DATABASE_DIR, exist_ok=True) 

# Initialize ChromaDB client pointing to our single database directory
chroma_client = chromadb.PersistentClient(path=os.path.join(DATABASE_DIR, "chroma_db"))
COLLECTION_NAME = "all_resumes" 

# --- Helper Function: Get Embedding (Uses OpenAI) ---
def get_embedding(text, model="text-embedding-3-small"):
    """
    Generates a numerical 'embedding' (a list of numbers) for a given piece of text.
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
            "name": metadata.get("name", "N/A"),
            "job_title": metadata.get("job_title", "N/A"),
            "level": metadata.get("level", "N/A"),
            "industry": metadata.get("industry", "N/A"),
            "skills": ", ".join(metadata.get("skills", [])), 
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

# --- Tool Function: resume_search_tool ---
# This function performs the actual search and returns raw candidate data.
# Claude will call this function.
def resume_search_tool(query: str, num_results: int = 5) -> list[dict]:
    """
    Searches the resume database for candidates matching the given query.
    Returns a list of candidate dictionaries with their raw resume text.
    This function is designed to be called by an AI model as a tool.
    """
    print(f"\n--- Tool Call: resume_search_tool(query='{query}', num_results={num_results}) ---")
    if not query:
        print("Tool Error: Query cannot be empty.")
        return [{"error": "Query cannot be empty for resume search."}]

    try:
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
        if collection.count() == 0:
            print("Tool Error: The resume database is empty.")
            return [{"error": "Resume database is empty. Initialize it first."}]
    except Exception as e:
        print(f"Tool Error: Could not access the resume database. Details: {e}")
        return [{"error": f"Database access error: {e}"}]

    query_embedding = get_embedding(query)
    if query_embedding is None:
        print("Tool Error: Could not generate embedding for query.")
        return [{"error": "Failed to generate embedding for query."}]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=num_results,
        include=['metadatas'] # Only need metadatas; IDs are always returned
    )

    candidates_data = []
    raw_resumes_dir = os.path.join(DATABASE_DIR, "raw_resumes")

    if results and results['ids'] and results['ids'][0]:
        print(f"Tool: Found {len(results['ids'][0])} potential candidates via semantic search.")
        for i, resume_id in enumerate(results['ids'][0]):
            metadata = results['metadatas'][0][i]
            
            resume_file_path = os.path.join(raw_resumes_dir, f"{resume_id}.json")
            try:
                with open(resume_file_path, "r") as f:
                    full_resume_data = json.load(f)
                    raw_text = full_resume_data.get("raw_text", "No raw text available.") 
            except FileNotFoundError:
                raw_text = "Raw resume file not found on disk."
                print(f"Warning: Raw resume file not found for {resume_id}")
            except json.JSONDecodeError:
                raw_text = "Could not parse raw resume JSON file."
                print(f"Warning: Could not parse raw resume JSON for {resume_id}.")
            
            candidates_data.append({
                "id": resume_id,
                "name": metadata.get("name", "N/A"),
                "job_title": metadata.get("job_title", "N/A"),
                "level": metadata.get("level", "N/A"),
                "industry": metadata.get("industry", "N/A"),
                "skills": metadata.get("skills", "N/A"),
                "raw_resume_text": raw_text # This is the crucial part sent to the LLM
            })
    else:
        print("Tool: No candidates found for the query.")
        return [{"message": "No candidates found matching the search criteria."}]
    
    return candidates_data

# --- Tool Schema Definition ---
# This JSON structure tells Claude about our 'resume_search_tool'
resume_search_tool_schema = {
    "name": "resume_search_tool",
    "description": "Searches a proprietary database of candidate resumes to find profiles matching a specific job query. This tool is useful for finding relevant candidates based on skills, experience, and role requirements.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The natural language job description or requirements for the candidate being sought."
            },
            "num_results": {
                "type": "integer",
                "description": "The maximum number of top relevant resume profiles to retrieve from the database. Defaults to 5. Max 15.",
                "default": 5
            }
        },
        "required": ["query"]
    }
}

# --- Claude Tool Use Orchestration Function ---
# This function now returns a dictionary ready to be converted to JSON.
def perform_claude_search_with_tool(user_query: str, num_profiles_to_retrieve: int = 7) -> dict:
    """
    Orchestrates the interaction with Claude using its tool-use capabilities.
    Claude will decide when to call the resume_search_tool.
    Returns a structured dictionary containing the analysis.
    """
    print(f"\n--- Orchestrating Claude Search for: '{user_query}' ---")

    messages = [
        {
            "role": "user",
            "content": user_query
        }
    ]

    system_message = """You are an expert HR recruitment assistant. You have access to a `resume_search_tool` to find candidates. 
    Use this tool when the user asks to find candidates or describes job requirements.

    After successfully using the `resume_search_tool`, analyze the results carefully. Provide a clear, concise summary of the top candidates, their fit, and confidence scores. 
    Always recommend the top 3 best-fit candidates with reasons.

    YOUR FINAL OUTPUT MUST BE VALID JSON. Structure your response as follows:
    ```json
    {
      "overall_summary": "Overall summary of the search results.",
      "candidates": [
        {
          "id": "candidate_id_from_tool_output",
          "name": "Candidate Name from GlobalSolutions Inc.",
          "confidence_score": "X/5" or "Y%",
          "job_title": "Candidate Job Title", # Added job_title for clarity in structured output
          "justification": "Detailed reason why this candidate is a good/bad fit, linking their profile details to the query. Refer to their skills, experience, and industry from the tool output."
        },
        // ... more candidates up to the number provided by the tool output
      ],
      "overall_recommendation": "Overall recommendation or final thoughts."
    }
    ```
    Ensure the JSON is well-formed, including commas and correct curly braces. Do not include markdown outside the ```json block unless asked for.
    """

    # First call to Claude: Provide the user's query and the available tool
    try:
        response = client_anthropic.messages.create(
            model="claude-3-haiku-20240307", # Using the Haiku model that worked
            max_tokens=2000, # Allow more tokens for Claude's reasoning
            temperature=0.0, # Set temperature low for more deterministic tool use
            tools=[resume_search_tool_schema], # Provide the tool schema
            messages=messages,
            system=system_message # Pass the system message here
        )
    except Exception as e:
        print(f"Error initiating conversation with Claude: {e}. Check your Anthropic API key.")
        return {"status": "error", "message": f"Error initiating conversation with Claude: {e}"}

    # Process Claude's first response
    if response.stop_reason == "tool_use":
        tool_use = response.content[0]
        tool_name = tool_use.name
        tool_input = tool_use.input

        print(f"Claude requested to use tool: {tool_name} with input: {tool_input}")

        # Validate and execute the tool
        if tool_name == "resume_search_tool":
            requested_num_results = min(tool_input.get("num_results", 5), num_profiles_to_retrieve)
            
            tool_output = resume_search_tool(
                query=tool_input.get("query"),
                num_results=requested_num_results
            )
            print(f"Tool execution complete. Results: {len(tool_output)} candidates found.")
            
            # Send the tool's output back to Claude
            messages.append({"role": "assistant", "content": [tool_use]}) 
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": json.dumps(tool_output) # Send tool output as JSON string
                    },
                    # ADD THIS NEW MESSAGE TO INSTRUCT CLAUDE FOR JSON OUTPUT
                    {
                        "type": "text",
                        "text": """
                        Based on the tool results and the initial query, provide your analysis and top recommendations.
                        Format your entire response as a JSON object with the structure defined in the system prompt.
                        Ensure the JSON is well-formed and valid. Do not include any text outside the JSON block.
                        """
                    }
                ]
            })

            # Second call to Claude: Now with the tool results
            try:
                final_response = client_anthropic.messages.create(
                    model="claude-3-haiku-20240307", # Use the same model
                    max_tokens=1500, # Increased max_tokens as JSON can be verbose
                    temperature=0.5, 
                    messages=messages,
                    system=system_message # Pass the system message again
                )
                
                # Attempt to parse Claude's response as JSON
                try:
                    # Claude sometimes wraps JSON in markdown, so extract it
                    json_string = final_response.content[0].text
                    if json_string.startswith("```json") and json_string.endswith("```"):
                        json_string = json_string[len("```json"): -len("```")].strip()
                    
                    parsed_json = json.loads(json_string)
                    return {"status": "success", "analysis_data": parsed_json}

                except json.JSONDecodeError as json_e:
                    print(f"Error parsing Claude's JSON response: {json_e}")
                    print(f"Claude's raw response was: {final_response.content[0].text}")
                    return {"status": "error", "message": "LLM response was not valid JSON.", "raw_llm_output": final_response.content[0].text}

            except Exception as e:
                print(f"Error getting final response from Claude after tool use: {e}")
                return {"status": "error", "message": f"Error getting final response from Claude after tool use: {e}"}
        else:
            return {"status": "error", "message": f"Claude requested an unknown tool: {tool_name}"}
    elif response.stop_reason == "end_turn":
        # Claude decided it didn't need a tool (e.g., simple greeting)
        print("Claude finished its turn without tool use.")
        return {"status": "success", "message": response.content[0].text}
    else:
        print(f"Unexpected Claude response stop reason: {response.stop_reason}. Content: {response.content[0].text if response.content else 'No content'}")
        return {"status": "error", "message": "Unexpected response from Claude."}


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
        
        # --- FIX APPLIED HERE: Simplified email generation to avoid f-string complexity ---
        email_local_part = name.lower().replace(' ', '.')
        email_local_part = email_local_part.replace('..', '.') # Clean up potential double dots if name has multiple spaces

        full_raw_text = (
            f"Name: {name}\n"
            f"Email: {email_local_part}@example.com | Phone: (123) 555-{i:04d}\n\n" # SIMPLIFIED LINE
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
