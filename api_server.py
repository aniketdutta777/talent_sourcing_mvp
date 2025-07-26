from fastapi import FastAPI, HTTPException, Depends, Security 
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials 
from pydantic import BaseModel, Field 
import uvicorn
import json 
import os 
# from dotenv import load_dotenv # <--- ENSURE THIS LINE IS COMMENTED OR DELETED (not needed when deployed)

# Import core_logic as a module to correctly access its global variables and functions
import core_logic # <--- CRUCIAL CHANGE: Import core_logic as a module

# A simple in-memory "database" of valid API keys.
# In a real production system, this would eventually be a database table.
VALID_API_KEYS = {
    "a885598e-1a1b-463b-9c23-acc4e4275330",  # e.g., "a885598e-1a1b-463b-..."
    "05f84d16-3908-4da3-8804-5c455d4acc1c"
}

# Initialize FastAPI app
app = FastAPI(
    title="Talent Search MCP API (Mock)",
    description="A mock API exposing LLM-powered talent search over a proprietary resume database. This simulates your productized database as an MCP server.",
    version="1.0.0"
)

# --- Authentication Setup ---
security_scheme = HTTPBearer()

# Dependency function to check the API key
def get_api_key(credentials: HTTPAuthorizationCredentials = Depends(security_scheme)):
    """
    Checks if the provided API key is valid and returns it.
    """
    if credentials.credentials in VALID_API_KEYS:
        return credentials.credentials  # Return the key on success
    raise HTTPException(
        status_code=401,
        detail="Unauthorized: Invalid or missing API Key."
    )
# Pydantic models for structured responses (unchanged)

# In api_server.py

class ContactInfo(BaseModel):
    email: str = Field(..., description="Candidate's email address.")
    phone: str = Field(..., description="Candidate's phone number.")

class Candidate(BaseModel):
    name: str = Field(..., description="Name of the candidate.")
    contact_information: ContactInfo = Field(..., description="Candidate's contact details.")
    summary: str = Field(..., description="A concise summary of the candidate's fit for the role.")
    resume_pdf_url: str = Field(..., description="A direct link to the candidate's resume PDF.")
    
class AnalysisResponse(BaseModel):
    overall_summary: str = Field(..., description="Overall summary of the candidate evaluation.")
    candidates: list[Candidate] = Field(..., description="List of analyzed candidates with their details.")
    overall_recommendation: str = Field(..., description="Overall recommendation for top candidates.")

class SearchResponseWrapper(BaseModel):
    status: str = Field(..., description="Status of the API request (e.g., 'success', 'error').")
    analysis_data: AnalysisResponse = Field(..., description="Structured analysis result from the LLM.")

# Pydantic model for incoming search requests (unchanged)
class SearchRequest(BaseModel):
    query: str
    num_results: int = Field(7, description="Number of top candidates to retrieve and analyze. Default is 7.")

@app.on_event("startup")
async def startup_event():
    """
    Event handler that runs automatically when the FastAPI application starts.
    Initializes API clients (loading API keys) and checks/initializes the database.
    """
    print("\n--- FastAPI Startup: Initializing API Clients & Checking Database ---")
    try:
        # CRUCIAL CALL: This function (from core_logic.py) will now load API keys and initialize clients
        core_logic.initialize_api_clients() # <--- CALL WITH core_logic. PREFIX
        print("DEBUG: API clients initialized.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to initialize API clients: {e}")
        raise HTTPException(status_code=500, detail=f"API Client Initialization Failed: {e}") 

    # Rest of the database initialization check 
    try:
        # Use core_logic.chroma_client and core_logic.COLLECTION_NAME
        collection = core_logic.chroma_client.get_collection(name=core_logic.COLLECTION_NAME) 
        if collection.count() == 0:
            print("Database empty. Initializing with 100 mock resumes...")
            core_logic.initialize_database(100) # <--- CALL WITH core_logic. PREFIX
            print("Database initialization complete.")
        else:
            print(f"Database already initialized with {collection.count()} resumes.")
    except Exception as e:
        print(f"Error during database startup check/initialization: {e}")

@app.post("/v1/search_candidates",
          summary="Search for candidates using LLM intelligence",
          response_model=SearchResponseWrapper)
async def search_candidates(request: SearchRequest, api_key: str = Depends(get_api_key)):
    """
    Receives a natural language query for candidates and returns LLM-analyzed results.
    This endpoint now logs token usage for each successful request.
    """
    print(f"\n--- API Call: /v1/search_candidates (Key ending with '...{api_key[-4:]}') ---")
    print(f"Received query: '{request.query}' for {request.num_results} results.")

    # Ensure database is accessible/initialized before performing search
    try:
        collection = core_logic.chroma_client.get_collection(name=core_logic.COLLECTION_NAME)
        if collection.count() == 0:
            raise HTTPException(status_code=500, detail="Database not initialized.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database access error: {e}")

    try:
        result_data = core_logic.perform_claude_search_with_tool(
            user_query=request.query,
            num_profiles_to_retrieve=request.num_results
        )

        if result_data["status"] == "success" and "analysis_data" in result_data:
            # --- COST TRACKING LOG ---
            usage = result_data.get("usage", {"input_tokens": 0, "output_tokens": 0})
            print(f"COST_LOG: key='{api_key}' usage={usage}")
            # -------------------------

            return SearchResponseWrapper(
                status=result_data["status"],
                analysis_data=result_data["analysis_data"]
            )
        else:
            raise HTTPException(status_code=500, detail=f"LLM analysis failed: {result_data.get('message', 'Unknown error')}")

    except Exception as e:
        print(f"Error during LLM analysis in API: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error during LLM analysis: {e}")

# THE DUPLICATED search_candidates FUNCTION BLOCK HAS BEEN DELETED FROM HERE

@app.get("/", summary="API Root / Health Check")
async def read_root():
    return {"message": "Talent Search MCP API is running. Go to /docs for API documentation."}