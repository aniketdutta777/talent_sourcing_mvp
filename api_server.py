from fastapi import FastAPI, HTTPException, Depends, Security 
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials 
from pydantic import BaseModel, Field 
import uvicorn
import json 
import os 
# from dotenv import load_dotenv # <--- ENSURE THIS LINE IS COMMENTED OR DELETED (not needed when deployed)

# Import core_logic as a module to correctly access its global variables and functions
import core_logic # <--- CRUCIAL CHANGE: Import core_logic as a module

# Initialize FastAPI app
app = FastAPI(
    title="Talent Search MCP API (Mock)",
    description="A mock API exposing LLM-powered talent search over a proprietary resume database. This simulates your productized database as an MCP server.",
    version="1.0.0"
)

# --- Authentication Setup ---
security_scheme = HTTPBearer()

# Dependency function to check the API key
def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security_scheme)):
    # print(f"DEBUG AUTH: Received credential: '{credentials.credentials}'") 
    # print(f"DEBUG AUTH: Expected key: '{core_logic.global_mock_api_key}'") # Use core_logic.global_mock_api_key here

    # Check if the global_mock_api_key was loaded successfully by initialize_api_clients
    if core_logic.global_mock_api_key is None: # <--- Use core_logic.global_mock_api_key
        raise HTTPException(status_code=500, detail="Server Error: API Key not initialized. Please ensure the API server started correctly and clients initialized.")

    if credentials.credentials == core_logic.global_mock_api_key: # <--- Use core_logic.global_mock_api_key
        return True 
    raise HTTPException(
        status_code=401, 
        detail="Unauthorized: Invalid API Key. Please provide a valid 'Bearer YOUR_API_KEY' in the Authorization header."
    )

# Pydantic models for structured responses (unchanged)
class Candidate(BaseModel):
    id: str = Field(..., description="Unique ID of the candidate.")
    name: str = Field(..., description="Name of the candidate.")
    confidence_score: str = Field(..., description="Confidence score for the candidate's fit (e.g., 'X/5' or 'Y%').")
    job_title: str = Field(..., description="Job title of the candidate.") 
    justification: str = Field(..., description="Detailed justification for why this candidate is a good fit.")

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
          response_model=SearchResponseWrapper, 
          dependencies=[Depends(verify_api_key)]) 
async def search_candidates(request: SearchRequest):
    """
    Receives a natural language query for candidates and returns LLM-analyzed results.
    This endpoint serves as your "MCP" interface to your talent database.
    It takes a client's query (potentially from their LLM context) and returns
    candidates analyzed by your Claude instance.
    """
    print(f"\n--- API Call: /v1/search_candidates ---")
    print(f"Received query: '{request.query}' for {request.num_results} results.")

    # Ensure database is accessible/initialized before performing search
    try:
        # Use core_logic.chroma_client and core_logic.COLLECTION_NAME
        collection = core_logic.chroma_client.get_collection(name=core_logic.COLLECTION_NAME) 
        if collection.count() == 0:
            raise HTTPException(status_code=500, detail="Database not initialized. Please ensure the API server started correctly.")
        # Verify global clients are still initialized (redundant with app.state, but adds safety)
        if core_logic.global_client_openai is None or core_logic.global_client_anthropic is None:
            raise HTTPException(status_code=500, detail="LLM clients not available post-startup. Check server logs.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database access error: {e}. Please ensure the API server started correctly.")

    try:
        # Call perform_claude_search_with_tool with core_logic. prefix
        result_data = core_logic.perform_claude_search_with_tool( 
            user_query=request.query, 
            num_profiles_to_retrieve=request.num_results
        )

        if result_data["status"] == "success" and "analysis_data" in result_data:
            return SearchResponseWrapper(
                status=result_data["status"],
                analysis_data=result_data["analysis_data"]
            )
        else:
            raise HTTPException(status_code=500, detail=f"LLM analysis failed: {result_data.get('message', 'Unknown error')}. Raw output: {result_data.get('raw_llm_output', 'N/A')}")

    except Exception as e:
        print(f"Error during LLM analysis in API: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error during LLM analysis: {e}")

@app.get("/", summary="API Root / Health Check")
async def read_root():
    return {"message": "Talent Search MCP API is running. Go to /docs for API documentation."}