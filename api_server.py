from fastapi import FastAPI, HTTPException, Depends, Security 
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials 
from pydantic import BaseModel, Field 
import uvicorn
import json 
import os 
# from dotenv import load_dotenv # <--- ENSURE THIS LINE IS COMMENTED OR DELETED (not needed when deployed)

# Import the core logic functions and GLOBAL variables from core_logic.py
# IMPORTANT: initialize_api_clients MUST be called in @app.on_event("startup")
# global_mock_api_key will be available after initialize_api_clients() runs.
from core_logic import (
    perform_claude_search_with_tool, 
    initialize_database, 
    chroma_client, 
    COLLECTION_NAME, 
    initialize_api_clients, # <--- CRUCIAL: Import this function to initialize clients
    global_mock_api_key     # <--- CRUCIAL: This will hold the YOUR_MOCK_API_KEY after initialization
)

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
    # print(f"DEBUG AUTH: Received credential: '{credentials.credentials}'") # Can uncomment for debugging
    # print(f"DEBUG AUTH: Expected key: '{global_mock_api_key}'") # Can uncomment for debugging

    # Check if the global_mock_api_key was loaded successfully by initialize_api_clients
    if global_mock_api_key is None:
        # This error occurs if initialize_api_clients() failed or wasn't called.
        # This will now correctly raise an internal error if env var setup fails on Railway.
        raise HTTPException(status_code=500, detail="Server Error: API Key not initialized. Please ensure the API server started correctly and clients initialized.")

    if credentials.credentials == global_mock_api_key: # <--- Use the GLOBAL key from core_logic
        return True 
    raise HTTPException(
        status_code=401, # 401 Unauthorized for invalid credentials
        detail="Unauthorized: Invalid API Key. Please provide a valid 'Bearer YOUR_API_KEY' in the Authorization header."
    )

# Pydantic models for structured responses (unchanged from previous correct version)
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
        initialize_api_clients() 
        print("DEBUG: API clients initialized.")
                # Store initialized clients and key in app.state for access by request handlers
        app.state.global_client_openai = core_logic.global_client_openai
        app.state.global_client_anthropic = core_logic.global_client_anthropic
        app.state.global_mock_api_key = core_logic.global_mock_api_key

    except Exception as e:
        print(f"CRITICAL ERROR: Failed to initialize API clients: {e}")
        # Re-raise the exception to prevent the application from starting if clients can't be initialized
        raise HTTPException(status_code=500, detail=f"API Client Initialization Failed: {e}") 
        
    # Rest of the database initialization check (unchanged)
    try:
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
        if collection.count() == 0:
            print("Database empty. Initializing with 100 mock resumes...")
            initialize_database(100) 
            print("Database initialization complete.")
        else:
            print(f"Database already initialized with {collection.count()} resumes.")
    except Exception as e:
        print(f"Error during database startup check/initialization: {e}")
        # In a production system, you'd want more robust error handling or health checks here.


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
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
        if collection.count() == 0:
            raise HTTPException(status_code=500, detail="Database not initialized. Please ensure the API server started correctly.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database access error: {e}. Please ensure the API server started correctly.")

    try:
        result_data = perform_claude_search_with_tool(
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

# --- Root endpoint for health check or basic info --- (unchanged)
@app.get("/", summary="API Root / Health Check")
async def read_root():
    """A simple health check endpoint."""
    return {"message": "Talent Search MCP API is running. Go to /docs for API documentation."}

# You can run this API server from your terminal using:
# uvicorn api_server:app --reload --port 8000