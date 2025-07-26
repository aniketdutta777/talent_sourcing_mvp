from fastapi import FastAPI, HTTPException, Depends, Security # Added Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials # Added HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field 
import uvicorn
import json 
import os 
from dotenv import load_dotenv # Added load_dotenv

# Load environment variables (API_SECRET_KEY)
load_dotenv()
# This key is used by the API server to VERIFY incoming requests
YOUR_MOCK_API_KEY_SERVER = os.getenv("YOUR_MOCK_API_KEY") 

# --- CRITICAL DEBUG PRINT for API Server ---
if not YOUR_MOCK_API_KEY_SERVER:
    print("CRITICAL ERROR (api_server.py): YOUR_MOCK_API_KEY is EMPTY or not loaded! Check .env file.")
else:
    print(f"DEBUG (api_server.py): YOUR_MOCK_API_KEY_SERVER loaded: '{YOUR_MOCK_API_KEY_SERVER}'")
# --------------------------------------------


# Import the core logic functions from core_logic.py
# Note: core_logic.py also loads YOUR_MOCK_API_KEY, but this is the server's definitive check
from core_logic import perform_claude_search_with_tool, initialize_database, chroma_client, COLLECTION_NAME 

# Initialize FastAPI app
app = FastAPI(
    title="Talent Search MCP API (Mock)",
    description="A mock API exposing LLM-powered talent search over a proprietary resume database. This simulates your productized database as an MCP server.",
    version="1.0.0"
)

# --- Authentication Setup ---
# This defines an HTTP Bearer Token scheme for authentication
security_scheme = HTTPBearer()

# Dependency function to check the API key
# This function runs before any request hits the endpoint
def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security_scheme)):
    # print(f"DEBUG AUTH: Received credential: '{credentials.credentials}'") # You can uncomment this to debug what the server receives
    # print(f"DEBUG AUTH: Expected key: '{YOUR_MOCK_API_KEY_SERVER}'") # And what it expects

    if not YOUR_MOCK_API_KEY_SERVER:
        raise HTTPException(status_code=500, detail="Server Error: API Key not configured internally.")

    if credentials.credentials == YOUR_MOCK_API_KEY_SERVER:
        return True # API key is valid
    raise HTTPException(
        status_code=401, # 401 Unauthorized for invalid credentials
        detail="Unauthorized: Invalid API Key. Please provide a valid 'Bearer YOUR_API_KEY' in the Authorization header."
    )

# Pydantic model to define the structure of a single candidate in the response
class Candidate(BaseModel):
    id: str = Field(..., description="Unique ID of the candidate.")
    name: str = Field(..., description="Name of the candidate.")
    confidence_score: str = Field(..., description="Confidence score for the candidate's fit (e.g., 'X/5' or 'Y%').")
    job_title: str = Field(..., description="Job title of the candidate.") 
    justification: str = Field(..., description="Detailed justification for why this candidate is a good fit.")

# Pydantic model to define the structure of the overall analysis response
class AnalysisResponse(BaseModel):
    overall_summary: str = Field(..., description="Overall summary of the candidate evaluation.")
    candidates: list[Candidate] = Field(..., description="List of analyzed candidates with their details.")
    overall_recommendation: str = Field(..., description="Overall recommendation for top candidates.")

# New Pydantic model for the overall API response structure (what the API sends back)
class SearchResponseWrapper(BaseModel):
    status: str = Field(..., description="Status of the API request (e.g., 'success', 'error').")
    analysis_data: AnalysisResponse = Field(..., description="Structured analysis result from the LLM.")

# Pydantic model for incoming search requests
class SearchRequest(BaseModel):
    query: str
    num_results: int = Field(7, description="Number of top candidates to retrieve and analyze. Default is 7.")

@app.on_event("startup")
async def startup_event():
    """
    Event handler that runs automatically when the FastAPI application starts.
    It checks if the mock database is initialized. If not, it initializes it.
    """
    print("\n--- FastAPI Startup: Checking Database Initialization ---")
    try:
        collection_check = chroma_client.get_collection(name=COLLECTION_NAME)
        if collection_check.count() == 0:
            print("Database empty. Initializing with 100 mock resumes...")
            initialize_database(100) 
            print("Database initialization complete.")
        else:
            print(f"Database already initialized with {collection_check.count()} resumes.")
    except Exception as e:
        print(f"Error during database startup check/initialization: {e}")
        # In a production system, you'd want more robust error handling or health checks here.


@app.post("/v1/search_candidates", 
          summary="Search for candidates using LLM intelligence", 
          response_model=SearchResponseWrapper, # Specify the response model for structured output
          dependencies=[Depends(verify_api_key)]) # Add authentication dependency
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
        # Call your core LLM orchestration logic (which uses Claude's tool use internally)
        # This function now returns a structured dictionary
        result_data = perform_claude_search_with_tool(
            user_query=request.query, 
            num_profiles_to_retrieve=request.num_results
        )
        
        if result_data["status"] == "success" and "analysis_data" in result_data:
            # Return the parsed JSON directly, matching the AnalysisResponse model
            # Pydantic will validate this and convert it to the expected AnalysisResponse object
            return SearchResponseWrapper(
                status=result_data["status"],
                analysis_data=result_data["analysis_data"]
            )
        else:
            # Handle errors from perform_claude_search_with_tool
            raise HTTPException(status_code=500, detail=f"LLM analysis failed: {result_data.get('message', 'Unknown error')}. Raw output: {result_data.get('raw_llm_output', 'N/A')}")
            
    except Exception as e:
        print(f"Error during LLM analysis in API: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error during LLM analysis: {e}")

# --- Root endpoint for health check or basic info ---
@app.get("/", summary="API Root / Health Check")
async def read_root():
    """A simple health check endpoint."""
    return {"message": "Talent Search MCP API is running. Go to /docs for API documentation."}

# You can run this API server from your terminal using:
# uvicorn api_server:app --reload --port 8000