from fastapi import FastAPI, HTTPException, Depends, Request # <-- ADD Request IMPORT
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import core_logic

# --- Rate Limiting Setup ---
limiter = Limiter(key_func=get_remote_address)

# Initialize FastAPI app
app = FastAPI(
    title="Lark Talent API",
    description="An API exposing LLM-powered talent search over a proprietary resume database.",
    version="1.0.0"
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- In-Memory Key Store ---
VALID_API_KEYS = {
    "a885598e-1a1b-463b-9c23-acc4e4275330",
    "05f84d16-3908-4da3-8804-5c455d4acc1c"
}

# --- Authentication ---
security_scheme = HTTPBearer()

def get_api_key(credentials: HTTPAuthorizationCredentials = Depends(security_scheme)):
    if credentials.credentials in VALID_API_KEYS:
        return credentials.credentials
    raise HTTPException(status_code=401, detail="Unauthorized: Invalid or missing API Key.")

# --- Pydantic Models ---
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
    status: str
    analysis_data: AnalysisResponse

class SearchRequest(BaseModel):
    query: str
    source: str = Field(..., description="The data source to search.")
    num_results: int = Field(7, description="Number of top candidates to retrieve.")

# --- Startup Event ---
@app.on_event("startup")
async def startup_event():
    print("\n--- FastAPI Startup: Initializing ---")
    core_logic.initialize_api_clients()
    core_logic.initialize_database(100)
    print("--- Startup Complete ---")

# --- API Endpoints ---
@app.get("/", summary="API Root / Health Check")
async def read_root():
    return {"message": "Lark Talent API is running."}

@app.post("/v1/search_candidates", summary="Search for candidates", response_model=SearchResponseWrapper)
@limiter.limit("20/minute")
# --- UPDATED FUNCTION SIGNATURE ---
async def search_candidates(request: Request, search_request: SearchRequest, api_key: str = Depends(get_api_key)):
    print(f"\n--- API Call: /v1/search_candidates (Key ending with '...{api_key[-4:]}') ---")
    # --- UPDATED VARIABLE NAME ---
    print(f"Received query: '{search_request.query}' for source: '{search_request.source}'")

    try:
        # --- UPDATED VARIABLE NAMES ---
        result_data = core_logic.perform_claude_search_with_tool(
            user_query=search_request.query,
            num_profiles_to_retrieve=search_request.num_results
        )

        if result_data.get("status") == "success":
            usage = result_data.get("usage", {})
            print(f"COST_LOG: key='{api_key}' usage={usage}")
            return SearchResponseWrapper(
                status="success",
                analysis_data=result_data["analysis_data"]
            )
        else:
            raise HTTPException(status_code=500, detail=result_data.get("message", "Unknown LLM error"))
    except Exception as e:
        print(f"Error during API call: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")