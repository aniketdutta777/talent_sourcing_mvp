from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.exceptions import RequestValidationError
from fastapi.responses import PlainTextResponse
from typing import Optional
import core_logic

# --- Rate Limiting Setup ---
limiter = Limiter(key_func=get_remote_address)

# Initialize FastAPI app
app = FastAPI(
    title="Lark Talent API",
    description="An API exposing LLM-powered talent search over a proprietary resume database.",
    version="1.0.0"
)

# Custom exception handler for 422 errors to get detailed logs
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print(f"!!! DETAILED VALIDATION ERROR !!!\n{exc}\n!!! END OF ERROR !!!")
    return PlainTextResponse(str(exc), status_code=422)

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

# --- Pydantic Models (Unchanged) ---
class ContactInfo(BaseModel):
    email: str
    phone: str

class Candidate(BaseModel):
    name: str
    contact_information: ContactInfo
    summary: str
    resume_pdf_url: str

class AnalysisResponse(BaseModel):
    overall_summary: str
    candidates: list[Candidate]
    overall_recommendation: str

class SearchResponseWrapper(BaseModel):
    status: str
    analysis_data: AnalysisResponse

class SearchRequest(BaseModel):
    query: str
    source: str
    num_results: int = 7
    google_drive_folder_ids: list[str] = Field([])
    google_auth_token: Optional[dict] = Field(None)

# --- CORRECTED Startup Event ---
@app.on_event("startup")
async def startup_event():
    """
    Initializes all clients and databases in a controlled sequence during startup.
    """
    print("\n--- FastAPI Startup: Initializing ---")
    core_logic.initialize_api_clients()
    core_logic.initialize_chroma_client() # <-- MOVED DB INITIALIZATION HERE
    core_logic.initialize_database(100)
    print("--- Startup Complete ---")

# --- API Endpoints ---
@app.get("/", summary="API Root / Health Check")
async def read_root():
    return {"message": "Lark Talent API is running."}

@app.post("/v1/search_candidates", summary="Search for candidates", response_model=SearchResponseWrapper)
@limiter.limit("20/minute")
async def search_candidates(request: Request, search_request: SearchRequest, api_key: str = Depends(get_api_key)):
    try:
        result_data = core_logic.perform_claude_search_with_tool(
            user_query=search_request.query,
            num_profiles_to_retrieve=search_request.num_results,
            source=search_request.source,
            folder_ids=search_request.google_drive_folder_ids,
            user_id=api_key,
            token=search_request.google_auth_token
        )
        if result_data.get("status") == "success":
            usage = result_data.get("usage", {})
            print(f"COST_LOG: key='{api_key}' usage={usage}")
            return SearchResponseWrapper(status="success", analysis_data=result_data["analysis_data"])
        else:
            raise HTTPException(status_code=500, detail=result_data.get("message", "Unknown LLM error"))
    except Exception as e:
        print(f"Error during API call: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")