import streamlit as st
import requests 
import json 
import os 
from dotenv import load_dotenv # Added load_dotenv for app.py

# Load environment variables (API_SECRET_KEY for client)
load_dotenv()
# This key is needed by the Streamlit client to authenticate with the FastAPI server
YOUR_MOCK_API_KEY = os.getenv("YOUR_MOCK_API_KEY") 

# --- DEBUG PRINT: VERY IMPORTANT TO SEE THIS OUTPUT ---
if not YOUR_MOCK_API_KEY:
    st.sidebar.error("CRITICAL DEBUG (app.py): YOUR_MOCK_API_KEY is EMPTY or not loaded! Check .env file.")
    print("CRITICAL DEBUG (app.py): YOUR_MOCK_API_KEY is EMPTY or not loaded! Check .env file.") # Print to terminal too
else:
    # Display partially in UI for confirmation, without exposing the full key
    st.sidebar.success(f"API Key loaded. Ready to connect. (Key starts with '{YOUR_MOCK_API_KEY[:4]}...')") 
    print(f"DEBUG (app.py): YOUR_MOCK_API_KEY loaded: '{YOUR_MOCK_API_KEY}'")
# ------------------------------------------------------------------------------

# Define the URL of your new FastAPI server
API_URL = "https://web-production-97a15.up.railway.app"

st.set_page_config(layout="wide", page_title="LLM Talent Sourcing MVP (API Client)")

st.title("🌟 LLM-Powered Talent Sourcing (API Client MVP)")
st.subheader("Your Streamlit app now queries your local MCP API with authentication.")

st.sidebar.header("1. API Server Status (Requires `api_server.py` running)")
st.sidebar.markdown(f"**API Endpoint:** `{API_URL}`")
st.sidebar.markdown(f"**Your API Key (from .env):** `{YOUR_MOCK_API_KEY}`") # Display for debugging
st.sidebar.markdown("---")
st.sidebar.info("To use this app, you MUST first run the API server in a separate terminal:")
st.sidebar.code("uvicorn api_server:app --reload --port 8000")
st.sidebar.markdown("The API server will automatically initialize the database on startup if it's empty.")
st.sidebar.markdown("---")
st.sidebar.warning("If you get connection errors, make sure `api_server.py` is running!")


st.header("2. Query the Resume Database via API")
    
query = st.text_area(
    "Enter your hiring manager query (e.g., 'Find a Senior Software Engineer with expertise in Python, AWS, and Machine Learning, with experience in FinTech. Must have led a team.')",
    "Looking for a Mid-level Marketing Manager with B2B SaaS experience, strong in lead generation and content strategy. Can you find some candidates for me?" 
)

num_profiles_to_retrieve = st.slider(
    "Number of top similar profiles for the AI to retrieve and analyze:",
    min_value=3, max_value=15, value=7,
    help="This controls how many candidates Claude's search tool will retrieve for its analysis."
)

if st.button("Find Matching Profiles via API", type="primary"):
    if not query:
        st.warning("Please enter your search query.")
    elif not YOUR_MOCK_API_KEY: # This check directly uses the loaded key
        st.error("API Key is missing. Please set `YOUR_MOCK_API_KEY` in your `.env` file and restart the app.")
    else:
        with st.spinner("Sending query to API... Waiting for LLM analysis..."):
            try:
                # Add the API key to the request headers
                headers = {
                    "Authorization": f"Bearer {YOUR_MOCK_API_KEY}", # <--- THIS IS THE AUTH HEADER
                    "Content-Type": "application/json" # Good practice for POST requests
                }

                # Make an HTTP POST request to your FastAPI server
                response = requests.post(
                    API_URL, 
                    json={
                        "query": query, 
                        "num_results": num_profiles_to_retrieve
                    },
                    headers=headers # Pass the headers with the API key
                )
                response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
                
                api_response_data = response.json()
                
                if api_response_data.get("status") == "success":
                    st.markdown("### AI's Analysis & Top Recommendations (from API):")
                    
                    # Display structured JSON data nicely
                    # 'analysis_data' is the key expected from perform_claude_search_with_tool's successful return
                    analysis = api_response_data.get("analysis_data", {}) 
                    if analysis:
                        st.markdown(f"**Overall Summary:** {analysis.get('overall_summary', 'No summary provided.')}")
                        st.markdown("---")
                        
                        for i, candidate in enumerate(analysis.get('candidates', [])):
                            st.markdown(f"**{i+1}. {candidate.get('name', 'N/A')}** (Confidence: {candidate.get('confidence_score', 'N/A')})")
                            st.markdown(f"* Job Title: {candidate.get('job_title', 'N/A')}")
                            st.markdown(f"* Justification: {candidate.get('justification', 'N/A')}")
                            st.markdown("") # Add a blank line for spacing
                        
                        st.markdown("---")
                        st.markdown(f"**Overall Recommendation:** {analysis.get('overall_recommendation', 'N/A')}")
                    else:
                        st.warning("No structured analysis data found in API response.")

                    st.success("Search complete!")
                else:
                    # Improved error display if API returns error status
                    st.error(f"API returned an error: {api_response_data.get('message', 'Unknown error')}. Raw Output: {api_response_data.get('raw_llm_output', 'N/A')}")

            except requests.exceptions.ConnectionError:
                st.error("Connection Error: Is the `api_server.py` running? Please start it using the command in the sidebar.")
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 401:
                    st.error("Authentication Error (401): Invalid API Key. Please check `YOUR_MOCK_API_KEY` in your `.env` file and restart both apps.")
                elif e.response.status_code == 403: # Specifically catch 403 for Forbidden
                    st.error("Access Forbidden (403): Your API Key might be valid but not authorized for this specific action. This often indicates a mismatch. Check your API server logs.")
                else:
                    st.error(f"HTTP Error: {e.response.status_code} - {e.response.text}. Detail: {e.response.json().get('detail', 'No detail provided')}")
            except json.JSONDecodeError:
                st.error("Error: Could not decode JSON response from API. Check API server logs for valid JSON output.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

st.markdown("---")
st.markdown("💡 *This MVP demonstrates your Streamlit app acting as a client to your local MCP API. This is the foundation for allowing other LLMs/apps to connect!*")