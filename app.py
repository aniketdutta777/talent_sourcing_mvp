import streamlit as st
import requests 
import json 

# --- PRODUCT CONFIGURATION ---
# Define the URL of your deployed FastAPI server. This is the only configuration needed.
API_URL = "https://web-production-97a15.up.railway.app/v1/search_candidates"

# --- USER INTERFACE SETUP ---
st.set_page_config(layout="wide", page_title="AI Talent Finder")

st.title("🌟 AI Talent Finder")
st.subheader("Find the perfect candidates by describing your ideal hire in plain English.")

# --- SIDEBAR FOR USER INPUT AND INSTRUCTIONS ---
st.sidebar.header("Access Configuration")
user_api_key = st.sidebar.text_input(
    "Enter the API Key we provided", 
    type="password",
    help="You should have received an API Key to access this service."
)

st.sidebar.header("How to Use")
st.sidebar.info(
    """
    1.  Enter your API Key in the 'Access Configuration' box above.
    2.  In the main text area, describe the role and skills you're looking for.
    3.  Adjust the slider to control how many profiles the AI should analyze.
    4.  Click 'Find Matching Profiles' to get your results!
    """
)


# --- MAIN CONTENT AREA ---
st.header("1. Describe Your Ideal Candidate")
query = st.text_area(
    "Enter your hiring manager query (e.g., 'Find a Senior Software Engineer with expertise in Python, AWS, and Machine Learning, with experience in FinTech. Must have led a team.')",
    "Looking for a Mid-level Marketing Manager with B2B SaaS experience, strong in lead generation and content strategy. Can you find some candidates for me?",
    height=150
)

st.header("2. Set Your Search Options")
num_profiles_to_retrieve = st.slider(
    "Number of profiles for the AI to analyze:",
    min_value=3, max_value=15, value=7,
    help="This controls how many top candidates the AI will retrieve for its detailed analysis."
)

if st.button("Find Matching Profiles", type="primary"):
    # --- INPUT VALIDATION ---
    if not user_api_key:
        st.error("⚠️ Please enter your API Key in the sidebar to connect.")
    elif not query:
        st.warning("⚠️ Please enter a description of the candidate you are looking for.")
    else:
        # --- API CALL LOGIC ---
        with st.spinner("🧠 Analyzing resumes... This may take a moment."):
            try:
                # Prepare headers with the API key from the user's input
                headers = {
                    "Authorization": f"Bearer {user_api_key}", 
                    "Content-Type": "application/json" 
                }

                # Make the HTTP POST request to your FastAPI server
                response = requests.post(
                    API_URL, 
                    json={
                        "query": query, 
                        "num_results": num_profiles_to_retrieve
                    },
                    headers=headers
                )
                response.raise_for_status() # Raise an exception for bad responses (4xx or 5xx)
                
                api_response_data = response.json()
                
                # --- DISPLAY RESULTS ---
                if api_response_data.get("status") == "success":
                    st.success("✅ Analysis Complete!")
                    st.markdown("### AI Analysis & Top Recommendations")
                    
                    analysis = api_response_data.get("analysis_data", {}) 
                    if analysis:
                        st.markdown(f"**Overall Summary:** {analysis.get('overall_summary', 'No summary provided.')}")
                        st.markdown("---")
                        
                        # Loop through and display each recommended candidate
                        for i, candidate in enumerate(analysis.get('candidates', [])):
                            st.markdown(f"**{i+1}. {candidate.get('name', 'N/A')}** (Confidence: {candidate.get('confidence_score', 'N/A')})")
                            st.markdown(f"**Job Title:** {candidate.get('job_title', 'N/A')}")
                            with st.expander("See AI Justification"):
                                st.markdown(f"{candidate.get('justification', 'N/A')}")
                            st.markdown("") 
                        
                        st.markdown("---")
                        st.markdown(f"**Final Recommendation:** {analysis.get('overall_recommendation', 'N/A')}")
                    else:
                        st.warning("The AI provided a response, but it did not contain structured analysis data.")

                else:
                    # Handle cases where the API returns a success status but an error message
                    st.error(f"An error occurred: {api_response_data.get('message', 'Unknown error from API.')}")
                    st.code(f"Raw LLM Output (for debugging):\n{api_response_data.get('raw_llm_output', 'N/A')}")

            # --- USER-FRIENDLY ERROR HANDLING ---
            except requests.exceptions.ConnectionError:
                st.error(f"Connection Error: Could not connect to the Talent Finder service. Please try again later.")
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 401:
                    st.error("Authentication Error: The API Key you provided is invalid. Please check the key and try again.")
                else:
                    st.error(f"An error occurred while communicating with the service (Error {e.response.status_code}). Please try again.")
            except json.JSONDecodeError:
                st.error("Error: Received an invalid response from the server. The service may be experiencing issues.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

st.markdown("---")
st.markdown("<div style='text-align: center;'>Powered by Advanced AI</div>", unsafe_allow_html=True)