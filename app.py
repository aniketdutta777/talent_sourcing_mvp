import streamlit as st
import requests
from streamlit_oauth import OAuth2Component

# --- CONFIGURATION ---
API_URL = "https://web-production-97a15.up.railway.app/v1/search_candidates"
try:
    GOOGLE_CLIENT_ID = st.secrets["GOOGLE_CLIENT_ID"]
    GOOGLE_CLIENT_SECRET = st.secrets["GOOGLE_CLIENT_SECRET"]
except KeyError:
    st.error("Google credentials not found in secrets. Please configure them.")
    st.stop()
REDIRECT_URI = "http://localhost:8501"

# --- UI SETUP & STYLING ---
st.set_page_config(layout="centered", page_title="Lark | AI Talent Finder")

# --- NEW: DARK THEME-AWARE CSS ---
# This CSS is designed to complement Streamlit's native dark theme
st.markdown("""
<style>
    /* Style for the bordered containers (cards) */
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
        border: 1px solid #4F4F4F; /* A subtle border for dark theme */
        border-radius: 10px;
        padding: 20px;
        background-color: #1C1C1E; /* A slightly lighter shade than the default dark background */
    }
    /* Primary button styling */
    div[data-testid="stButton"] > button[kind="primary"] {
        background-color: #1a73e8;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.title(" Lark")
st.subheader("Find talent in your system or ours.")

# --- AUTHENTICATION ---
st.sidebar.header("Access Configuration")
user_api_key = st.sidebar.text_input("Enter your Product API Key", type="password")

if not user_api_key:
    st.info("Please enter your API Key in the sidebar to get started.")
    st.stop()

# --- MAIN APPLICATION FLOW ---
st.markdown("##### 1. Choose Your Data Source(s)")
col1, col2 = st.columns(2)

# Source 1: Lark's Database
with col1:
    with st.container():
        st.markdown("###### Lark's Database")
        st.checkbox("Search Lark's proprietary talent pool.", value=True, key="lark_db_selected")

# Source 2: Google Drive
with col2:
    with st.container():
        st.markdown("###### Your Own Database")
        is_google_connected = 'token' in st.session_state

        gdrive_selected = st.checkbox(
            "Search in my own database (via Google Drive)",
            key="gdrive_selected"
        )

        if gdrive_selected:
            oauth2 = OAuth2Component(
                client_id=GOOGLE_CLIENT_ID, client_secret=GOOGLE_CLIENT_SECRET,
                authorize_endpoint="https://accounts.google.com/o/oauth2/v2/auth",
                token_endpoint="https://oauth2.googleapis.com/token",
            )
            if not is_google_connected:
                result = oauth2.authorize_button(
                    name="Connect Google Drive", icon="https://www.google.com/favicon.ico",
                    redirect_uri=REDIRECT_URI, scope="https://www.googleapis.com/auth/drive.readonly",
                    key="google", use_container_width=True
                )
                if result:
                    st.session_state.token = result.get('token')
                    st.rerun()
            else:
                st.success("Google Drive is connected.")
                if st.button("Disconnect", use_container_width=True, type="secondary"):
                    del st.session_state.token
                    st.session_state.gdrive_selected = False
                    st.rerun()

# --- STEP 2: DEFINE SEARCH QUERY ---
st.markdown("##### 2. Describe the role you're hiring for")
with st.container():
    query = st.text_area(
        "Paste a job description or list the key requirements, skills, and experience.",
        "Senior Backend Engineer with 5+ years of Python and AWS experience, who has led a team in a FinTech company.",
        height=150, label_visibility="collapsed"
    )

# --- SEARCH BUTTON & LOGIC ---
# In app.py

if st.button("Find Matching Profiles", type="primary", use_container_width=True):
    # Determine the source based on checkbox selections
    source_selection = ""
    if st.session_state.lark_db_selected and st.session_state.get('gdrive_selected', False):
        source_selection = "Both"
    elif st.session_state.lark_db_selected:
        source_selection = "Lark's Database"
    elif st.session_state.get('gdrive_selected', False):
        source_selection = "Google Drive"

    if not source_selection:
        st.warning("Please select at least one data source to search.")
    else:
        with st.spinner("🧠 Analyzing resumes..."):
            try:
                # --- THIS IS THE CORRECTED PART ---
                # Get the token from the session state
                token = st.session_state.get('token')
                
                # Parse folder URLs from the text area
                folder_ids = [url.strip() for url in google_drive_folders.split('\n') if url.strip()]
                
                # Build the complete payload, including the token
                payload = {
                    "query": query,
                    "source": source_selection,
                    "num_results": 7,
                    "google_drive_folder_ids": folder_ids,
                    "google_auth_token": token # Send the token to the backend
                }
                # --- END CORRECTION ---

                headers = {"Authorization": f"Bearer {user_api_key}"}
                response = requests.post(API_URL, json=payload, headers=headers)
                response.raise_for_status()
                api_response_data = response.json()

                st.markdown("---")
                st.subheader("AI Analysis & Top Recommendations")
                analysis = api_response_data.get("analysis_data", {})
                
                if analysis and analysis.get('candidates'):
                    st.markdown(f"**Overall Summary:** {analysis.get('overall_summary', 'No summary available.')}")
                    st.markdown("---")
                    for i, candidate in enumerate(analysis.get('candidates', [])):
                        st.markdown(f"**{i+1}. {candidate.get('name', 'N/A')}**")
                        contact_info = candidate.get('contact_information', {})
                        st.markdown(f"**Contact:** {contact_info.get('email', 'N/A')} | {contact_info.get('phone', 'N/A')}")
                        st.markdown(f"**Resume:** [Link to PDF]({candidate.get('resume_pdf_url', '#')})")
                        with st.expander("See AI Summary"):
                            st.markdown(f"{candidate.get('summary', 'No summary provided.')}")
                        st.markdown("---")
                    st.markdown(f"**Overall Recommendation:** {analysis.get('overall_recommendation', 'N/A')}")
                else:
                    st.warning("No matching candidates were found for your query.")
                    summary = analysis.get('overall_summary', 'The AI could not find any matching profiles that fit the criteria.')
                    st.markdown(f"**AI Summary:** {summary}")

            except Exception as e:
                st.error(f"An error occurred: {e}")