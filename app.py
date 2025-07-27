import streamlit as st
import requests
from streamlit_oauth import OAuth2Component

# --- CONFIGURATION ---
API_URL = "https://web-production-97a15.up.railway.app/v1/search_candidates"
# For local testing, Streamlit will read from your .streamlit/secrets.toml file
# For deployment, it will read from the secrets you pasted in the App settings
try:
    GOOGLE_CLIENT_ID = st.secrets["GOOGLE_CLIENT_ID"]
    GOOGLE_CLIENT_SECRET = st.secrets["GOOGLE_CLIENT_SECRET"]
except KeyError:
    st.error("Google credentials not found in secrets. Please configure them.")
    st.stop()
# This must match what you've set in the Google Cloud Console for local testing
REDIRECT_URI = "http://localhost:8501" 

# --- UI SETUP ---
st.set_page_config(layout="wide", page_title="AI Talent Finder")
st.title("🌟 AI Talent Finder")

# --- AUTHENTICATION & MAIN APP LOGIC ---
st.sidebar.header("Access Configuration")
user_api_key = st.sidebar.text_input("Enter your Product API Key", type="password")

if not user_api_key:
    st.info("Please enter your API Key in the sidebar to get started.")
else:
    # --- MAIN APP UI (shown only after API Key is entered) ---
    st.subheader("1. Configure Your Data Source")

    # --- GOOGLE DRIVE CONNECTION (ON/OFF SWITCH) ---
    col1, col2 = st.columns([1, 3])
    with col1:
        st.write("##### Google Drive Connection")
        oauth2 = OAuth2Component(
            client_id=GOOGLE_CLIENT_ID,
            client_secret=GOOGLE_CLIENT_SECRET,
            authorize_endpoint="https://accounts.google.com/o/oauth2/v2/auth",
            token_endpoint="https://oauth2.googleapis.com/token",
        )
        if 'token' not in st.session_state:
            result = oauth2.authorize_button(
                name="Connect Google Drive",
                icon="https://www.google.com/favicon.ico",
                redirect_uri=REDIRECT_URI,
                scope="https://www.googleapis.com/auth/drive.readonly",
                key="google", use_container_width=True
            )
            if result:
                st.session_state.token = result.get('token')
                st.rerun()
        else:
            st.success("Google Drive Connected!")
            if st.button("Disconnect", use_container_width=True):
                del st.session_state.token
                st.rerun()

    # --- DATA SOURCE SELECTION ---
    with col2:
        st.write("##### Search Target")
        search_options = ["My Database"]
        if 'token' in st.session_state:
            search_options.extend(["Google Drive", "Both"])
        source_selection = st.radio(
            "Where should the AI search for profiles?",
            options=search_options, horizontal=True, label_visibility="collapsed"
        )
    
    st.markdown("---")

    # --- SEARCH QUERY AREA ---
    st.subheader("2. Describe Your Ideal Candidate")
    query = st.text_area(
        "Enter your hiring manager query...",
        "Find a Senior Software Engineer with expertise in Python and AWS.",
        height=150, label_visibility="collapsed"
    )

    if st.button("Find Matching Profiles", type="primary"):
        with st.spinner("🧠 Analyzing resumes... This may take a moment."):
            try:
                headers = {"Authorization": f"Bearer {user_api_key}"}
                payload = {
                    "query": query,
                    "source": source_selection,
                    "num_results": 7
                }
                response = requests.post(API_URL, json=payload, headers=headers)
                response.raise_for_status()
                api_response_data = response.json()

                st.subheader("AI Analysis & Top Recommendations")
                analysis = api_response_data.get("analysis_data", {})
                if analysis:
                    st.markdown(f"**Overall Summary:** {analysis.get('overall_summary', 'No summary.')}")
                    st.markdown("---")
                    for i, candidate in enumerate(analysis.get('candidates', [])):
                        st.markdown(f"**{i+1}. {candidate.get('name', 'N/A')}**")
                        contact_info = candidate.get('contact_information', {})
                        st.markdown(f"**Contact:** {contact_info.get('email', 'N/A')} | {contact_info.get('phone', 'N/A')}")
                        st.markdown(f"**Resume:** [Link to PDF]({candidate.get('resume_pdf_url', '#')})")
                        with st.expander("See AI Summary"):
                            st.markdown(f"{candidate.get('summary', 'No summary.')}")
                        st.markdown("---")
                    st.markdown(f"**Overall Recommendation:** {analysis.get('overall_recommendation', 'N/A')}")
                else:
                    st.warning("Analysis data not found in the response.")
            except Exception as e:
                st.error(f"An error occurred: {e}")