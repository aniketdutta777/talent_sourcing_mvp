# LLM-Powered Talent Sourcing MVP

This is a Minimum Viable Product (MVP) demonstrating how Large Language Models (LLMs), specifically Anthropic's Claude, can be used to intelligently query a proprietary resume database for talent sourcing. It leverages OpenAI's embedding models for semantic search and Claude for detailed candidate analysis and ranking.

---

## Features:
-   Initializes a mock resume database (for demonstration).
-   Allows natural language queries for candidate profiles.
-   Uses Claude's tool-use capabilities to search the database.
-   Provides AI-generated analysis, confidence scores, and recommendations.

---

## Setup and Running the Application

### Prerequisites:
-   **Python 3.9+:** [Download from python.org](https://www.python.org/downloads/) (recommended to install via Homebrew on macOS).
-   **Git:** [Install Git](https://git-scm.com/downloads) (usually pre-installed on macOS/Linux).
-   **VS Code (Recommended IDE):** [Download VS Code](https://code.visualstudio.com/) and install the Python Extension.
-   **OpenAI API Key:** For embedding models (costs apply). Get it from [platform.openai.com](https://platform.openai.com/).
-   **Anthropic API Key:** For Claude LLM (costs apply). Get it from [console.anthropic.com](https://console.anthropic.com/).

### Installation Steps:

1.  **Clone the Repository:**
    Open your Terminal/Command Prompt and run:
    ```bash
    git clone [https://github.com/aniketdutta777/talent_sourcing_mvp.git](https://github.com/aniketdutta777/talent_sourcing_mvp.git)
    cd talent_sourcing_mvp
    ```
    *(Make sure to copy the correct HTTPS URL from your GitHub repo's green "Code" button.)*

2.  **Set up Virtual Environment:**
    ```bash
    python3 -m venv venv
    # On macOS/Linux:
    source venv/bin/activate
    # On Windows:
    # .\venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up API Keys (`.env` file):**
    * Create a new file named `.env` in the root of your `talent_sourcing_mvp` directory.
    * Add your API keys to this file (replace placeholders with your actual keys):
        ```
        OPENAI_API_KEY="your_openai_api_key_here"
        ANTHROPIC_API_KEY="your_anthropic_api_key_here"
        ```
    * **CRITICAL:** Never share your `.env` file publicly. It is already included in `.gitignore`.

### Running the App:

1.  **Start the Streamlit Application:**
    ```bash
    streamlit run app.py
    ```
2.  Open your browser to `http://localhost:8501/`.
3.  **Initialize the Database:** In the app's sidebar, click "Initialize Database with X Resumes" (do this once on first run or after clearing).
4.  **Query the Database:** Enter your hiring manager query and click "Find Matching Profiles".

---

## Project Structure:
-   `app.py`: Streamlit web interface.
-   `core_logic.py`: Contains the core logic for database initialization, embedding, resume search (as a tool), and Claude's tool-use orchestration.
-   `requirements.txt`: Lists all Python dependencies.
-   `.env`: (Ignored by Git) Stores sensitive API keys.
-   `mock_resume_database/`: (Ignored by Git) Local storage for ChromaDB data and raw mock resumes.

---

## Future Enhancements (Ideas):
-   Robust resume parsing (PDF/DOCX).
-   Multi-tenancy for multiple clients.
-   Scalable cloud deployment.
-   Advanced UI/dashboards.
-   Integration with Applicant Tracking Systems (ATS).