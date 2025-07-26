# LLM-Powered Talent Search API & Client

This project is a full-stack application that demonstrates how to build an intelligent talent sourcing platform using a modern Python technology stack. It leverages Large Language Models (LLMs) for understanding natural language queries and analyzing candidate data from a specialized vector database.

## 🌟 Overview

The system allows a hiring manager to enter a complex, natural language query (e.g., "Find me a senior backend developer with Python and AWS experience who has led a team"). The backend processes this query, finds the most relevant candidates from a database of resumes, and uses an LLM to provide a detailed analysis, justification, and recommendation for each candidate.

This repository contains two main components:
1.  **`api_server.py`**: A FastAPI backend that serves the core logic through a REST API.
2.  **`app.py`**: A Streamlit web client that provides a user interface for interacting with the API.

---

## 🏛️ Architecture

The application is built on a decoupled **3-Tier Architecture**, ensuring scalability and separation of concerns.

1.  **Presentation Tier (Client):** The `app.py` file is a **Streamlit** application that provides the user interface. It is responsible for sending user queries and the API key to the backend and rendering the results in a user-friendly format.

2.  **Application Tier (Server):** The `api_server.py` file is a **FastAPI** web server.
    * It exposes a secure REST API endpoint (`/v1/search_candidates`).
    * It handles authentication using HTTP Bearer tokens.
    * It delegates all business logic to the `core_logic` module.
    * [cite_start]It uses `uvicorn` as its ASGI server for high performance.

3.  **Data & Services Tier:** This composite layer includes:
    * **`core_logic.py`**: The "brain" of the application. This module orchestrates all the backend tasks, including calls to the LLMs and the database.
    * **Vector Database**: We use **ChromaDB** to store resume data and their corresponding vector embeddings for efficient semantic search.
    * **External AI Services**:
        * **OpenAI API**: Used to generate vector embeddings from text.
        * **Anthropic (Claude) API**: Used for advanced reasoning. [cite_start]It acts as the primary "analyst," using a defined tool to search the database and then synthesize the findings.

### ⚙️ Technology Stack

* [cite_start]**Backend API:** FastAPI, Uvicorn [cite: 1, 2]
* **Frontend Client:** Streamlit 
* [cite_start]**Vector Database:** ChromaDB 
* [cite_start]**AI Services:** OpenAI (Embeddings), Anthropic (Reasoning) 
* **Data Handling:** Pydantic, Pandas 

---

## 🔄 Data Flow (How It Works)

1.  A user enters a query in the **Streamlit UI**.
2.  The Streamlit client sends a POST request, including the query and a security token, to the **/v1/search_candidates** endpoint on the **FastAPI server**.
3.  The FastAPI server authenticates the request and calls the `perform_claude_search_with_tool` function from the `core_logic` module.
4.  This function sends the user's query to the **Anthropic (Claude) API**, providing it with a schema for the `resume_search_tool`.
5.  Claude determines that it needs to use this tool and sends a request back to our `core_logic`.
6.  The `resume_search_tool` function is executed:
    a. It first sends the query text to the **OpenAI API** to get a vector embedding.
    b. It uses this embedding to perform a similarity search against the **ChromaDB** vector database.
7.  ChromaDB returns the most relevant candidate profiles (raw text).
8.  These profiles are sent back to Claude as the result of its tool execution.
9.  Claude analyzes the profiles in the context of the original query and generates a final, structured JSON response containing a summary, candidate analysis, and recommendations.
10. This JSON is passed back through the FastAPI server to the Streamlit client.
11. The Streamlit client parses the JSON and displays the information neatly for the user.

---

## 🚀 Setup and Installation

### Prerequisites
* Python 3.9+
* An `.env` file with your API keys.

### 1. Environment Variables

Create a `.env` file in the root directory and add your secret keys. The application code loads these variables on startup.